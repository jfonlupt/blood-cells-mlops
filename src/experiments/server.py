import os
import time
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 -- @bentoml.api lit Path au runtime via Annotated[]
from typing import Annotated

import bentoml
import jwt
import mlflow
import pandas as pd
import requests
import torch
from bentoml.validators import ContentType
from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.metrics import Accuracy, F1Score, Precision, Recall
from fastapi import FastAPI, Form
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.experiments import drift
from src.features.transform import load_image_rgb

# Ne pas hardcoder les secrets en production
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
USER = os.environ.get("BENTO_USER")
PASSWD = os.environ.get("BENTO_PASSWORD")
DESCRIPTORS = ["width", "height", "aspect_ratio", "brightness", "contrast", "sharpness"]
DESCRIPTORS += [f"hist_{c}_{i}" for c in ("r", "g", "b") for i in range(8)]
CHAMPION_CACHE_TTL_SECONDS = float(os.environ.get("CHAMPION_CACHE_TTL_SECONDS", "600"))

api_requests_total = bentoml.metrics.Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
)

api_request_duration_seconds = bentoml.metrics.Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint", "method", "status_code"],
)

predictions_by_category = bentoml.metrics.Counter(
    "predictions_by_category",
    "Number of predictions by category",
    ["category", "model"],
)

model_accuracy = bentoml.metrics.Gauge("model_accuracy", "Accuracy of the model", ["model"])
model_precision = bentoml.metrics.Gauge("model_precision", "Precision of the model", ["model"])
model_recall = bentoml.metrics.Gauge("model_recall", "Recall of the model", ["model"])
model_f1_score = bentoml.metrics.Gauge("model_f1_score", "F1 score of the model", ["model"])

model_descriptor_columns_drifted_value = bentoml.metrics.Gauge(
    "model_descriptor_columns_drifted_value", "Number of descriptor column drifted", ["model"]
)

model_descriptor_columns_drifted_share = bentoml.metrics.Gauge(
    "model_descriptor_columns_drifted_share", "Share of descriptor column drifted", ["model"]
)

model_descriptor_drift = bentoml.metrics.Gauge(
    "model_descriptor_drift", "Drift for model descriptor colomns", ["model", "column"]
)

model_embeddings_drift = bentoml.metrics.Gauge(
    "model_embeddings_drift", "Drift for model embeddings", ["model"]
)

model_drift_detected = bentoml.metrics.Gauge(
    "model_drift_detected",
    "1 si une dérive (descripteurs ou embeddings) est détectée, 0 sinon",
    ["model"],
)


def _set_drift_gauge(gauge, model: str, drift_result: drift.DriftMetricsResult) -> None:
    gauge.labels(model=model).set(1.0 if drift.is_drift_detected(drift_result) else 0.0)


class Credentials(BaseModel):
    username: str
    password: str


def create_jwt_token(user_id: str) -> str:
    expiration = datetime.now(UTC) + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.endswith("/predict"):
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(
                    status_code=401, content={"detail": "Missing authentication token"}
                )
            try:
                token = token.split()[1]  # Bearer <token>
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        return await call_next(request)


@bentoml.service
class BloodCellModelService:
    def __init__(self):
        self.client = mlflow.MlflowClient()
        self.models: dict[str, drift.ChampionReference] = {}
        self._loaded_at: dict[str, float] = {}

    def _get_champion(self, model: str) -> drift.ChampionReference:
        """Charge (et met en cache, avec TTL) la référence de drift du champion ``model``.

        Le cache est rafraîchi si l'entrée a plus de ``CHAMPION_CACHE_TTL_SECONDS``. Si le
        rechargement échoue alors qu'une entrée (désormais périmée) existe déjà, celle-ci est
        conservée et un avertissement est émis plutôt que de lever une exception -- l'objectif
        est de ne jamais transformer une panne MLflow transitoire en 404 pour un modèle qui
        servait encore des réponses valides l'instant précédent. ``_loaded_at[model]`` est mis à
        jour même quand le rechargement échoue, pour que le prochain essai respecte le même
        TTL plutôt que de retenter un rechargement (potentiellement lent) à chaque requête.

        Exceptions attrapées sur ce chemin de secours (entrée déjà en cache) : ``ValueError``
        (modèle/alias introuvable, modèle pt2-export, modèle sans ``.features()``) ainsi que
        les erreurs de panne d'infrastructure réalistes -- ``FileNotFoundError`` (référence de
        drift absente), ``mlflow.MlflowException`` (autres codes d'erreur MLflow),
        ``requests.RequestException`` et ``OSError`` (réseau/connexion sous-jacents).

        Raises:
            bentoml.exceptions.InvalidArgument: si le modèle/alias est introuvable ou
                si le champion n'expose pas ``.features()`` (cf. ``drift.load_champion_reference``),
                et qu'aucune entrée n'est déjà en cache pour ``model``.
            FileNotFoundError, mlflow.MlflowException, OSError, requests.RequestException:
                propagées telles quelles pour un premier chargement (aucune entrée en cache) --
                une vraie panne d'infra à ce stade ne doit pas être travestie en 404.
        """
        now = time.monotonic()
        loaded_at = self._loaded_at.get(model)
        fresh = loaded_at is not None and now - loaded_at < CHAMPION_CACHE_TTL_SECONDS
        if fresh:
            return self.models[model]

        has_stale_entry = model in self.models
        try:
            self.models[model] = drift.load_champion_reference(model, self.client, "champion")
            self._loaded_at[model] = now
        except ValueError as exc:
            if not has_stale_entry:
                raise bentoml.exceptions.InvalidArgument(str(exc), error_code=404) from exc
            reload_exc = exc
        except (FileNotFoundError, mlflow.MlflowException, OSError, requests.RequestException) as exc:
            if not has_stale_entry:
                raise
            reload_exc = exc
        else:
            return self.models[model]

        warnings.warn(
            f"rechargement du champion {model!r} échoué, référence conservée : {reload_exc}",
            stacklevel=2,
        )
        self._loaded_at[model] = now
        return self.models[model]

    @bentoml.api
    def list_models(self):
        return [
            {
                "name": res.name,
                "description": res.description,
                "tags": res.tags,
                "last_updated_timestamp": res.last_updated_timestamp,
            }
            for res in self.client.search_registered_models(
                filter_string="tag.experience = 'blood_cells'"
            )
            if "champion" in res.aliases
        ]

    def predict_images(
        self,
        champion: drift.ChampionReference,
        images: list[Annotated[Path, ContentType("image/*")]],
    ):
        tensors = [champion.transform(load_image_rgb(img)) for img in images]
        batch = torch.stack(tensors)
        with torch.inference_mode():
            logits = champion.model(batch)

        if champion.task.kind == "binary":
            p1 = torch.sigmoid(logits.reshape(-1))
            indices = (p1 > 0.5).long()
            probs = torch.stack([1 - p1, p1], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            indices = probs.argmax(dim=1)

        probs = probs.cpu().tolist()
        indices = indices.cpu().tolist()

        results = []
        for p, idx in zip(probs, indices):
            label = champion.task.class_names[idx]
            results.append(
                {
                    "probs": dict(zip(champion.task.class_names, p)),
                    "index": idx,
                    "label": label,
                }
            )
            predictions_by_category.labels(model=champion.model_name, category=label).inc()
        return results

    @bentoml.api
    def predict(
        self, model: str, images: list[Annotated[Path, ContentType("image/*")]]
    ) -> list[dict]:
        start_time = time.time()
        try:
            champion = self._get_champion(model)
        except bentoml.exceptions.InvalidArgument:
            api_requests_total.labels(endpoint="/predict", method="POST", status_code="404").inc()
            api_request_duration_seconds.labels(
                endpoint="/predict", method="POST", status_code="404"
            ).observe(time.time() - start_time)
            raise

        results = self.predict_images(champion, images)

        api_requests_total.labels(endpoint="/predict", method="POST", status_code="200").inc()
        api_request_duration_seconds.labels(
            endpoint="/predict", method="POST", status_code="200"
        ).observe(time.time() - start_time)

        return results

    @bentoml.api(route="/evaluate")
    def evaluate(
        self,
        model: str,
        labels: Annotated[list[str], Form()],
        images: list[Annotated[Path, ContentType("image/*")]],
    ):
        start_time = time.time()
        try:
            champion = self._get_champion(model)
        except bentoml.exceptions.InvalidArgument:
            api_requests_total.labels(endpoint="/evaluate", method="POST", status_code="404").inc()
            api_request_duration_seconds.labels(
                endpoint="/evaluate", method="POST", status_code="404"
            ).observe(time.time() - start_time)
            raise

        current = drift.build_current_dataset(champion, images)
        drift_result = drift.compute_drift_report(champion, current)

        result_dict = drift_result.snapshot.dict()
        drifted_columns_result = next(
            m for m in result_dict["metrics"] if "DriftedColumnsCount" in m["metric_name"]
        )
        model_descriptor_columns_drifted_value.labels(model=model).set(
            drifted_columns_result["value"]["count"]
        )
        model_descriptor_columns_drifted_share.labels(model=model).set(
            drifted_columns_result["value"]["share"]
        )
        for m in (m for m in result_dict["metrics"] if "ValueDrift" in m["metric_name"]):
            model_descriptor_drift.labels(model=model, column=m["config"]["column"]).set(m["value"])
        if drift_result.embeddings_drifted is not None:
            embeddings_result = next(
                m for m in result_dict["metrics"] if "EmbeddingsDrift" in m["metric_name"]
            )
            model_embeddings_drift.labels(model=model).set(embeddings_result["value"])
        _set_drift_gauge(model_drift_detected, model, drift_result)

        results = self.predict_images(champion, images)
        results = [r["probs"] | {"target": t} for r, t in zip(results, labels)]
        current_df = pd.DataFrame(results)
        print(current_df.head())
        data_def = DataDefinition(
            classification=[
                MulticlassClassification(
                    target="target",
                    prediction_probas=champion.task.class_names,
                )
            ]
        )
        current_ds = Dataset.from_pandas(current_df, data_definition=data_def)
        report = Report(
            [
                Accuracy(classification_name="default"),
                Precision(classification_name="default"),
                Recall(classification_name="default"),
                F1Score(classification_name="default"),
            ]
        )
        snapshot = report.run(current_data=current_ds)
        metrics = snapshot.dict()["metrics"]
        model_accuracy.labels(model=model).set(metrics[0]["value"])
        model_precision.labels(model=model).set(metrics[1]["value"])
        model_recall.labels(model=model).set(metrics[2]["value"])
        model_f1_score.labels(model=model).set(metrics[3]["value"])

        api_requests_total.labels(endpoint="/evaluate", method="POST", status_code="200").inc()
        api_request_duration_seconds.labels(
            endpoint="/evaluate", method="POST", status_code="200"
        ).observe(time.time() - start_time)
        return {"metrics": metrics + result_dict["metrics"]}


app = FastAPI()


@bentoml.service(metrics={"enabled": True, "namespace": "blood_cells_server"})
@bentoml.asgi_app(app)
class BloodCellsService:
    model_service = bentoml.depends(BloodCellModelService)

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials):
        if credentials.username == USER and credentials.password == PASSWD:
            token = create_jwt_token(credentials.username)
            return {"token": token}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @app.get("/models")
    def list_model(self):
        return self.model_service.list_models()

    @bentoml.api(route="/predict")
    def predict(
        self, model: str, images: list[Annotated[Path, ContentType("image/*")]]
    ) -> list[dict]:
        return self.model_service.predict(model, images)

    @bentoml.api(route="/evaluate")
    def evaluate(
        self,
        model: str,
        labels: Annotated[list[str], Form()],
        images: list[Annotated[Path, ContentType("image/*")]],
    ):
        return self.model_service.evaluate(model, labels, images)


# Appliquer le middleware sur tout le service
BloodCellsService.add_asgi_middleware(JWTAuthMiddleware)
