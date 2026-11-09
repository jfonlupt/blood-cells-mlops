# ---------- Étape 1 : build ----------
FROM python:3.14-slim AS builder

# Récupère le binaire uv depuis l'image officielle (épingle une version précise en prod)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Dépendances seules → couche mise en cache tant que le lockfile ne change pas
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --group server

# 2) Code du projet -- COPY explicite (pas `COPY . /app`) : seuls ces
# chemins sont nécessaires pour construire le paquet `src` (cf.
# [tool.hatch.build.targets.wheel] packages = ["src"] et readme =
# "README.md" dans pyproject.toml). `data/`, `.venv/`, `results/`, `.git/`
# etc. ne sont ni copiés ni même considérés par le daemon, qu'importe le
# fonctionnement de .dockerignore pour ces chemins -- `data/` est monté en
# volume à l'exécution (docker-compose.yaml), jamais nécessaire dans l'image.
COPY pyproject.toml uv.lock README.md /app/
COPY src/ /app/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group server --no-editable

# ---------- Étape 2 : runtime ----------
FROM python:3.14-slim

# Commit du repo au moment du build (passé via --build-arg, cf. docker-compose.yaml) :
# seul .venv est copié dans cette image, jamais .git ni le binaire git, donc la
# détection automatique du commit par mlflow (GitRunContext) ne peut pas fonctionner
# ici. src.experiments.tracking lit GIT_COMMIT pour taguer les runs à la place.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

RUN useradd --create-home app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
USER app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.experiments.server:app", "--host", "0.0.0.0", "--port", "8000"]