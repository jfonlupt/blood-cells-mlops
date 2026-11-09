"""File to store all constants across the project."""

import os
from pathlib import Path

BARCELONA_KEY = "barcelona"
MUNICH_KEY = "munich"
MUNICH_TRAIN_KEY = "munich_train"
MUNICH_EVAL_KEY = "munich_eval"
DRIFT_TRAIN_KEY = "data_drift_train"
DRIFT_EVAL_KEY = "data_drift_eval"
DATASETS_KEYS = [
    BARCELONA_KEY,
    MUNICH_KEY,
    MUNICH_TRAIN_KEY,
    MUNICH_EVAL_KEY,
    DRIFT_TRAIN_KEY,
    DRIFT_EVAL_KEY,
]

BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR dérive de l'emplacement du package installé : en local (install
# éditable), il pointe sur la racine du repo ; dans l'image Docker (wheel
# non-éditable), il pointe dans .venv/site-packages, sans rapport avec /app où
# les volumes sont montés. DATA_DIR permet de forcer l'emplacement réel des
# données (ex. docker-compose.yaml fixe DATA_DIR=/app/data).
DATA_PATH = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
DATASET_PATHS = {
    BARCELONA_KEY: DATA_PATH / "PBC_dataset_normal_DIB",
    MUNICH_KEY: DATA_PATH / "PKG-AML-Cytomorphology_LMU",
    MUNICH_TRAIN_KEY: DATA_PATH / "train" / "munich",
    MUNICH_EVAL_KEY: DATA_PATH / "test" / "munich",
    DRIFT_TRAIN_KEY: DATA_PATH / "train" / "data_drift",
    DRIFT_EVAL_KEY: DATA_PATH / "test" / "data_drift",
}
RESULTS_PATH = BASE_DIR / "reports"

FINAL_CLASSES = [
    "Basophil",
    "Blast",
    "Eosinophil",
    "Erythroblast",
    "IG",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
    "Promyelocyte",
]
BARCELONA_TO_FINAL = {
    "BA": "Basophil",
    "BNE": "Neutrophil",
    "EO": "Eosinophil",
    "ERB": "Erythroblast",
    "IG": "IG",  # générique non sous-typé -> versé dans IG par défaut
    "LY": "Lymphocyte",
    "MMY": "IG",  # métamyélocyte
    "MO": "Monocyte",
    "MY": "IG",  # myélocyte
    "NEUTROPHIL": "Neutrophil",
    "PLATELET": "Platelet",
    "PMY": "Promyelocyte",
    "SNE": "Neutrophil",
}
MUNICH_TO_FINAL = {
    "BAS": "Basophil",
    "EBO": "Erythroblast",
    "EOS": "Eosinophil",
    "KSC": None,  # smudge cell -> exclue
    "LYA": "Lymphocyte",
    "LYT": "Lymphocyte",
    "MMZ": "IG",  # métamyélocyte
    "MOB": "Blast",  # monoblaste -> fusionné avec myéloblaste
    "MON": "Monocyte",
    "MYB": "IG",  # myélocyte
    "MYO": "Blast",  # myéloblaste
    "NGB": "Neutrophil",  # neutrophile en bande
    "NGS": "Neutrophil",  # neutrophile segmenté
    "PMB": "Promyelocyte",  # promyélocyte bilobé
    "PMO": "Promyelocyte",
}
MAPPING_DICT = {
    BARCELONA_KEY: BARCELONA_TO_FINAL,
    MUNICH_KEY: MUNICH_TO_FINAL,
    MUNICH_TRAIN_KEY: MUNICH_TO_FINAL,
    MUNICH_EVAL_KEY: MUNICH_TO_FINAL,
    DRIFT_TRAIN_KEY: BARCELONA_TO_FINAL,
    DRIFT_EVAL_KEY: BARCELONA_TO_FINAL,
}

CRITICALITY_CLASSES = ["normal", "intermediaire", "critique"]
CRITICALITY = {
    "Blast": "critique",
    "Promyelocyte": "critique",
    "IG": "intermediaire",
    "Erythroblast": "intermediaire",
    "Neutrophil": "normal",
    "Lymphocyte": "normal",
    "Eosinophil": "normal",
    "Monocyte": "normal",
    "Platelet": "normal",
    "Basophil": "normal",
}
BINARY_CRITICALITY_CLASSES = ["normal", "abnormal"]
BINARY_CRITICALITY = {k: v if v == "normal" else "abnormal" for k, v in CRITICALITY.items()}

TYPE_TO_LABEL = {k: i for i, k in enumerate(FINAL_CLASSES)}
LABEL_TO_TYPE = dict(enumerate(FINAL_CLASSES))
DATASET_TO_LABEL = {BARCELONA_KEY: 0, MUNICH_KEY: 1}
LABEL_TO_DATASET = {0: BARCELONA_KEY, 1: MUNICH_KEY}
CRITICALITY_TO_LABEL = {k: i for i, k in enumerate(CRITICALITY_CLASSES)}
LABEL_TO_CRITICALITY = dict(enumerate(CRITICALITY_CLASSES))
BINARY_CRITICALITY_TO_LABEL = {k: i for i, k in enumerate(BINARY_CRITICALITY_CLASSES)}
LABEL_TO_BINARY_CRITICALITY = dict(enumerate(BINARY_CRITICALITY_CLASSES))
CRITICAL_LABELS = [
    TYPE_TO_LABEL[x[0]] for x in filter(lambda x: x[1] == "critique", CRITICALITY.items())
]
INTERMEDIARY_LABELS = [
    TYPE_TO_LABEL[x[0]] for x in filter(lambda x: x[1] == "intermediaire", CRITICALITY.items())
]
NORMAL_LABELS = [
    TYPE_TO_LABEL[x[0]] for x in filter(lambda x: x[1] == "normal", CRITICALITY.items())
]
ABNORMAL_LABELS = CRITICAL_LABELS + INTERMEDIARY_LABELS
MODE_TO_LABEL = {
    "dataset": DATASET_TO_LABEL,
    "type": TYPE_TO_LABEL,
    "criticality": CRITICALITY_TO_LABEL,
    "binary_criticality": BINARY_CRITICALITY_TO_LABEL,
}

IMG_SIZE = (224, 224)  # (H, W)
