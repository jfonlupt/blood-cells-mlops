# src/metadata/cfg.py

from pathlib import Path

# ============================================================
# ROOT DU PROJET
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATASETS
# ============================================================

DATA_DIR = ROOT / "data"

# Barcelona
BARCELONA_DIR = DATA_DIR / "PBC_dataset_normal_DIB"

# Munich
MUNICH_DIR = DATA_DIR / "PKG-AML-Cytomorphology_LMU"


# ============================================================
# SORTIE DES METADONNEES
# ============================================================

OUTPUT_DIR = ROOT / "data" / "metadata"

BARCELONA_METADATA_PATH = OUTPUT_DIR / "barcelona_metadata.csv"
MUNICH_METADATA_PATH = OUTPUT_DIR / "munich_metadata.csv"
ALL_METADATA_PATH = OUTPUT_DIR / "all_metadata"


# ============================================================
# IMAGES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# PARAMETRES
# ============================================================

# Taille des chunks utilisée pour le calcul MD5
MD5_CHUNK_SIZE = 2**16

# ============================================================
# mapping
# ============================================================

MAPPING = {
    "class_code": {
        "BA": "BAS",
        "EO": "EOS",
        "ERB": "EBO",
        "LY": "LYT",
        "MO": "MON",
        "BNE": "NGB",
        "SNE": "NGS",
        "NEUTROPHIL": "NEUTROPHIL",
        "PMY": "PMO",
        "MY": "MYB",
        "MMY": "MMZ",
        "MYO": "MYO",
        "IG": "IG",
        "PLATELET": "PLATELET",
        "MOB": "MOB",
        "PMB": "PMB",
        "KSC": "KSC",
        "LYA": "LYA",
    },

    "class_folder": {
        "BAS": "Basophil",
        "EOS": "Eosinophil",
        "EBO": "Erythroblast",
        "LYT": "Lymphocyte (typical)",
        "MON": "Monocyte",
        "NGB": "Neutrophil (band)",
        "NGS": "Neutrophil (segmented)",
        "NEUTROPHIL": "Neutrophil (unknown)",
        "PMO": "Promyelocyte",
        "MYB": "Myelocyte",
        "MMZ": "Metamyelocyte",
        "IG": "Immature Granulocyte",
        "PLATELET": "Platelet",
        "MYO" : "Myeloblast",
        "MOB": "Monoblast",
        "PMB": "Promyelocyte (bilobled)",
        "KSC": "Smudge cell",
        "LYA": "Lymphocyte (atypical)",
    }
}
