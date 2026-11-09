from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.cfg import IMAGE_EXTENSIONS, MAPPING
from src.features.common import load_image


class BloodCellMetadata:
    def __init__(self, root_dir: Path, source: str | None = None, read_image: bool = False):
        self.root_dir_ = Path(root_dir)
        self.source_ = source or self.root_dir_.name
        self.read_image_ = read_image
        self.df_ = self.build()
        self.remap_classes()

    def remap_classes(self):

        self.df_["class_code"] = (self.df_["class_code"].map(MAPPING["class_code"]).fillna(self.df_["class_code"]))
        self.df_["class_folder"] = (self.df_["class_folder"].map(MAPPING["class_folder"]).fillna(self.df_["class_folder"]))

    def get_image_paths(self) -> list[Path]:
        """Parcourt le dataset et retourne les chemins des images."""
        image_paths = []

        for class_dir in sorted(self.root_dir_.iterdir()):

            if not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.iterdir()):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image_paths.append(image_path)

        return image_paths

    def build(self) -> pd.DataFrame:

        rows = []
        for image_path in tqdm(self.get_image_paths(), desc="Building metadata"):
            dic = {
                "source": self.source_,
                "relative_path": str(image_path.relative_to(self.root_dir_.parents[1])),
                "class_folder": str(image_path.parent.relative_to(self.root_dir_)),
                "class_code": str(image_path.stem.split("_")[0]),
                "file_type": image_path.suffix.lower(),
                "file_size": image_path.stat().st_size,
            }

            if self.read_image_:
                try:
                    image = load_image(image_path)
                    # Passage en niveaux de gris pour les mesures
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                    height, width = gray.shape
                    dic["height"] =  height
                    dic["width"] = width,
                    dic["channels"] = 1 if image.ndim == 2 else image.shape[2]
                    dic["brightness"] = float(np.mean(gray))
                    dic["gray_min"] = int(np.min(gray))
                    dic["gray_max"] = int(np.max(gray))
                    dic["std"] = float(np.std(gray))
                    dic["contrast"] = float(np.sqrt(np.mean((gray.astype(np.float32) - np.mean(gray)) ** 2)))
                    dic["aspect_ratio"] = width / height
                    dic["jpeg_quality"] = estimate_jpeg_quality(image)
                    dic["hue_mean"] = float(np.mean(hsv[:, :, 0]))
                    dic["saturation_mean"] = float(np.mean(hsv[:, :, 1]))
                    dic["value_mean"] = float(np.mean(hsv[:, :, 2]))
                    dic["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if image.ndim == 3:
                        dic["r_mean"] = float(np.mean(image[:, :, 0]))
                        dic["g_mean"] = float(np.mean(image[:, :, 1]))
                        dic["b_mean"] = float(np.mean(image[:, :, 2]))
                        dic["r_std"] = float(np.std(image[:, :, 0]))
                        dic["g_std"] = float(np.std(image[:, :, 1]))
                        dic["b_std"] = float(np.std(image[:, :, 2]))
                except Exception:  # noqa: BLE001 -- une image illisible ne doit pas interrompre le build
                    print("can't read image path:", image_path)

            rows.append(dic)

        return pd.DataFrame(rows)

    def get_df(self):
        return self.df_

def estimate_jpeg_quality(image):

    best_quality = None
    best_error = float("inf")

    for quality in range(10, 101):

        encode_param = [ cv2.IMWRITE_JPEG_QUALITY, quality ]
        _, encoded = cv2.imencode( ".jpg", image, encode_param )
        decoded = cv2.imdecode( encoded, cv2.IMREAD_COLOR )
        error = np.mean( (image.astype(np.float32) - decoded.astype(np.float32)) ** 2 )

        if error < best_error:
            best_error = error
            best_quality = quality

    return best_quality
