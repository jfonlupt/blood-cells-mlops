import cv2
import numpy as np


def load_image(path):
    image = cv2.imread(
                    path,
                    cv2.IMREAD_UNCHANGED
                )

    if image is None:
        raise ValueError(
            f"Impossible de lire l'image : {path}"
        )

    # -------------------------
    # Conversion RGB
    # -------------------------

    if image.ndim == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    elif image.ndim == 3:

        if image.shape[2] == 3:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        elif image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGB
            )

        else:
            raise ValueError(
                f"Nombre de canaux inattendu : "
                f"{image.shape}"
            )

    else:
        raise ValueError(
            f"Format d'image inattendu : "
            f"{image.shape}"
        )

    return image
