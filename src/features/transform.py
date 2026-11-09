from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, v2


def _import_kornia():
    """Import différé de kornia (utilisé seulement par CLAHE).

    Évite son coût d'import quand CLAHE est désactivé, et masque le
    ``FutureWarning`` que kornia déclenche à l'import sous Python 3.14
    (``@torch.jit.script`` sur une fonction epipolar qu'on n'appelle jamais).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*", category=FutureWarning)
        import kornia

        return kornia


if TYPE_CHECKING:
    from src.config import TransformConfig

AUTOTUNE_WORKERS = 4  # a ajuster selon le nombre de coeurs disponibles (os.cpu_count())

# Statistiques ImageNet, attendues par les backbones timm / torchvision pre-entraines.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# 1-2. Lecture + redimensionnement (avec padding, preserve le ratio d'origine)
# ---------------------------------------------------------------------------


def load_image_rgb(path):
    """Lecture native torchvision : pas de PIL, pas d'OpenCV.

    Retourne un tenseur CHW uint8, RGB (ImageReadMode.RGB force 3 canaux
    quelle que soit l'image source).
    """
    with Image.open(path) as img:
        img = img.convert("RGB")  # force 3 canaux, gere aussi les TIFF en niveaux de gris/CMYK/etc.
        return v2.functional.pil_to_tensor(img)


def resize_with_pad(image, target_size, fill=255):
    """Equivalent natif torchvision de tf.image.resize_with_pad.

    Redimensionne en conservant le ratio d'origine (pas de distorsion) puis
    complete avec du blanc (fill=255, coherent avec le fond clair des lames).
    image : CHW uint8.
    """
    target_h, target_w = target_size
    _, h, w = image.shape
    scale = min(target_w / w, target_h / h)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))

    image = v2.functional.resize(
        image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return v2.functional.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)


# ---------------------------------------------------------------------------
# 4. Normalisation colorimetrique : Reinhard (torchstain) + CLAHE (kornia)
# ---------------------------------------------------------------------------
# Deterministes (pas d'alea) -> appliquees identiquement sur train/val/test.


def apply_reinhard(image_0_1, normalizer):
    """image_0_1 : CHW float [0,1].

    torchstain attend du [0,255] -> conversion explicite dans les deux sens
    autour de l'appel.

    NOTE (verifie empiriquement, pas documente) : bien que l'entree de
    TorchReinhardNormalizer.normalize() soit CHW, sa sortie ressort en HWC --
    torch.tensordot() dans son lab2rgb() interne replace l'axe des canaux en
    dernier. D'ou le .permute() explicite ci-dessous pour revenir en CHW,
    sans quoi les etapes suivantes (kornia, etc.) recoivent un tenseur mal
    forme et echouent (ou pire, s'executent silencieusement sur les mauvais axes).
    """
    image_255 = image_0_1 * 255.0
    normalized_uint8 = normalizer.normalize(I=image_255)  # -> HWC uint8, [0,255] (quirk torchstain)
    normalized_uint8 = normalized_uint8.permute(2, 0, 1)  # HWC -> CHW
    return normalized_uint8.float() / 255.0


def apply_clahe(image_0_1, clip_limit=40.0, grid_size=8):
    """image_0_1 : CHW float [0,1].

    CLAHE applique uniquement au canal L (luminance) en espace Lab -- comme la
    version cv2 -- pour ne pas alterer la teinte/saturation, qui porte un
    signal diagnostique (couleur des granulations).
    NOTE : clip_limit de kornia n'est PAS sur la meme echelle que le clipLimit
    d'OpenCV (defaut 2.0-4.0 chez cv2, defaut 40.0 chez kornia) -- ne pas
    reutiliser l'ancienne valeur telle quelle, revalider visuellement.
    """
    kornia = _import_kornia()
    lab = kornia.color.rgb_to_lab(
        image_0_1.unsqueeze(0)
    )  # (1,3,H,W) ; L in [0,100], a/b in [-128,127]
    l_channel = lab[:, 0:1, :, :] / 100.0  # -> [0,1] pour equalize_clahe
    l_channel = kornia.enhance.equalize_clahe(
        l_channel, clip_limit=clip_limit, grid_size=(grid_size, grid_size)
    )
    lab = torch.cat([l_channel * 100.0, lab[:, 1:, :, :]], dim=1)
    rgb = kornia.color.lab_to_rgb(lab)  # -> [0,1], clip=True par defaut
    return rgb.squeeze(0)


# ---------------------------------------------------------------------------
# 3+5-6. Un seul v2.Compose pour toute la chaine (resize -> augmentation ->
# Reinhard/CLAHE -> normalisation), + un Dataset qui se contente de l'appliquer
# ---------------------------------------------------------------------------
# v2.Compose ne fait rien de plus que `for t in transforms: image = t(image)`.
# N'importe quel objet appelable (fonction ou classe avec __call__) peut donc y
# figurer -- pas besoin d'heriter de v2.Transform ni de nn.Module pour ca, ce
# n'est utile que si on veut rendre le transform torch.jit.script-able. Les
# briques "maison" (resize+pad, Reinhard, CLAHE) sont donc juste enveloppees
# dans de petites classes __call__, au meme niveau que les transforms v2 natifs.


class ResizeWithPad:
    """Enveloppe resize_with_pad() pour la rendre compatible Compose."""

    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        return resize_with_pad(image, self.size, fill=self.fill)


class ToUnitRange:
    """uint8 [0,255] -> float32 [0,1] (convention native torchvision/kornia)."""

    def __call__(self, image):
        return image.float() / 255.0


class ReinhardTransform:
    """Enveloppe apply_reinhard().

    normalizer=None -> no-op (permet de garder le meme Compose meme quand
    Reinhard est desactive).
    """

    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, image):
        if self.normalizer is None:
            return image
        return apply_reinhard(image, self.normalizer)


class CLAHETransform:
    """Enveloppe apply_clahe()."""

    def __init__(self, clip_limit=40.0, grid_size=8):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, image):
        return apply_clahe(image, self.clip_limit, self.grid_size)


def build_transform(
    cfg: TransformConfig,
    *,
    training: bool,
    reinhard_normalizer=None,
    dataset_mean=None,
    dataset_std=None,
) -> v2.Compose:
    """Construit le ``v2.Compose`` complet (resize -> aug -> Reinhard/CLAHE -> normalize).

    Seule la partie augmentation (rotation / flip / jitter) est conditionnee par
    ``training`` ; Reinhard / CLAHE / normalisation sont identiques sur tous les
    splits (harmonisation, pas augmentation).

    Args:
        cfg: Configuration de pretraitement (taille, toggles, probabilites).
        training: Si vrai, ajoute les etapes d'augmentation aleatoire.
        reinhard_normalizer: Normaliseur Reinhard deja fitte, ou ``None`` pour ignorer.
        dataset_mean: Moyenne par canal pour ``Normalize``. Requis (et seul utilise)
            quand ``cfg.norm_stats == "dataset"`` ; ignore quand ``"imagenet"``.
        dataset_std: Ecart-type par canal pour ``Normalize`` (memes regles).

    Returns:
        Le ``v2.Compose`` pret a etre applique a un tenseur CHW uint8.

    Raises:
        ValueError: si ``cfg.norm_stats == "dataset"`` sans ``dataset_mean`` /
            ``dataset_std``.
    """
    steps: list = [ResizeWithPad(cfg.img_size, fill=255)]
    if cfg.jpeg:
        steps.append(v2.JPEG(cfg.jpeg_quality))
    steps.append(ToUnitRange())

    if training:
        # geometrique : n'affecte pas les statistiques de couleur
        steps += [
            v2.RandomApply(
                [
                    v2.RandomRotation(
                        degrees=180, fill=1.0, interpolation=InterpolationMode.BILINEAR
                    )
                ],
                p=cfg.p_rotation,
            ),
            v2.RandomHorizontalFlip(p=cfg.p_flip),  # probabilite 0.5 native, pas de double gating
            v2.RandomVerticalFlip(p=cfg.p_flip),  # probabilite 0.5 native, pas de double gating
        ]

    # normalisation colorimetrique, deterministe, meme traitement train/val/test
    if reinhard_normalizer is not None:
        steps.append(ReinhardTransform(reinhard_normalizer))
    if cfg.clahe:
        steps.append(CLAHETransform(cfg.clahe_clip_limit, cfg.clahe_grid_size))

    if training:
        # jitter colorimetrique : APRES Reinhard, pour ne pas etre annule par
        # son remapping vers des statistiques de couleur fixes
        steps += [
            v2.RandomApply([v2.ColorJitter(brightness=cfg.brightness_delta)], p=cfg.p_brightness),
            v2.RandomApply([v2.ColorJitter(contrast=cfg.contrast_delta)], p=cfg.p_contrast),
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        cfg.blur_kernel_size, sigma=(cfg.blur_sigma_min, cfg.blur_sigma_max)
                    )
                ],
                p=cfg.p_blur,
            ),
        ]

    # normalisation finale : une seule source pour (mean, std)
    if cfg.norm_stats == "imagenet":
        mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
    elif dataset_mean is None or dataset_std is None:
        raise ValueError(
            "norm_stats='dataset' exige dataset_mean et dataset_std "
            "(passer les stats du split, ou utiliser norm_stats='imagenet')."
        )
    else:
        mean = dataset_mean.tolist() if torch.is_tensor(dataset_mean) else list(dataset_mean)
        std = dataset_std.tolist() if torch.is_tensor(dataset_std) else list(dataset_std)
    steps.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(steps)
