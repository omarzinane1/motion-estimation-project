from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path):
    """Charge une image couleur BGR avec OpenCV."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    return image


def to_gray(image):
    """Convertit une image BGR/RGB en niveaux de gris."""
    if image is None:
        raise ValueError("L'image fournie est vide.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Format d'image non supporte: shape={image.shape}")


def apply_clahe(gray, clipLimit=2.0, tileGridSize=(8, 8)):
    """Ameliore le contraste local avec CLAHE."""
    if gray.ndim != 2:
        raise ValueError("CLAHE attend une image en niveaux de gris.")
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)


def apply_blur(image, kernel_size=(5, 5)):
    """Reduit le bruit avec un flou gaussien."""
    return cv2.GaussianBlur(image, kernel_size, 0)


def preprocess_image(image):
    """Pipeline: BGR/RGB -> grayscale -> CLAHE -> GaussianBlur."""
    gray = to_gray(image)
    enhanced = apply_clahe(gray, clipLimit=2.0, tileGridSize=(8, 8))
    blurred = apply_blur(enhanced, kernel_size=(5, 5))
    return blurred


def load_image_sequence(img_dir):
    """Retourne la liste triee des chemins d'images d'une sequence."""
    directory = Path(img_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Dossier images introuvable: {directory}")

    image_paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"Aucune image trouvee dans: {directory}")
    return image_paths

