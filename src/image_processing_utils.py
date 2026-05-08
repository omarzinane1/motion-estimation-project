import cv2
import numpy as np

# =========================
# 🔹 PREPROCESSING OPTIONNEL
# =========================
def preprocess_image(image_bgr, target_size=(640, 360), clip_limit=2.0, tile_grid_size=(8, 8)):
    resized = cv2.resize(image_bgr, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    normalized = enhanced.astype(np.float32) / 255.0
    return normalized


# =========================
# 🔹 FILTRES
# =========================
def gaussian_filter(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_filter(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


# =========================
# 🔹 DETECTION DE CONTOURS
# =========================
def sobel_edges(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(np.clip(sobel, 0, 255))


def laplacian_edges(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(lap), 0, 255))


def canny_edges(img, low=50, high=150):
    return cv2.Canny(img, low, high)


# =========================
# 🔹 ROI VOITURE
# =========================
def extract_roi(img, bbox):
    x, y, w, h = bbox.astype(int)
    return img[y:y+h, x:x+w]


# =========================
# 🔹 OVERLAY CONTOURS ROUGES
# =========================
def overlay_edges_on_roi(roi, edges):
    roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    roi_color[edges > 0] = [0, 0, 255]  # rouge BGR
    return roi_color


# =========================
# 🔹 DRAW BBOX
# =========================
def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x, y, w, h = bbox.astype(int)
    img_copy = img.copy()
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, thickness)
    return img_copy