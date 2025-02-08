"""
Ensemble Model for Butterfly Hybrid Detection

This model uses a DINO backbone from HuggingFace Transformers and an ensemble of three classifiers
(KNN, SGD, SVM) whose weights are stored in:
    - trained_knn_classifier.pkl
    - trained_sgd_classifier.pkl
    - trained_svm_classifier.pkl

The model implements the following methods:
    - load: loads the backbone and the classifiers.
    - predict: given an input PIL image, returns an anomaly score (float).
    - fit: not supported (raises NotImplementedError).
    - save: saves the ensemble model information.
"""

import os
import pickle
from typing import Any
import torch
from PIL import Image, ImageEnhance
import cv2
from torchvision import transforms
from transformers import AutoModel
import numpy as np

# ------------------------------
# Transformation Functions
# ------------------------------

def scribble_image(image: Image.Image,
                   cross_width_factor: float = 0.05,
                   bottom_area_ratio: float = 0.15,
                   square_ratio: float = 0.2) -> Image.Image:
    """
    Applies a scribble mask to the image:
      1. A central cross (width = cross_width_factor * image width).
      2. The bottom area (height = bottom_area_ratio * image height).
      3. A square at the bottom-right (side = square_ratio * image width).
    """
    img_array = np.array(image)
    height, width, _ = img_array.shape

    cross_thickness = int(width * cross_width_factor)
    center_x = width // 2
    center_y = height // 2
    half_th = cross_thickness // 2

    # Draw vertical and horizontal lines (set pixels to 255)
    img_array[:, center_x - half_th: center_x + half_th, :] = 255
    img_array[center_y - half_th: center_y + half_th, :, :] = 255

    # Fill bottom area
    bottom_start = int(height * (1 - bottom_area_ratio))
    img_array[bottom_start: height, :, :] = 255

    # Fill bottom-right square
    square_side = int(width * square_ratio)
    x_start = width - square_side
    y_start = height - square_side
    img_array[y_start: height, x_start: width, :] = 255

    return Image.fromarray(img_array)

def transform_scribble_image(image: Image.Image) -> Image.Image:
    return scribble_image(
        image,
        cross_width_factor=0.05,
        bottom_area_ratio=0.15,
        square_ratio=0.23
    )

def enhance_color_and_contrast(image: Image.Image,
                               color_factor: float = 1.5,
                               contrast_factor: float = 1.2) -> Image.Image:
    color_enhancer = ImageEnhance.Color(image)
    color_img = color_enhancer.enhance(color_factor)
    contrast_enhancer = ImageEnhance.Contrast(color_img)
    final_img = contrast_enhancer.enhance(contrast_factor)
    return final_img

def transform_enhance_color_and_contrast(image: Image.Image) -> Image.Image:
    return enhance_color_and_contrast(image, color_factor=1.5, contrast_factor=1.2)

def sharpen_image(image: Image.Image, kernel_strength: float = 5) -> Image.Image:
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, kernel_strength, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened_bgr = cv2.filter2D(src=img_bgr, ddepth=-1, kernel=sharpen_kernel)
    sharpened_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sharpened_rgb)

def transform_sharpen_image(image: Image.Image) -> Image.Image:
    return sharpen_image(image, kernel_strength=5.3)

def replace_background_with_white(image: Image.Image, tolerance: int = 70) -> Image.Image:
    img_array = np.array(image)
    height, width, _ = img_array.shape
    # Estimate background color from the top corners
    LU = img_array[0, 0]
    RU = img_array[0, width - 1]
    bg_color = np.mean([LU, RU], axis=0).astype(int)
    distance = np.linalg.norm(img_array[:, :, :3] - bg_color, axis=2)
    mask = distance <= tolerance
    img_array[mask] = [255, 255, 255]
    return Image.fromarray(img_array)

def transform_replace_background(image: Image.Image) -> Image.Image:
    return replace_background_with_white(image, tolerance=70)

# ------------------------------
# Ensemble Model Class
# ------------------------------

class Model:
    def __init__(self):
        self.dino_name = 'facebook/dinov2-base'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pil_transform_fn = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.Lambda(transform_enhance_color_and_contrast),
            #transforms.Lambda(transform_sharpen_image),
            #transforms.Lambda(transform_replace_background),
            transforms.Lambda(transform_scribble_image),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.is_trained = False

    def load(self):
        """
        Loads the DINO backbone and the three ensemble classifiers.
        Expects the weight files (trained_knn_classifier.pkl, trained_sgd_classifier.pkl,
        trained_svm_classifier.pkl) to be in the same directory as this file.
        """
        # Load DINO backbone from HuggingFace
        self.backbone = AutoModel.from_pretrained(self.dino_name)
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # Load classifiers from local files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        knn_path = os.path.join(current_dir, "trained_knn_classifier.pkl")
        sgd_path = os.path.join(current_dir, "trained_sgd_classifier.pkl")
        svm_path = os.path.join(current_dir, "trained_svm_classifier.pkl")
        with open(knn_path, "rb") as f:
            self.knn_clf = pickle.load(f)
        with open(sgd_path, "rb") as f:
            self.sgd_clf = pickle.load(f)
        with open(svm_path, "rb") as f:
            self.svm_clf = pickle.load(f)
        self.is_trained = True

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from input tensor using the DINO backbone.
        The feature vector is obtained by concatenating the [CLS] token and
        the mean of the patch tokens.
        """
        feats = self.backbone(x)[0]
        cls_token = feats[:, 0]
        patch_tokens = feats[:, 1:]
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return features

    def predict(self, image: Image.Image) -> float:
        """
        Given a PIL image, applies the transformations, extracts features, and
        returns an anomaly score (ensemble probability) based on the three classifiers.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not loaded. Please call load() before predict().")
        x_tensor = self.pil_transform_fn(image).to(self.device).unsqueeze(0)
        features = self._get_features(x_tensor)
        np_features = features.detach().cpu().numpy()
        # Get probabilities from each classifier (assumes predict_proba returns probability for class 1 at index 1)
        prob_knn = self.knn_clf.predict_proba(np_features)[0, 1]
        prob_sgd = self.sgd_clf.predict_proba(np_features)[0, 1]
        prob_svm = self.svm_clf.predict_proba(np_features)[0, 1]
        # Ensemble by averaging the probabilities
        ensemble_prob = (prob_knn + prob_sgd + prob_svm) / 3.0
        return ensemble_prob

    def fit(self, X: Any, y: Any):
        """
        Not implemented.
        """
        raise NotImplementedError("Fit method is not implemented for the ensemble model.")

    def save(self, path: str):
        """
        Saves the ensemble model information to a file.
        (Typically, the backbone is loaded from a pre-trained model, so here we save only the classifier info.)
        """
        model_data = {
            'dino_name': self.dino_name,
            'knn': self.knn_clf,
            'sgd': self.sgd_clf,
            'svm': self.svm_clf
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
