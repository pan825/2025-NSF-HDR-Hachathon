import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import cv2

def scribble_image(image: Image.Image,
                   cross_width_factor: float = 0.05,
                   bottom_area_ratio: float = 0.15,
                   square_ratio: float = 0.2) -> Image.Image:
    """
    將影像做以下塗白處理:
      1. 圖片中心十字架 (寬度占圖片寬度的 cross_width_factor)。
      2. 圖片下半部 bottom_area_ratio 的區域。
      3. 右下角一個正方形區域 (邊長 = square_ratio * 圖片寬度)。
    
    :param image:       PIL Image (RGB)。
    :param cross_width_factor: 十字架占圖片寬度的比例 (預設 0.05)。
    :param bottom_area_ratio:  下半部要塗白的高度比例 (預設 0.15)。
    :param square_ratio:       右下角正方形邊長相對於圖片寬度的比例 (預設 0.2)。
    :return:            塗白後的 PIL Image。
    """

    # 將 PIL Image 轉為 numpy array (RGB)
    img_array = np.array(image)
    height, width, channels = img_array.shape  # shape: (H, W, C)

    # ------------------------------------------------
    # 1. 以圖片中心為焦點的十字架 (cross)
    # ------------------------------------------------
    cross_thickness = int(width * cross_width_factor)  # 十字架寬度 (pixel)
    center_x = width // 2
    center_y = height // 2

    # 十字架包括：垂直線 + 水平線
    # 垂直線：x ∈ [center_x - half_th, center_x + half_th]
    # 水平線：y ∈ [center_y - half_th, center_y + half_th]

    half_th = cross_thickness // 2
    
    # 塗白(255) -- 注意三個 channel (RGB)
    # 垂直線
    img_array[:, center_x - half_th : center_x + half_th, :] = 255
    # 水平線
    img_array[center_y - half_th : center_y + half_th, :, :] = 255

    # ------------------------------------------------
    # 2. 圖片下半部 bottom_area_ratio 的區域
    # ------------------------------------------------
    # 下半部 15% => y ∈ [0.85 * height, height] (以 0.85 為例)
    bottom_start = int(height * (1 - bottom_area_ratio))
    img_array[bottom_start : height, :, :] = 255

    # ------------------------------------------------
    # 3. 右下角正方形 (邊長 = square_ratio * width)
    # ------------------------------------------------
    square_side = int(width * square_ratio)
    # 右下角：x ∈ [width - square_side, width], y ∈ [height - square_side, height]
    x_start = width - square_side
    y_start = height - square_side
    img_array[y_start : height, x_start : width, :] = 255

    # 將修改後的陣列轉回 PIL Image
    result_img = Image.fromarray(img_array)

    return result_img

def transform_scribble_image(image: Image.Image) -> Image.Image:
    """
    Wrapper function for transforms.Lambda() 。
    直接呼叫 scribble_image，並可在這裡調整參數。
    """
    return scribble_image(
        image,
        cross_width_factor=0.05,   # 十字架寬度占圖片寬度 5%
        bottom_area_ratio=0.15,    # 下半部 15%
        square_ratio=0.23           # 右下角正方形 20% 的寬度
    )


def enhance_color_and_contrast(image: Image.Image,
                               color_factor: float = 1.5,
                               contrast_factor: float = 1.2) -> Image.Image:
    """
    使用 PIL 的 ImageEnhance 將圖片的色彩與對比調高，讓圖片更鮮豔或對比更明顯。

    :param image: PIL Image (RGB)。
    :param color_factor: 控制色彩飽和度的增強倍數 (>1 代表增加)。
    :param contrast_factor: 控制對比度的增強倍數 (>1 代表增加)。
    :return: 進行增強後的 PIL Image (RGB)。
    """
    # 1. 色彩增強
    color_enhancer = ImageEnhance.Color(image)
    color_img = color_enhancer.enhance(color_factor)

    # 2. 對比度增強
    contrast_enhancer = ImageEnhance.Contrast(color_img)
    final_img = contrast_enhancer.enhance(contrast_factor)

    return final_img

def transform_enhance_color_and_contrast(image: Image.Image) -> Image.Image:
    """
    Wrapper function，提供給 transforms.Lambda 使用。
    你可以在這裡設定預設的 color_factor 與 contrast_factor。
    """
    return enhance_color_and_contrast(
        image,
        color_factor=1.5,      # 調高數值可讓顏色更濃烈
        contrast_factor=1.2    # 調高數值可讓對比更顯著
    )


def sharpen_image(image: Image.Image, kernel_strength: float = 5) -> Image.Image:
    """
    使用 OpenCV 對 PIL Image 進行銳化 (Sharpen)。
    :param image: PIL Image (RGB)。
    :param kernel_strength: 控制中心權重的大小 (預設 5)。
    :return: 銳化後的 PIL Image (RGB)。
    """
    # PIL -> numpy array (RGB)
    img_array = np.array(image)  # shape: (H, W, 3) in RGB

    # 將 RGB 轉成 BGR，因為 cv2.filter2D 預期的色彩順序是 BGR
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 根據 kernel_strength 來定義 3x3 銳化濾波器
    # 中心點 kernel_strength，其餘 -1，並保留原先的結構
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1, kernel_strength, -1],
        [ 0, -1,  0]

    ], dtype=np.float32)

    # 執行 filter2D 進行銳化
    sharpened_bgr = cv2.filter2D(src=img_bgr, ddepth=-1, kernel=sharpen_kernel)

    # BGR -> RGB
    sharpened_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)

    # 轉回 PIL Image
    sharpened_image = Image.fromarray(sharpened_rgb)

    return sharpened_image

def transform_sharpen_image(image: Image.Image) -> Image.Image:
    """
    Wrapper function for transforms.Lambda to銳化影像。
    你可以在這裡修改參數 (如 kernel_strength)。
    """
    return sharpen_image(image, kernel_strength=5.3)



def replace_background_with_white(image, tolerance=70):
    """
    Replace all pixels matching bg_color with white.

    :param image: PIL Image in RGB mode.
    :param tolerance: Integer representing the color tolerance for matching background colors.
    :return: PIL Image with background set to white.
    """
    img_array = np.array(image)
    height, width, _ = img_array.shape
    LU = img_array[0, 0]
    RU = img_array[0, width-1]
    corners = np.array((LU, RU))
    bg_color = np.mean(np.array(corners), axis=0).astype(int)
    distance = np.linalg.norm(img_array[:, :, :3] - bg_color, axis=2)
    mask = distance <= tolerance
    img_array[mask] = [255, 255, 255]
    masked_image = Image.fromarray(img_array)
    print("image is white background")
    return masked_image

def transform_replace_background(image):
    """
    Wrapper function to replace background with white.
    """
    return replace_background_with_white(image, tolerance=70)

def data_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),

        # # ------ 去背功能 (若不需去背，可以註解這行) ------
        # transforms.Lambda(transform_replace_background),

        # # ------ 銳化功能 (若不需銳化，可以註解或移除這行) ------
        # transforms.Lambda(transform_sharpen_image),
        # transforms.Lambda(transform_enhance_color_and_contrast),
        # transforms.Lambda(transform_scribble_image),

        transforms.Lambda(transform_enhance_color_and_contrast),
        transforms.Lambda(transform_sharpen_image),
        transforms.Lambda(transform_replace_background),
        transforms.Lambda(transform_scribble_image),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_feats_and_meta(dloader: DataLoader, model: torch.nn.Module, device: str, ignore_feats: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
    all_feats = None
    labels = []
    camids = []

    for img, lbl, meta, _ in tqdm(dloader, desc="Extracting features"):
        with torch.no_grad():
            feats = None
            if not ignore_feats:
                out = model(img.to(device))['image_features']
                feats = out.detach().cpu().numpy()
            if all_feats is None:
                all_feats = feats
            else:
                all_feats = np.concatenate((all_feats, feats), axis=0) if feats is not None else all_feats

        labels.extend(lbl.detach().cpu().numpy().tolist())
        camids.extend(list(meta))
        
    labels = np.array(labels)
    return all_feats, labels, camids

def _filter(dataframe: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    bad_row_idxs = []
    
    for idx, row in tqdm(dataframe.iterrows(), desc="Filtering bad urls"):
        fname = row['filename']
        path = os.path.join(img_dir, fname)
    
        if not os.path.exists(path):
            print(f"File not found: {path}")
            bad_row_idxs.append(idx)
        else:
            try:
                Image.open(path)
            except Exception as e:
                print(f"Error opening {path}: {e}")
                bad_row_idxs.append(idx)

    print(f"Bad rows: {len(bad_row_idxs)}")

    return dataframe.drop(bad_row_idxs)

def load_data(data_path: str, img_dir: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _filter(pd.read_csv(data_path), img_dir)
    train_data, test_data = train_test_split(df, test_size=test_size, stratify=df["hybrid_stat"], random_state=random_state)
    
    return train_data, test_data

