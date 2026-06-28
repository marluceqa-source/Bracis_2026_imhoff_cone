# -*- coding: utf-8 -*-
"""
Image processing of the Imhoff cone and calibration of the settleable solids model.

Suggested dependencies:
    pip install opencv-python pillow matplotlib numpy pandas openpyxl rembg tqdm


"""

from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")


# =============================================================================
# GENERAL SETTINGS
# =============================================================================

@dataclass(frozen=True)
class PipelineConfig:
    """Main settings for the processing pipeline."""

    base_dir: Path = Path("/2_lote_V1.0")
    target_width: int = 461
    target_height: int = 1280
    crop_left_ratio: float = 0.22
    crop_right_ratio: float = 0.70
    standardization_mode: str = "central"  # central, proportional, or exact
    output_format: str = "PNG"
    jpeg_quality: int = 95
    white_background: bool = True
    total_cone_volume_ml: float = 1000.0
    show_images: bool = False

    @property
    def input_original(self) -> Path:
        return self.base_dir / "1_original_cone"

    @property
    def output_crop(self) -> Path:
        return self.base_dir / "2_crop"

    @property
    def output_white_background(self) -> Path:
        return self.base_dir / "3_white_background"

    @property
    def output_standardized(self) -> Path:
        return self.base_dir / "4_standardized_image"

    @property
    def output_sediment(self) -> Path:
        return self.base_dir / "5_sediments"

    @property
    def output_liquid_height(self) -> Path:
        return self.base_dir / "6_cone_height"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def mount_google_drive_if_needed() -> None:
    """Mounts Google Drive when the script is running in Colab."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        print("Colab environment not detected. Skipping Google Drive mount.")
        return

    drive_path = Path("/content/drive/MyDrive")
    if drive_path.exists():
        print("Google Drive is already mounted.")
        return

    print("Mounting Google Drive...")
    drive.mount("/content/drive")


def ensure_dir(path: Path) -> None:
    """Creates a directory if it does not exist yet."""
    path.mkdir(parents=True, exist_ok=True)


def list_images(directory: Path) -> list[Path]:
    """Lists valid images in a directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort()

    if not images:
        raise RuntimeError(f"No images found in: {directory}")

    return images


def show_image(image_rgb: np.ndarray | Image.Image, title: str, figsize: tuple[int, int] = (8, 12)) -> None:
    """Displays an image only when visual inspection is needed."""
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# STEP 1: CENTRAL CROP
# =============================================================================

def crop_center_images(config: PipelineConfig) -> list[Path]:
    """Crops the central region of the original images and saves it with a white background."""
    input_images = list_images(config.input_original)
    ensure_dir(config.output_crop)

    output_paths: list[Path] = []

    for image_path in tqdm(input_images, desc="Cropping images"):
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            left = int(width * config.crop_left_ratio)
            right = int(width * config.crop_right_ratio)
            cropped = image.crop((left, 0, right, height))

            canvas = Image.new("RGB", cropped.size, (255, 255, 255))
            canvas.paste(cropped, (0, 0))

            output_path = config.output_crop / f"{image_path.stem}_without_rectangle.png"
            canvas.save(output_path)
            output_paths.append(output_path)
        except Exception as exc:
            print(f"Error while cropping {image_path.name}: {exc}")

    print(f"Cropped images saved: {len(output_paths)}")
    return output_paths


# =============================================================================
# STEP 2: BACKGROUND REMOVAL AND WHITE BACKGROUND
# =============================================================================

def remove_background_to_white(input_dir: Path, output_dir: Path) -> list[Path]:
    """Removes the background with rembg and adds a white background."""
    from rembg import remove

    input_images = list_images(input_dir)
    ensure_dir(output_dir)

    output_paths: list[Path] = []

    for image_path in tqdm(input_images, desc="Removing background"):
        try:
            input_bytes = image_path.read_bytes()
            output_bytes = remove(input_bytes)

            image_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            white_bg = Image.new("RGBA", image_rgba.size, (255, 255, 255, 255))
            final_image = Image.alpha_composite(white_bg, image_rgba).convert("RGB")

            output_path = output_dir / f"{image_path.stem}_white_background.png"
            final_image.save(output_path)
            output_paths.append(output_path)
        except Exception as exc:
            print(f"Error while removing background from {image_path.name}: {exc}")

    print(f"Images with white background saved: {len(output_paths)}")
    return output_paths


# =============================================================================
# STEP 3: SIZE STANDARDIZATION
# =============================================================================

def standardize_image_size(
    image_path: Path,
    output_dir: Path,
    target_size: tuple[int, int],
    mode: str = "central",
    output_format: str = "PNG",
    jpeg_quality: int = 95,
    white_background: bool = True,
) -> Path:
    """Standardizes the image size by central crop, proportional resize, or exact resize."""
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    target_width, target_height = target_size

    if mode == "central":
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height

        if original_ratio > target_ratio:
            new_width = int(original_height * target_ratio)
            left = (original_width - new_width) / 2
            crop_box = (left, 0, left + new_width, original_height)
        else:
            new_height = int(original_width / target_ratio)
            top = (original_height - new_height) / 2
            crop_box = (0, top, original_width, top + new_height)

        image = image.crop(crop_box).resize(target_size, Image.LANCZOS)

    elif mode == "proportional":
        resized = image.copy()
        resized.thumbnail(target_size, Image.LANCZOS)

        if white_background:
            image = Image.new("RGB", target_size, (255, 255, 255))
            offset_x = (target_width - resized.size[0]) // 2
            offset_y = (target_height - resized.size[1]) // 2
            image.paste(resized, (offset_x, offset_y))
        else:
            image = resized

    elif mode == "exact":
        image = image.resize(target_size, Image.LANCZOS)

    else:
        raise ValueError("Invalid mode. Use: 'central', 'proportional', or 'exact'.")

    ensure_dir(output_dir)
    extension = output_format.lower()
    output_path = output_dir / f"{image_path.stem}_standardized.{extension}"

    if output_format.upper() == "JPEG":
        image.save(output_path, quality=jpeg_quality)
    else:
        image.save(output_path)

    return output_path


def standardize_directory(config: PipelineConfig) -> list[Path]:
    """Standardizes all images in the white-background folder."""
    input_images = list_images(config.output_white_background)
    target_size = (config.target_width, config.target_height)

    output_paths: list[Path] = []
    for image_path in tqdm(input_images, desc="Standardizing images"):
        try:
            output_path = standardize_image_size(
                image_path=image_path,
                output_dir=config.output_standardized,
                target_size=target_size,
                mode=config.standardization_mode,
                output_format=config.output_format,
                jpeg_quality=config.jpeg_quality,
                white_background=config.white_background,
            )
            output_paths.append(output_path)
        except Exception as exc:
            print(f"Error while standardizing {image_path.name}: {exc}")

    print(f"Standardized images saved: {len(output_paths)}")
    return output_paths


# =============================================================================
# STEP 4: SEDIMENT DETECTION
# =============================================================================

def process_sediment_image(
    image_path: Path,
    output_dir: Path,
    show: bool = False,
) -> Optional[dict[str, float | int | str]]:
    """Segments the dark sediment in the lower region and calculates its height in pixels."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Error while loading image: {image_path}")
        return None

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_region_mask = np.zeros((height, width), dtype=np.uint8)
    lower_region_mask[int(height * 0.65):height, :] = 255

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 75])
    sediment_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    sediment_mask = cv2.bitwise_and(sediment_mask, sediment_mask, mask=lower_region_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sediment_mask = cv2.morphologyEx(sediment_mask, cv2.MORPH_OPEN, kernel)
    sediment_mask = cv2.morphologyEx(sediment_mask, cv2.MORPH_CLOSE, kernel)

    ys, _ = np.where(sediment_mask > 0)

    y_min = -1
    y_max = -1
    sediment_height_px = 0
    sediment_volume_ml = 0.0

    if len(ys) > 0:
        y_min = int(ys.min())
        y_max = int(ys.max())
        sediment_height_px = y_max - y_min

        cone_top = int(height * 0.10)
        cone_base = height - 1
        cone_height_px = cone_base - cone_top

        if cone_height_px > 0:
            sediment_volume_ml = (sediment_height_px / cone_height_px) * 1000.0
    else:
        print(f"No sediment detected in: {image_path.name}")

    image_vis = image_rgb.copy()
    if y_min >= 0 and y_max >= 0:
        cv2.line(image_vis, (0, y_min), (width, y_min), (255, 0, 0), 3)
        cv2.line(image_vis, (0, y_max), (width, y_max), (255, 0, 0), 3)

    text = f"Volume: {sediment_volume_ml:.2f} mL"
    cv2.putText(image_vis, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

    ensure_dir(output_dir)
    output_path = output_dir / f"{image_path.stem}_sediment_visualized.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))

    if show:
        show_image(sediment_mask, f"Sediment mask - {image_path.name}")
        show_image(image_vis, f"Detected sediment - {image_path.name}")

    return {
        "filename": image_path.name,
        "sediment_volume_ml": sediment_volume_ml,
        "sediment_height_px": sediment_height_px,
        "sediment_visualization": output_path.name,
    }


def process_sediment_directory(config: PipelineConfig) -> pd.DataFrame:
    """Processes sediment detection in all standardized images."""
    input_images = list_images(config.output_standardized)
    results: list[dict[str, float | int | str]] = []

    for image_path in tqdm(input_images, desc="Detecting sediment"):
        result = process_sediment_image(image_path, config.output_sediment, show=config.show_images)
        if result is not None:
            results.append(result)

    df = pd.DataFrame(results)
    ensure_dir(config.output_liquid_height)
    output_csv = config.output_liquid_height / "sediment_results.csv"
    df.to_csv(output_csv, index=False, sep=";")

    print(f"Sediment results saved to: {output_csv}")
    return df


# =============================================================================
# STEP 5: CONE DETECTION AND LIQUID HEIGHT
# =============================================================================

def detect_cone_and_liquid_volume(
    image_path: Path,
    total_volume_ml: float = 1000.0,
) -> Optional[dict[str, object]]:
    """Detects the cone, the air-liquid interface, and calculates the approximate liquid volume."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    lab_b = lab[:, :, 2].astype(np.float32) - 128

    lower_bg = np.array([0, 0, 180])
    upper_bg = np.array([180, 50, 255])
    background_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    object_mask = cv2.bitwise_not(background_mask)

    kernel = np.ones((7, 7), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    obj_ys, obj_xs = np.where(object_mask > 0)
    cone_tip_y = int(obj_ys.max())
    cone_tip_x = int(obj_xs[obj_ys == cone_tip_y].mean())

    y_values = np.sort(np.unique(obj_ys))
    widths: list[int] = []
    centers: list[float] = []

    for y in y_values:
        x_values = obj_xs[obj_ys == y]
        widths.append(int(x_values.max() - x_values.min()))
        centers.append(float(x_values.mean()))

    max_width = max(widths)
    object_top_y = int(y_values[0])
    object_bottom_y = int(y_values[-1])
    object_height = object_bottom_y - object_top_y

    mouth_candidates = []
    for y, row_width, center_x in zip(y_values, widths, centers):
        is_wide = row_width > max_width * 0.5
        is_near_top = y < object_top_y + object_height * 0.4
        if is_wide and is_near_top:
            mouth_candidates.append((int(y), row_width, center_x))

    if mouth_candidates:
        mouth_y, mouth_width, mouth_x = min(mouth_candidates, key=lambda item: item[0])
    else:
        widest_index = int(np.argmax(widths))
        mouth_y = int(y_values[widest_index])
        mouth_width = widths[widest_index]
        mouth_x = centers[widest_index]

    cone_mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(mouth_y, cone_tip_y + 1):
        denominator = cone_tip_y - mouth_y
        t = (y - mouth_y) / denominator if denominator > 0 else 0
        half_width = (mouth_width / 2) * (1 - t)
        x_left = max(0, int(cone_tip_x - half_width))
        x_right = min(width - 1, int(cone_tip_x + half_width))
        cone_mask[y, x_left:x_right + 1] = 255

    cone_height_px = cone_tip_y - mouth_y + 1

    edges = cv2.Canny(gray, 30, 100)
    edges_in_cone = cv2.bitwise_and(edges, edges, mask=cone_mask)
    lines = cv2.HoughLinesP(
        edges_in_cone,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=int(width * 0.1),
        maxLineGap=15,
    )

    interface_y: Optional[int] = None
    horizontal_lines: list[tuple[int, int]] = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 8 and x2 - x1 > width * 0.15:
                y_average = (y1 + y2) // 2
                is_valid_region = mouth_y + cone_height_px * 0.1 < y_average < mouth_y + cone_height_px * 0.8
                if is_valid_region:
                    horizontal_lines.append((int(y_average), abs(x2 - x1)))

    if horizontal_lines:
        horizontal_lines.sort(key=lambda item: item[1], reverse=True)
        interface_y = horizontal_lines[0][0]

    if interface_y is None:
        interface_y = _find_interface_by_color_features(
            cone_mask=cone_mask,
            mouth_y=mouth_y,
            cone_tip_y=cone_tip_y,
            cone_height_px=cone_height_px,
            saturation=saturation,
            value=value,
            lab_b=lab_b,
        )

    if interface_y is None:
        return None

    liquid_height_px = cone_tip_y - interface_y + 1
    ratio = liquid_height_px / cone_height_px if cone_height_px > 0 else 0
    liquid_volume_ml = total_volume_ml * (ratio ** 3)

    clean_image = np.ones_like(image_bgr) * 255
    clean_image[cone_mask > 0] = image_bgr[cone_mask > 0]

    image_vis = image_bgr.copy()
    cv2.line(image_vis, (0, interface_y), (width, interface_y), (255, 0, 0), 3)
    cv2.line(image_vis, (0, cone_tip_y), (width, cone_tip_y), (0, 0, 255), 3)
    cv2.line(image_vis, (0, mouth_y), (width, mouth_y), (0, 255, 255), 2)
    cv2.line(image_vis, (width // 2, interface_y), (width // 2, cone_tip_y), (0, 255, 0), 3)

    cv2.putText(
        image_vis,
        f"Volume: {liquid_volume_ml:.1f} mL / {total_volume_ml:.0f} mL",
        (10, max(30, interface_y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_vis,
        f"h_liq={liquid_height_px}px | h_cone={cone_height_px}px",
        (10, max(60, interface_y - 45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return {
        "filename": image_path.name,
        "liquid_volume_ml": liquid_volume_ml,
        "liquid_height_px": liquid_height_px,
        "cone_height_px": cone_height_px,
        "visualization_bgr": image_vis,
        "clean_image_bgr": clean_image,
        "cone_mask": cone_mask,
    }


def _find_interface_by_color_features(
    cone_mask: np.ndarray,
    mouth_y: int,
    cone_tip_y: int,
    cone_height_px: int,
    saturation: np.ndarray,
    value: np.ndarray,
    lab_b: np.ndarray,
) -> Optional[int]:
    """Alternative search for the air-liquid interface based on color variations."""
    features_by_y: list[dict[str, float]] = []

    for y in range(mouth_y, cone_tip_y + 1):
        cone_xs = np.where(cone_mask[y, :] > 0)[0]
        if len(cone_xs) <= 10:
            continue

        start = int(len(cone_xs) * 0.2)
        end = int(len(cone_xs) * 0.8)
        center_xs = cone_xs[start:end]

        features_by_y.append({
            "y": float(y),
            "sat": float(saturation[y, center_xs].mean()),
            "val": float(value[y, center_xs].mean()),
            "lab_b": float(lab_b[y, center_xs].mean()),
        })

    if len(features_by_y) < 10:
        return None

    ys = np.array([f["y"] for f in features_by_y])
    sats = np.array([f["sat"] for f in features_by_y])
    vals = np.array([f["val"] for f in features_by_y])
    lab_bs = np.array([f["lab_b"] for f in features_by_y])

    best_score = 0.0
    best_y: Optional[int] = None

    for i, y in enumerate(ys):
        above_idx = np.where(ys <= y - 20)[0]
        below_idx = np.where(ys >= y + 20)[0]

        if len(above_idx) == 0 or len(below_idx) == 0:
            continue

        ia = above_idx[-1]
        ib = below_idx[0]
        score = (
            abs(sats[ia] - sats[ib]) * 0.5
            + abs(vals[ia] - vals[ib]) * 0.3
            + abs(lab_bs[ia] - lab_bs[ib]) * 0.2
        )

        is_valid_region = mouth_y + cone_height_px * 0.1 < y < mouth_y + cone_height_px * 0.8
        if is_valid_region and score > best_score:
            best_score = float(score)
            best_y = int(y)

    return best_y


def process_liquid_height_directory(config: PipelineConfig) -> pd.DataFrame:
    """Processes the liquid height in the sediment visualization images."""
    input_images = list_images(config.output_sediment)
    ensure_dir(config.output_liquid_height)

    results: list[dict[str, float | int | str]] = []

    for image_path in tqdm(input_images, desc="Detecting cone and liquid"):
        result = detect_cone_and_liquid_volume(image_path, config.total_cone_volume_ml)
        if result is None:
            print(f"Detection failed: {image_path.name}")
            continue

        visualization_bgr = result.pop("visualization_bgr")
        clean_image_bgr = result.pop("clean_image_bgr")
        result.pop("cone_mask")

        output_vis = config.output_liquid_height / f"{image_path.stem}_liquid_height_vis.png"
        output_clean = config.output_liquid_height / f"{image_path.stem}_isolated_cone.png"

        cv2.imwrite(str(output_vis), visualization_bgr)  # type: ignore[arg-type]
        cv2.imwrite(str(output_clean), clean_image_bgr)  # type: ignore[arg-type]

        result["liquid_visualization"] = output_vis.name
        results.append(result)  # type: ignore[arg-type]

        if config.show_images:
            show_image(cv2.cvtColor(visualization_bgr, cv2.COLOR_BGR2RGB), f"Liquid height - {image_path.name}", (15, 5))  # type: ignore[arg-type]

    df = pd.DataFrame(results)
    output_csv = config.output_liquid_height / "liquid_height_results.csv"
    df.to_csv(output_csv, index=False, sep=";")

    print(f"Liquid height results saved to: {output_csv}")
    return df


# =============================================================================
# STEP 6: MERGING HEIGHT RESULTS
# =============================================================================

def merge_height_results(
    sediment_df: pd.DataFrame,
    liquid_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Merges sediment height and cone/liquid height into a final spreadsheet."""
    if sediment_df.empty or liquid_df.empty:
        print("Could not merge results: at least one DataFrame is empty.")
        return pd.DataFrame()

    sediment = sediment_df.copy()
    liquid = liquid_df.copy()

    sediment["filename_base"] = sediment["filename"].astype(str).str.replace("_standardized.png", "", regex=False)
    sediment["filename_base"] = sediment["filename_base"].str.replace(".png", "", regex=False)

    liquid["filename_base"] = liquid["filename"].astype(str).str.replace("_sediment_visualized.png", "", regex=False)
    liquid["filename_base"] = liquid["filename_base"].str.replace("_standardized", "", regex=False)

    merged = pd.merge(
        sediment[["filename", "filename_base", "sediment_height_px"]],
        liquid[["filename_base", "liquid_height_px", "cone_height_px"]],
        on="filename_base",
        how="inner",
    )

    output_excel = output_dir / "final_liquid_sediment_height_results.xlsx"
    merged.to_excel(output_excel, index=False, sheet_name="Heights")

    print(f"Final spreadsheet saved to: {output_excel}")
    return merged


# =============================================================================
# STEP 7: CALIBRATION MODEL AND EVALUATION
# =============================================================================

BATCH_1_DATA = [
    ["1_lote1", 1166, 230, 60.4],
    ["2_lote1", 1133, 199, 45],
    ["3_lote1", 1085, 61, 6],
    ["4_lote1", 1043, 71, 6.1],
    ["5_lote1", 1051, 102, 17],
    ["9_lote1", 1226, 164, 10],
    ["10_lote1", 1029, 69, 12],
    ["11_lote1", 1058, 146, 31],
    ["12_lote1", 1127, 91, 8],
    ["13_lote1", 1146, 117, 14],
    ["14_lote1", 1113, 236, 43],
    ["15_lote1", 1214, 100, 9],
    ["17_lote1", 1074, 59, 6],
    ["18_lote1", 1017, 105, 20],
    ["19_lote1", 1105, 161, 21],
    ["20_lote1", 1166, 154, 21],
    ["22_lote1", 1100, 147, 17],
    ["23_lote1", 1079, 99, 10],
    ["24_lote1", 1114, 239, 58],
]

BATCH_2_DATA = [
    ["1_lote2", 1173, 154, 25],
    ["2_lote2", 1131, 115, 22],
    ["3_lote2", 1219, 67, 8],
    ["4_lote2", 1181, 64, 7],
    ["5_lote2", 1166, 104, 23],
    ["6_lote2", 1179, 173, 42],
    ["7_lote2", 1216, 112, 13],
    ["8_lote2", 1206, 116, 21],
    ["9_lote2", 1208, 100, 13],
    ["10_lote2", 1156, 105, 16],
    ["11_lote2", 1016, 113, 30],
    ["12_lote2", 1165, 70, 9],
    ["13_lote2", 1102, 84, 28],
    ["14_lote2", 1253, 159, 41],
    ["15_lote2", 1177, 82, 12],
    ["17_lote2", 1190, 65, 5],
    ["18_lote2", 1189, 113, 21],
    ["19_lote2", 1250, 96, 15],
    ["20_lote2", 1198, 146, 40],
    ["21_lote2", 1160, 52, 6],
    ["22_lote2", 1056, 92, 34],
    ["23_lote2", 1129, 47, 11],
    ["24_lote2", 1201, 129, 30],
]

# Global model used in the latest code version.
MODEL_A = 444.13
MODEL_B = 1.3845


def build_dataset(data: list[list[object]], batch_name: str) -> pd.DataFrame:
    """Converts a list of samples into a standardized DataFrame."""
    df = pd.DataFrame(data, columns=["filename", "px_total", "px_sediment", "SS_measured"])
    df["batch"] = batch_name
    return df


def calculate_model_metrics(
    data: list[list[object]] | pd.DataFrame,
    name: str,
    a: float = MODEL_A,
    b: float = MODEL_B,
    output_dir: Path | str = Path("."),
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Applies the power model and calculates R², MAE, RMSE, and residuals."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=["filename", "px_total", "px_sediment", "SS_measured"])

    df["x"] = df["px_sediment"] / df["px_total"]
    df["SS_estimated"] = a * (df["x"] ** b)
    df["residual"] = df["SS_measured"] - df["SS_estimated"]
    df["abs_error"] = df["residual"].abs()
    df["squared_error"] = df["residual"] ** 2

    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(df["squared_error"].mean()))
    ss_res = float(np.sum((df["SS_measured"] - df["SS_estimated"]) ** 2))
    ss_tot = float(np.sum((df["SS_measured"] - df["SS_measured"].mean()) ** 2))
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}

    residual_columns = [
        "filename",
        "px_total",
        "px_sediment",
        "x",
        "SS_measured",
        "SS_estimated",
        "residual",
        "abs_error",
    ]
    if "batch" in df.columns:
        residual_columns.insert(1, "batch")

    residuals = df[residual_columns]
    residuals.to_csv(output_dir / f"residual_table_{name}.csv", index=False, sep=";")

    print(f"\nResults - {name}")
    print(f"Applied equation: SS = {a:.2f} * x^{b:.4f}")
    print(f"R²   = {r2:.4f}")
    print(f"MAE  = {mae:.4f} mL/L")
    print(f"RMSE = {rmse:.4f} mL/L")

    return df, metrics


def plot_log_log_regression(
    df: pd.DataFrame,
    name: str,
    output_dir: Path | str,
    a: float = MODEL_A,
    b: float = MODEL_B,
) -> Path:
    """Generates a log-log chart of the model applied to the data."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    x_curve = np.linspace(float(df["x"].min()), float(df["x"].max()), 200)
    y_curve = a * (x_curve ** b)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["x"], df["SS_measured"], label=f"Experimental data - {name}")
    plt.plot(x_curve, y_curve, linewidth=2, label=f"Model: SS = {a:.2f}x^{b:.4f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x = px_sediment / px_total")
    plt.ylabel("Measured SS (mL/L)")
    plt.title(f"Log-log regression - {name}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    output_path = output_dir / f"log_log_chart_{name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def run_model_evaluation(output_dir: Path | str) -> pd.DataFrame:
    """Runs the evaluation for Batch 1, Batch 2, and the global dataset."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    batch_1 = build_dataset(BATCH_1_DATA, "Batch 1")
    batch_2 = build_dataset(BATCH_2_DATA, "Batch 2")
    global_df = pd.concat([batch_1, batch_2], ignore_index=True)

    datasets = {
        "batch_1_19_samples": batch_1,
        "batch_2_23_samples": batch_2,
        "global_42_samples": global_df,
    }

    summary_rows = []
    for name, df in datasets.items():
        evaluated_df, metrics = calculate_model_metrics(df, name, output_dir=output_dir)
        plot_log_log_regression(evaluated_df, name, output_dir=output_dir)

        summary_rows.append({
            "Dataset": name,
            "Samples": len(df),
            "R2": metrics["R2"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "global_model_comparison.csv", index=False, sep=";")
    summary.to_excel(output_dir / "global_model_comparison.xlsx", index=False, sheet_name="Metrics")

    print("\nFinal comparison")
    print(summary.round(4))
    return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_full_pipeline(config: PipelineConfig) -> None:
    """Runs the complete image and model pipeline."""
    mount_google_drive_if_needed()

    crop_center_images(config)
    remove_background_to_white(config.output_crop, config.output_white_background)
    standardize_directory(config)

    sediment_df = process_sediment_directory(config)
    liquid_df = process_liquid_height_directory(config)
    merge_height_results(sediment_df, liquid_df, config.output_liquid_height)

    run_model_evaluation(config.output_liquid_height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Imhoff cone processing pipeline.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("./BRACIS2026_Alg/2_lote_V1.0"),
        help="Base directory containing the folders 1_original_cone, 2_crop, etc.",
    )
    parser.add_argument(
        "--only-model",
        action="store_true",
        help="Runs only the model evaluation, without processing images.",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Displays intermediate images during processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(base_dir=args.base_dir, show_images=args.show_images)

    if args.only_model:
        run_model_evaluation(config.output_liquid_height)
    else:
        run_full_pipeline(config)


if __name__ == "__main__":
    main()
