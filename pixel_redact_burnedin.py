"""
pixel_redact_burnedin.py

Burned-in pixel redaction for DICOM files (multi-frame supported).

Designed for ultrasound cine loops (e.g. GE Vivid S70) where PHI is burned into
top-left/top-right corners, etc.

Key features:
- Works for MONOCHROME and RGB frames
- Works for single-frame and multi-frame
- Applies same rectangles to all frames
- Writes output with ExplicitVRLittleEndian (uncompressed) to avoid "encapsulation" errors
- Optionally removes overlays (60xx groups)
- Does NOT touch metadata PHI (use pseudonymise_bulk_sqlite_v2.py for that)

IMPORTANT:
- Burned-in redaction is modality/vendor dependent. Rectangles MUST be validated visually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.pixel_data_handlers.util import apply_color_lut, convert_color_space


Rect = Tuple[int, int, int, int]  # (x, y, w, h)


# -----------------------------
# Optional: overlay removal
# -----------------------------
def remove_overlays_60xx(ds: Dataset) -> None:
    """
    Remove overlay/graphics elements in groups 60xx.
    This does NOT change PixelData; it removes overlay planes/metadata.
    """
    for group in range(0x6000, 0x6020, 2):
        tags = [elem.tag for elem in ds.iterall() if elem.tag.group == group]
        for tag in tags:
            if tag in ds:
                del ds[tag]


# -----------------------------
# Force uncompressed output
# -----------------------------
def force_uncompressed_explicit_vr_le(ds: Dataset) -> None:
    """
    Ensure dataset will be written as uncompressed Explicit VR Little Endian.

    This avoids the common error:
      "Pixel Data hasn't been encapsulated as required for a compressed transfer syntax"
    when you modify PixelData but keep a compressed TransferSyntaxUID.
    """
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()

    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False


# -----------------------------
# Core pixel redaction
# -----------------------------
def _fill_value_for(arr: np.ndarray, fill: str) -> int:
    """
    Choose fill value: black=0, white=max.
    """
    if fill.lower() == "white":
        # For uint images, use max value
        return int(arr.max()) if np.issubdtype(arr.dtype, np.integer) else 255
    return 0


def redact_rectangles_in_pixels(arr: np.ndarray, rects: List[Rect], fill: str = "black") -> np.ndarray:
    """
    Apply rectangle redaction to a pixel array.

    Supports shapes:
      - (rows, cols)                          grayscale single-frame
      - (frames, rows, cols)                  grayscale multi-frame
      - (rows, cols, 3)                       RGB single-frame
      - (frames, rows, cols, 3)               RGB multi-frame
    """
    fill_val = _fill_value_for(arr, fill)

    # Identify image size (H, W)
    if arr.ndim == 2:
        H, W = arr.shape
    elif arr.ndim == 3:
        # Either (H, W, 3) or (F, H, W)
        if arr.shape[-1] == 3:
            H, W = arr.shape[0], arr.shape[1]
        else:
            H, W = arr.shape[1], arr.shape[2]
    elif arr.ndim == 4:
        H, W = arr.shape[1], arr.shape[2]
    else:
        raise ValueError(f"Unsupported pixel array shape: {arr.shape}")

    # Apply each rectangle
    for (x, y, w, h) in rects:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, x0 + int(w))
        y1 = min(H, y0 + int(h))

        if x1 <= x0 or y1 <= y0:
            continue

        if arr.ndim == 2:
            arr[y0:y1, x0:x1] = fill_val

        elif arr.ndim == 3:
            if arr.shape[-1] == 3:
                # (H, W, 3)
                arr[y0:y1, x0:x1, :] = fill_val
            else:
                # (F, H, W)
                arr[:, y0:y1, x0:x1] = fill_val

        elif arr.ndim == 4:
            # (F, H, W, 3)
            arr[:, y0:y1, x0:x1, :] = fill_val

    return arr


def decode_pixel_array(ds: Dataset) -> np.ndarray:
    """
    Decode ds.pixel_array, handling palette color if present.

    For ultrasound RGB, you usually get RGB already.
    """
    arr = ds.pixel_array

    # Palette color -> expand to RGB
    if getattr(ds, "PhotometricInterpretation", "") == "PALETTE COLOR":
        arr = apply_color_lut(arr, ds)

    return arr


def write_pixel_array_back(ds: Dataset, arr: np.ndarray) -> None:
    """
    Write modified pixel data back into ds safely.

    This keeps:
      - same frame count and shape
      - forces uncompressed ExplicitVRLittleEndian
      - updates key pixel attributes to match the array
    """
    # Ensure contiguous bytes
    arr = np.ascontiguousarray(arr)

    # Work out rows/cols and samples per pixel
    if arr.ndim == 2:
        rows, cols = arr.shape
        samples_per_pixel = 1
        photometric = "MONOCHROME2"
    elif arr.ndim == 3:
        if arr.shape[-1] == 3:
            rows, cols = arr.shape[0], arr.shape[1]
            samples_per_pixel = 3
            photometric = "RGB"
        else:
            # (F, H, W)
            rows, cols = arr.shape[1], arr.shape[2]
            samples_per_pixel = 1
            photometric = "MONOCHROME2"
    elif arr.ndim == 4:
        # (F, H, W, 3)
        rows, cols = arr.shape[1], arr.shape[2]
        samples_per_pixel = 3
        photometric = "RGB"
    else:
        raise ValueError(f"Unsupported pixel array shape: {arr.shape}")

    # Force uncompressed syntax before writing PixelData
    force_uncompressed_explicit_vr_le(ds)

    # Update DICOM pixel metadata to remain consistent
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = samples_per_pixel
    ds.PhotometricInterpretation = photometric

    if samples_per_pixel == 3:
        # RGB needs PlanarConfiguration for uncompressed
        ds.PlanarConfiguration = 0
    else:
        if "PlanarConfiguration" in ds:
            del ds.PlanarConfiguration

    # Choose a safe integer pixel type for output
    # For RGB 8-bit, uint8 is typical. For grayscale 8/16, preserve width.
    if samples_per_pixel == 3:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
    else:
        # Keep grayscale as uint16 for safety (common in ultrasound)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16, copy=False)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

    ds.PixelData = arr.tobytes()


def redact_burned_in_file(
    in_file: Path,
    out_file: Path,
    rects: List[Rect],
    fill: str = "black",
    remove_overlays: bool = True,
    set_burned_in_annotation_no: bool = True,
) -> None:
    """
    Redact burned-in PHI rectangles from one DICOM file and write to out_file.
    """
    ds = pydicom.dcmread(str(in_file))

    if remove_overlays:
        remove_overlays_60xx(ds)

    arr = decode_pixel_array(ds)
    arr = redact_rectangles_in_pixels(arr, rects=rects, fill=fill)

    write_pixel_array_back(ds, arr)

    if set_burned_in_annotation_no:
        ds.BurnedInAnnotation = "NO"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_file), write_like_original=False)


# -----------------------------
# Bulk runner
# -----------------------------
def bulk_redact_burned_in(
    input_root: Path,
    output_root: Path,
    rects: List[Rect],
    fill: str = "black",
    remove_overlays: bool = True,
) -> None:
    """
    Bulk redaction for a folder tree.
    Mirrors folder structure into output_root.
    """
    processed = 0
    skipped = 0
    failed = 0

    for path in input_root.rglob("*"):
        if not path.is_file():
            continue

        # Try reading as DICOM quickly
        try:
            ds_head = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            skipped += 1
            continue

        rel = path.relative_to(input_root)
        out_path = output_root / rel

        try:
            redact_burned_in_file(
                in_file=path,
                out_file=out_path,
                rects=rects,
                fill=fill,
                remove_overlays=remove_overlays,
            )
            processed += 1
        except Exception as e:
            failed += 1
            print("FAIL:", path, e)

    print("\nBurned-in redaction done")
    print("Processed:", processed)
    print("Skipped non-DICOM:", skipped)
    print("Failed:", failed)
    print("Output:", output_root)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example for your GE Vivid S70 cine:
    # pixel_array.shape was (frames, 708, 1016, 3)
    INPUT = Path(r"C:\Users\jseetohu\Documents\python\pydcanon\data_pseudonymised_v2")
    OUTPUT = Path(r"C:\Users\jseetohu\Documents\python\pydcanon\data_pixelredacted_v2")

    # Rectangles MUST be tuned visually for your dataset.
    # Format: (x, y, w, h)
    RECTS: List[Rect] = [
        (0, 0, 520, 170),      # top-left header
        (650, 0, 366, 170),    # top-right header
        (0, 620, 1016, 88),    # bottom strip (status text)
    ]

    bulk_redact_burned_in(INPUT, OUTPUT, RECTS, fill="black", remove_overlays=True)
