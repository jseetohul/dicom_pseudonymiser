"""
validate_release.py

Validation script for DICOM pseudonymisation + burned-in pixel redaction.

IMPORTANT:
On Windows, never use paths like "C:/Users\..." inside Python strings.
Use forward slashes instead: C:/Users/...
"""



from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset


Rect = Tuple[int, int, int, int]  # (x, y, w, h)


# -----------------------------
# Policy loading (YAML/JSON)
# -----------------------------
def load_policy(policy_path: Path) -> Dict[str, Any]:
    text = policy_path.read_text(encoding="utf-8")
    suf = policy_path.suffix.lower()
    if suf == ".json":
        return json.loads(text)
    if suf in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML not installed. Run: python -m pip install pyyaml") from e
        return yaml.safe_load(text)
    raise ValueError("Policy must be .json or .yaml/.yml")


# -----------------------------
# Helpers
# -----------------------------
def read_dicom_safe(path: Path, stop_before_pixels: bool = False) -> Optional[Dataset]:
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)
    except Exception:
        return None


def parse_rect(s: str) -> Rect:
    # "x,y,w,h"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Bad rect '{s}'. Use x,y,w,h")
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


def get_all_uids(ds: Dataset) -> List[str]:
    """Return list of all UI values (including nested sequences)"""
    uids: List[str] = []
    for elem in ds.iterall():
        if elem.VR == "UI":
            try:
                if elem.VM and elem.VM > 1:
                    uids.extend([str(v) for v in elem.value])
                else:
                    uids.append(str(elem.value))
            except Exception:
                continue
    return [u.strip() for u in uids if u and str(u).strip()]


def keyword_is_blank(ds: Dataset, kw: str) -> bool:
    """True if kw not present OR present but empty/blank."""
    if kw not in ds:
        return True
    try:
        v = ds.get(kw, "")
        if v is None:
            return True
        return str(v).strip() == ""
    except Exception:
        return False


def keyword_is_deleted(ds: Dataset, kw: str) -> bool:
    """True if kw not present."""
    return kw not in ds


def patient_pseudonym_ok(ds: Dataset, prefix: str) -> bool:
    pid = str(ds.get("PatientID", "")).strip()
    pname = str(ds.get("PatientName", "")).strip()
    if not pid.startswith(prefix):
        return False
    # Many pipelines set PatientName = PatientID pseudonym (you do)
    if pname != pid:
        return False
    return True


# -----------------------------
# Pixel redaction check
# -----------------------------
def _roi_stats(arr: np.ndarray, rect: Rect) -> Tuple[float, float]:
    x, y, w, h = rect

    # Determine frame handling
    if arr.ndim == 2:
        roi = arr[y:y+h, x:x+w]
    elif arr.ndim == 3:
        # either (H,W,3) OR (F,H,W)
        if arr.shape[-1] == 3:
            roi = arr[y:y+h, x:x+w, :]
        else:
            # treat as frames grayscale; use frame 0 by default
            roi = arr[0, y:y+h, x:x+w]
    elif arr.ndim == 4:
        # (F,H,W,3) -> use frame 0 by default here
        roi = arr[0, y:y+h, x:x+w, :]
    else:
        raise ValueError(f"Unsupported pixel shape: {arr.shape}")

    roi = np.asarray(roi)
    return float(np.min(roi)), float(np.max(roi))


def burned_in_redaction_ok(ds: Dataset, rects: List[Rect], tol_max: float = 2.0) -> Tuple[bool, str]:
    """
    Check whether each rect region is masked (near 0) on a few frames.
    tol_max allows a tiny tolerance, but with forced uncompressed output it should be 0.
    """
    try:
        arr = ds.pixel_array
    except Exception as e:
        return False, f"pixel_array decode failed: {e}"

    # Choose frames to sample for multi-frame data (start/middle/end)
    frame_indices: List[int] = [0]
    if arr.ndim in (3, 4) and arr.shape[0] >= 3 and (arr.shape[-1] == 3 or arr.ndim == 3):
        # If (F,H,W) or (F,H,W,3), sample middle + last
        frame_indices = [0, arr.shape[0] // 2, arr.shape[0] - 1]

    # Validate for each rect across sampled frames
    for rect in rects:
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            continue

        for fi in frame_indices:
            # Extract ROI for frame fi robustly
            if arr.ndim == 2:
                roi = arr[y:y+h, x:x+w]
            elif arr.ndim == 3:
                if arr.shape[-1] == 3:
                    roi = arr[y:y+h, x:x+w, :]
                else:
                    roi = arr[fi, y:y+h, x:x+w]
            elif arr.ndim == 4:
                roi = arr[fi, y:y+h, x:x+w, :]
            else:
                return False, f"Unsupported pixel shape: {arr.shape}"

            roi = np.asarray(roi)
            if roi.size == 0:
                return False, f"ROI empty for rect {rect} (check bounds)"

            mx = float(np.max(roi))
            if mx > tol_max:
                return False, f"Rect {rect} frame {fi} ROI max={mx} (not masked)"
    return True, "OK"


# -----------------------------
# Validation result container
# -----------------------------
@dataclass
class FileResult:
    rel_path: str
    out_path: str
    raw_exists: bool
    readable: bool
    pass_all: bool
    failures: str
    transfer_syntax: str
    patient_ok: bool
    tags_ok: bool
    dates_ok: bool
    uids_ok: bool
    pixel_ok: bool


# -----------------------------
# Core validation per file
# -----------------------------
def validate_one(
    raw_path: Optional[Path],
    out_path: Path,
    policy: Dict[str, Any],
    rects: List[Rect],
) -> FileResult:
    ds_out = read_dicom_safe(out_path, stop_before_pixels=False)
    if ds_out is None:
        return FileResult(
            rel_path="",
            out_path=str(out_path),
            raw_exists=bool(raw_path and raw_path.exists()),
            readable=False,
            pass_all=False,
            failures="Unreadable DICOM output",
            transfer_syntax="",
            patient_ok=False,
            tags_ok=False,
            dates_ok=False,
            uids_ok=False,
            pixel_ok=False,
        )

    # Transfer syntax
    ts = ""
    try:
        ts = str(ds_out.file_meta.TransferSyntaxUID)
    except Exception:
        ts = ""

    pseud_prefix = str(policy.get("pseudonym_prefix", "P")).strip() or "P"
    patient_ok = patient_pseudonym_ok(ds_out, pseud_prefix)

    # Policy-driven tag checks
    delete_list = policy.get("delete", []) or []
    blank_list = policy.get("blank", []) or []
    dates_cfg = policy.get("dates", {}) or {}
    date_include = dates_cfg.get("include", []) or []
    date_mode = str(dates_cfg.get("mode", "blank")).lower()

    tag_failures: List[str] = []

    # Deleted must be absent
    for kw in delete_list:
        if not keyword_is_deleted(ds_out, kw):
            tag_failures.append(f"Expected deleted but present: {kw}")

    # Blank must be empty (or absent)
    for kw in blank_list:
        if not keyword_is_blank(ds_out, kw):
            tag_failures.append(f"Expected blank but not blank: {kw}")

    tags_ok = len(tag_failures) == 0

    # Date checks:
    # - If mode blank, included date/time keywords should be blank (or absent)
    # - If mode offset, they may not be blank; we only check they are not identical to raw if raw exists
    date_failures: List[str] = []
    if date_include:
        if date_mode == "blank":
            for kw in date_include:
                if not keyword_is_blank(ds_out, kw):
                    date_failures.append(f"Expected date blank but not blank: {kw}")
        else:
            # offset mode: only validate vs raw if available
            if raw_path and raw_path.exists():
                ds_raw_head = read_dicom_safe(raw_path, stop_before_pixels=True)
                if ds_raw_head:
                    for kw in date_include:
                        if kw in ds_out and kw in ds_raw_head:
                            if str(ds_out.get(kw, "")).strip() == str(ds_raw_head.get(kw, "")).strip() and str(ds_out.get(kw, "")).strip():
                                date_failures.append(f"Date looks unchanged under offset mode: {kw}")
    dates_ok = len(date_failures) == 0

    # UID checks: ensure key UIDs differ from raw (and no UI overlap, sampled)
    uids_ok = True
    uid_failures: List[str] = []
    if raw_path and raw_path.exists():
        ds_raw = read_dicom_safe(raw_path, stop_before_pixels=True)
        ds_out_head = read_dicom_safe(out_path, stop_before_pixels=True)
        if ds_raw and ds_out_head:
            # Check major UIDs differ if present
            for kw in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
                if kw in ds_raw and kw in ds_out_head:
                    if str(ds_raw.get(kw, "")).strip() == str(ds_out_head.get(kw, "")).strip():
                        uid_failures.append(f"{kw} unchanged")

            # Check there isn't a large overlap of UI values (quick leak detector)
            raw_uids = set(get_all_uids(ds_raw))
            out_uids = set(get_all_uids(ds_out_head))
            overlap = raw_uids.intersection(out_uids)
            # Allow overlap to be empty; if not empty, flag (can be strict)
            if len(overlap) > 0:
                # sometimes SOPClassUID will overlap (it SHOULD), but SOPClassUID is VR=UI too.
                # We can ignore SOPClassUID by removing it from sets if present.
                sop_class_out = str(ds_out_head.get("SOPClassUID", "")).strip()
                sop_class_raw = str(ds_raw.get("SOPClassUID", "")).strip()
                if sop_class_out and sop_class_raw and sop_class_out == sop_class_raw:
                    overlap.discard(sop_class_out)

            if len(overlap) > 0:
                uid_failures.append(f"UID overlap detected (possible leakage): {list(sorted(overlap))[:3]} ...")
        # if we can't read raw/out head, keep uids_ok true but note nothing
    if uid_failures:
        uids_ok = False

    # Pixel redaction checks
    pixel_ok = True
    pixel_msg = "OK"
    if rects:
        pixel_ok, pixel_msg = burned_in_redaction_ok(ds_out, rects=rects, tol_max=2.0)

    failures = []
    if not patient_ok:
        failures.append("PatientID/PatientName not pseudonymised correctly")
    failures.extend(tag_failures)
    failures.extend(date_failures)
    failures.extend(uid_failures)
    if rects and not pixel_ok:
        failures.append(f"Pixel redaction failed: {pixel_msg}")

    pass_all = (patient_ok and tags_ok and dates_ok and uids_ok and (pixel_ok if rects else True))

    return FileResult(
        rel_path="",
        out_path=str(out_path),
        raw_exists=bool(raw_path and raw_path.exists()),
        readable=True,
        pass_all=pass_all,
        failures=" | ".join(failures) if failures else "",
        transfer_syntax=ts,
        patient_ok=patient_ok,
        tags_ok=tags_ok,
        dates_ok=dates_ok,
        uids_ok=uids_ok,
        pixel_ok=(pixel_ok if rects else True),
    )


# -----------------------------
# Main runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Validate DICOM pseudonymisation + burned-in redaction outputs")
    ap.add_argument("--raw_root", required=True, help="Root of raw/original DICOM tree (for comparison).")
    ap.add_argument("--out_root", required=True, help="Root of output/release DICOM tree to validate.")
    ap.add_argument("--policy", required=True, help="Policy YAML/JSON used for pseudonymisation.")
    ap.add_argument("--csv", required=True, help="Path to write CSV validation report.")
    ap.add_argument("--rects", action="append", default=[], help='Rectangle "x,y,w,h" (repeatable).')
    ap.add_argument("--require_uncompressed", action="store_true",
                    help="If set, fail files that are not Explicit VR Little Endian (recommended for pixel-redacted outputs).")

    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    policy_path = Path(args.policy)
    csv_path = Path(args.csv)

    rects: List[Rect] = [parse_rect(s) for s in (args.rects or [])]
    policy = load_policy(policy_path)

    results: List[FileResult] = []

    total = 0
    passed = 0
    failed = 0
    unreadable = 0

    for out_file in out_root.rglob("*"):
        if not out_file.is_file():
            continue
        total += 1

        rel = out_file.relative_to(out_root)
        raw_file = raw_root / rel

        res = validate_one(raw_path=raw_file, out_path=out_file, policy=policy, rects=rects)
        res.rel_path = str(rel)

        # Enforce uncompressed transfer syntax if requested
        if args.require_uncompressed:
            if res.readable:
                # Explicit VR Little Endian UID string
                if "1.2.840.10008.1.2.1" not in res.transfer_syntax:
                    res.pass_all = False
                    msg = f"Transfer syntax not ExplicitVRLittleEndian: {res.transfer_syntax}"
                    res.failures = (res.failures + " | " + msg).strip(" |")
            else:
                # already unreadable
                pass

        results.append(res)

        if not res.readable:
            unreadable += 1
        elif res.pass_all:
            passed += 1
        else:
            failed += 1

    # Write CSV report
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rel_path", "out_path", "raw_exists", "readable", "pass_all",
            "patient_ok", "tags_ok", "dates_ok", "uids_ok", "pixel_ok",
            "transfer_syntax", "failures"
        ])
        for r in results:
            w.writerow([
                r.rel_path, r.out_path, r.raw_exists, r.readable, r.pass_all,
                r.patient_ok, r.tags_ok, r.dates_ok, r.uids_ok, r.pixel_ok,
                r.transfer_syntax, r.failures
            ])

    print("\nVALIDATION SUMMARY")
    print("Raw root:", raw_root)
    print("Out root:", out_root)
    print("Policy:", policy_path)
    print("Rects:", rects if rects else "None")
    print("Total files scanned:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Unreadable:", unreadable)
    print("CSV report:", csv_path)

    # Print first few failures to help you fix quickly
    if failed > 0:
        print("\nFirst failures:")
        shown = 0
        for r in results:
            if r.readable and not r.pass_all:
                print("-", r.rel_path, "=>", r.failures)
                shown += 1
                if shown >= 10:
                    break


if __name__ == "__main__":
    main()
