# dicom_pseudonymiser
# Ultrasound DICOM Pseudonymisation Pipeline : Version 1

Author: Jenna Seetohul 
Institution: UCLH / Research Use  
Modality: Ultrasound (GE Vivid S70 TTE)

---

## Overview

This repository provides a two-stage pipeline for preparing DICOM datasets for research use:

1. Metadata pseudonymisation with reversible mapping (SQLite)
2. Burned-in pixel redaction for ultrasound cine loops

The process aligns broadly with DICOM PS3.15 confidentiality profiles and UK GDPR expectations.

Only the final pixel-redacted output should be released externally.

---

## Pipeline

    Raw DICOM  
        ↓  
pseudonymise_bulk_sqlite_v2.py  
        ↓  
data_pseudonymised_v2  
        ↓  
pixel_redact_burnedin.py  
        ↓  
data_pixelredacted_v2  (FINAL DATASET)

---

## Stage 1 – Metadata Pseudonymisation

Script: pseudonymise_bulk_sqlite_v2.py

Features:

- Replaces PatientID and PatientName with pseudonym (Pxxxxxxxx)
- Stores real↔pseudo mapping in SQLite
- Deterministic UID regeneration (Study/Series/SOP)
- Recursive UID replacement in nested sequences
- Policy-driven delete/blank rules (policy.yaml)
- Private tag removal or allowlist
- Overlay (60xx) removal

Re-identification is possible only via the local SQLite database.

---

## Stage 2 – Burned-In Pixel Redaction

Script: pixel_redact_burnedin.py

Features:

- Rectangle-based pixel masking
- Supports multi-frame RGB ultrasound
- Applies to all frames
- Forces Explicit VR Little Endian output
- Removes overlay planes
- Sets BurnedInAnnotation = NO

Rectangles must be validated visually per vendor/export format.

---

## Requirements

Python 3.10+

Packages:

- pydicom
- numpy
- pyyaml

Optional (for JPEG):

- pylibjpeg
- gdcm

---

## Usage (PowerShell)

Set secrets:

```powershell
setx PSEUD_SALT "strong-secret"
setx UID_SALT "strong-secret"
