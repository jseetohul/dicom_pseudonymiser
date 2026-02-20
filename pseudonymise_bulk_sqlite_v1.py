"""
pseudonymise_bulk_sqlite_v1.py

IMPORTANT:
- This file handles metadata PHI/pseudonymisation. Burned-in pixel redaction is separate.
- Keep the SQLite DB secure. It is your re-identification key.

How to run (PowerShell):
  python pseudonymise_bulk_sqlite_v2.py --policy policy.yaml

Environment variables (recommended):
  setx PSEUD_SALT "a-strong-secret"
  setx UID_SALT   "another-strong-secret"
"""

from __future__ import annotations

# ---- Standard library imports ----
import argparse           # CLI interface: python script.py --policy policy.yaml
import base64             # encode hashed bytes into safe printable tokens
import hashlib            # SHA256 hash used for deterministic tokens / UID mapping
import hmac               # keyed hashing for deterministic pseudonyms 
import json               # allow JSON policy file
import os                 # read environment variables (salts)
import sqlite3            # local mapping DB for reversible pseudonymisation
from dataclasses import dataclass  # tidy config holder
from datetime import datetime, timedelta  # date parsing and date shifting
from pathlib import Path   
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import logging

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------
# DEFAULT PATHS (edit or pass via CLI)
# -----------------------------
PROJECT_ROOT = Path(r"C:\Users\jseetohu\Documents\python\pydcanon") #replace with the desired file path 
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_pseudonymised_v2"
DEFAULT_DB_PATH = PROJECT_ROOT / "pseudonym_map.db"


# -----------------------------
# POLICY LOADING
# The policy defines - tags to delete
#                    - tags to blank
#                    - date handling strategy (blank or per-patient offset)
#                    - UID handling
#                    - private tag handling (remove_all vs allowlist)
#                    - overlay handling (remove 60xx groups)
# -----------------------------
def load_policy(policy_path: Path) -> Dict[str, Any]:
    """
    Supports JSON (.json) or YAML (.yml/.yaml) if pyyaml is installed.
    """
    text = policy_path.read_text(encoding="utf-8")
    suffix = policy_path.suffix.lower()

    if suffix in [".json"]:
        return json.loads(text)

    if suffix in [".yml", ".yaml"]:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("YAML policy requested but PyYAML not installed. Run: pip install pyyaml") from e
        return yaml.safe_load(text)

    raise ValueError("Policy file must be .json or .yaml/.yml")


# -----------------------------
# SQLITE MAPPING STORE
# -----------------------------
def init_db(db_path: Path) -> None: # Creates db and core patients table if missing
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Stores the reversible mapping
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()

        #'real-patient_id' is the primary key
        #'pseudo_id' is the unique key for de-identified outputs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                real_patient_id TEXT PRIMARY KEY,
                pseudo_id        TEXT NOT NULL UNIQUE,
                patient_name     TEXT,
                birth_date       TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_pseudo_id ON patients(pseudo_id)")
        conn.commit()


def get_pseudo_by_real(db_path: Path, real_patient_id: str) -> Optional[str]: 
    #fetches a pseudonym if mapping exists
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT pseudo_id FROM patients WHERE real_patient_id=?", (real_patient_id,))
        row = cur.fetchone()
        return row[0] if row else None


def get_real_by_pseudo(db_path: Path, pseudo_id: str) -> Optional[Tuple[str, str, str]]:
    #used for restoring the pseudonym's original ID
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT real_patient_id, patient_name, birth_date
            FROM patients WHERE pseudo_id=?
            """,
            (pseudo_id,),
        )
        row = cur.fetchone()
        return (row[0], row[1] or "", row[2] or "") if row else None


def insert_mapping(db_path: Path, real_patient_id: str, pseudo_id: str, name: str, dob: str) -> None:
    #Insert a new mapping into identity vault in sqlite
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO patients(real_patient_id, pseudo_id, patient_name, birth_date)
            VALUES (?, ?, ?, ?)
            """,
            (real_patient_id, pseudo_id, name, dob),
        )
        conn.commit()


# -----------------------------
# PSEUDONYM GENERATION (DETERMINISTIC)
    """
    Generates a stable token from an input string using HMAC-SHA256 (8 characters).
    - HMAC makes tokens which are harder to decrypt, hence with a secret salt
    - Base64 urlsafe encoding produces filesystem-safe characters
    - Truncation keeps pseudonym short (e.g. Pxxxxxxxxxx)

    Security note:
    - salt must be kept secret (environment variable recommended)
    """
# -----------------------------
def hmac_token(value: str, salt: str, length: int = 10) -> str:
    digest = hmac.new(
        key=salt.encode("utf-8"),
        msg=value.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    token = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")[:length]
    return token


def make_pseudonym(real_patient_id: str, prefix: str, salt: str) -> str:
    # Deterministic, stable across environments if salt is shared
    # Converts a real patient ID into a pseudonym like Pxxxxxxx
    return f"{prefix}{hmac_token(real_patient_id, salt, length=10)}"


def get_or_create_pseudonym(
    db_path: Path,
    real_patient_id: str,
    patient_name: str,
    birth_date: str,
    prefix: str,
    pseud_salt: str,
) -> str:
    existing = get_pseudo_by_real(db_path, real_patient_id)
    if existing:
        return existing

    pseudo_id = make_pseudonym(real_patient_id, prefix=prefix, salt=pseud_salt)
    insert_mapping(db_path, real_patient_id, pseudo_id, patient_name, birth_date)
    return pseudo_id # If mapping exists, reuse it. f not create a new pseudonym and store.


# -----------------------------
# UID MAPPING (DETERMINISTIC + RECURSIVE)
# -----------------------------
class UIDMapper:
    """
    Maps original UIDs to new UIDs deterministically so relationships remain consistent.
    Uses uid_root + numeric suffix derived from salted hash.
    """
    def __init__(self, uid_root: str, uid_salt: str):
        self.uid_root = uid_root.rstrip(".")
        self.uid_salt = uid_salt
        self._map: Dict[str, str] = {}

    def map_uid(self, old_uid: str) -> str:
        old_uid = (old_uid or "").strip()
        if not old_uid:
            return old_uid

        if old_uid in self._map:
            return self._map[old_uid]

        # Create numeric string from hash so it can fit into UID requirements
        h = hashlib.sha256((self.uid_salt + old_uid).encode("utf-8")).hexdigest()
        num = str(int(h, 16))  # big integer as decimal
        new_uid = f"{self.uid_root}.{num[:30]}"  # trim to keep total length < 64

        self._map[old_uid] = new_uid
        return new_uid


def replace_uids_everywhere(ds: Dataset, uid_mapper: UIDMapper) -> None:
    """
    Replace all UI VR elements everywhere, including nested sequences.
    Handles multivalue UI (VM > 1).
    """
    for elem in ds.iterall():
        if elem.VR == "UI":
            if elem.VM and elem.VM > 1:
                elem.value = [uid_mapper.map_uid(str(v)) for v in elem.value]
            else:
                elem.value = uid_mapper.map_uid(str(elem.value))

    # Keep file_meta consistent when present
    if hasattr(ds, "file_meta") and ds.file_meta:
        if "MediaStorageSOPInstanceUID" in ds.file_meta and "SOPInstanceUID" in ds:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID


# -----------------------------
# PRIVATE TAGS & OVERLAYS
# -----------------------------
def handle_private_tags(ds: Dataset, mode: str = "remove_all", allowlist_creators: Optional[List[str]] = None) -> None:
    """
    mode:
      - remove_all: remove every private tag which are hidden identifiers from vendior
      - allowlist: keep only private blocks whose creator is allowlisted for certain vendor-specific features; remove the rest
    """
    mode = (mode or "remove_all").lower()

    if mode == "remove_all":
        ds.remove_private_tags()
        return

    if mode == "allowlist":
        allow = set(allowlist_creators or [])
        for block in list(ds.private_blocks()):
            creator = block.private_creator
            if creator not in allow:
                block.remove_private_tags()
        return

    raise ValueError(f"Unknown private tag mode: {mode}")


def remove_overlays_60xx(ds: Dataset) -> None:
    """
    Remove all overlay/graphics elements in groups 60xx.
    This does NOT change PixelData, but removes overlay planes.
    """
    for group in range(0x6000, 0x6020, 2):
        tags = [elem.tag for elem in ds.iterall() if elem.tag.group == group]
        for tag in tags:
            if tag in ds:
                del ds[tag]


# -----------------------------
# TAG OPERATIONS (DELETE/BLANK)
# -----------------------------
def delete_keywords(ds: Dataset, keywords: Iterable[str]) -> None: #deletes type 3 tags (optional)
    for kw in keywords:
        if kw in ds:
            del ds[kw]


def blank_keywords(ds: Dataset, keywords: Iterable[str], value: str = "") -> None: #deletes blank tags (often type 2) by assigning empty string
    for kw in keywords:
        if kw in ds:
            ds.data_element(kw).value = value


# -----------------------------
# DATE/TIME HANDLING
# -----------------------------
def _parse_da(v: str) -> Optional[datetime]:
    # DA: YYYYMMDD
    v = (v or "").strip()
    if not v:
        return None
    try:
        return datetime.strptime(v, "%Y%m%d")
    except Exception:
        return None


def _parse_dt(v: str) -> Optional[datetime]:
    # DT can be: YYYYMMDDHHMMSS.FFFFFF&ZZXX. Prefix is parsed for decreasing time detail
    v = (v or "").strip()
    if not v:
        return None
    # take only digits up to seconds
    digits = "".join(ch for ch in v if ch.isdigit())
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H", "%Y%m%d"): # use only digits in this format
        try:
            return datetime.strptime(digits[: len(fmt.replace("%", ""))], fmt)
        except Exception:
            continue
    return None


def _format_da(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _format_dt(dt: datetime) -> str:
    # Keep it simple: YYYYMMDDHHMMSS
    return dt.strftime("%Y%m%d%H%M%S")


def patient_level_offset_days(pseudo_id: str, salt: str, max_days: int = 365) -> int:
    """
    Deterministically derive a small offset (1..max_days) from pseudo_id.
    Using salt so offsets are not identifiable.
    """
    token = hmac_token(pseudo_id, salt=salt, length=16)
    n = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
    return (n % max_days) + 1


def apply_date_policy(ds: Dataset, pseudo_id: str, policy: Dict[str, Any], date_salt: str) -> None:
    """
    policy["dates"]:
      mode: "blank" | "offset"
      include: [keywords...]
    """
    dates_cfg = policy.get("dates", {}) or {}
    mode = (dates_cfg.get("mode", "blank") or "blank").lower()
    include = dates_cfg.get("include", []) or []

    if not include:
        return

    if mode == "blank":
        blank_keywords(ds, include, value="")
        return

    if mode == "offset":
        # Apply consistent per-patient day offset across all included date/time fields
        days = patient_level_offset_days(pseudo_id, salt=date_salt, max_days=int(dates_cfg.get("max_days", 365)))
        delta = timedelta(days=days)

        for kw in include:
            if kw not in ds:
                continue

            elem = ds.data_element(kw)
            vr = elem.VR

            # DA
            if vr == "DA":
                dt = _parse_da(str(elem.value))
                if dt:
                    elem.value = _format_da(dt + delta)
                else:
                    elem.value = ""
                continue

            # DT
            if vr == "DT":
                dt = _parse_dt(str(elem.value))
                if dt:
                    elem.value = _format_dt(dt + delta)
                else:
                    elem.value = ""
                continue

            # TM: you can blank or keep; safest is blank unless you implement offset+wrap
            if vr == "TM":
                elem.value = ""
                continue

            # If VR unknown, blank it
            elem.value = ""
        return

    raise ValueError(f"Unknown date mode: {mode}")


# -----------------------------
# MAIN PSEUDONYMISATION CODE
# -----------------------------
@dataclass
class RunConfig:
    input_root: Path
    output_root: Path
    db_path: Path
    policy_path: Path


def read_dicom_safe(path: Path) -> Optional[Dataset]: #reads dicom
    try:
        return pydicom.dcmread(str(path))
    except Exception:
        return None


def is_already_processed(patient_id: str, pseud_prefix: str) -> bool: #if already processed, provide pseudonym
    pid = (patient_id or "").strip()
    if pid in {"", "ID", "ANONYMOUS"}:
        return True
    return pid.startswith(pseud_prefix)


def pseudonymise_one_file(
    in_file: Path,
    out_file: Path,
    db_path: Path,
    policy: Dict[str, Any],
    uid_mapper: Optional[UIDMapper],
    pseud_prefix: str,
    pseud_salt: str,
    date_salt: str,
) -> Optional[str]:
    ds = read_dicom_safe(in_file)
    if ds is None:
        return None

    real_id = str(ds.get("PatientID", "")).strip()
    name = str(ds.get("PatientName", "")).strip()
    dob = str(ds.get("PatientBirthDate", "")).strip()

    # HARD STOP: skip already-anonymised/pseudonymised/empty
    if is_already_processed(real_id, pseud_prefix):
        return None

    # You can choose a different stable key; here we REQUIRE PatientID for identity mapping.
    if not real_id:
        return None

    pseudo_id = get_or_create_pseudonym(
        db_path=db_path,
        real_patient_id=real_id,
        patient_name=name,
        birth_date=dob,
        prefix=pseud_prefix,
        pseud_salt=pseud_salt,
    )

    # 1) Replace direct patient identifiers (default baseline)
    ds.PatientID = pseudo_id
    ds.PatientName = pseudo_id
    if "PatientBirthDate" in ds:
        ds.PatientBirthDate = policy.get("patient_birth_date_replacement", "19000101")

    # 2) Apply policy: delete / blank (direct identifiers, quasi-identifiers, etc.)
    delete_keywords(ds, policy.get("delete", []) or [])
    blank_keywords(ds, policy.get("blank", []) or [], value="")

    # 3) Dates/times policy (includes AcquisitionDateTime if configured)
    apply_date_policy(ds, pseudo_id=pseudo_id, policy=policy, date_salt=date_salt)

    # 4) Private tags policy
    priv = policy.get("private_tags", {}) or {}
    handle_private_tags(
        ds,
        mode=priv.get("mode", "remove_all"),
        allowlist_creators=priv.get("allowlist_creators", []),
    )

    # 5) Overlay policy
    overlays = policy.get("overlays", {}) or {}
    if overlays.get("remove_60xx", True):
        remove_overlays_60xx(ds)

    # 6) Deterministic UID regeneration everywhere (including nested sequences)
    uids_cfg = policy.get("uids", {}) or {}
    if uids_cfg.get("enabled", True) and uid_mapper is not None:
        replace_uids_everywhere(ds, uid_mapper)

    # 7) Save
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_file), write_like_original=False)
    return pseudo_id


def bulk_pseudonymise(cfg: RunConfig) -> None:
    policy = load_policy(cfg.policy_path)

    # Secrets: use environment variables so you don't hardcode
    pseud_salt = os.environ.get("PSEUD_SALT", "")
    uid_salt = os.environ.get("UID_SALT", "")
    date_salt = os.environ.get("DATE_SALT", pseud_salt or "date-salt")

    if not pseud_salt:
        raise ValueError("Missing PSEUD_SALT env var. Set it to a strong secret for deterministic pseudonyms.")

    # UID mapper optional based on policy
    uids_cfg = policy.get("uids", {}) or {}
    uid_mapper = None
    if uids_cfg.get("enabled", True):
        uid_root = str(uids_cfg.get("root", "")).strip()
        if not uid_root:
            raise ValueError("uids.enabled is true but uids.root is empty in policy.")
        if not uid_salt:
            raise ValueError("Missing UID_SALT env var. Set it to a strong secret for deterministic UID remap.")
        uid_mapper = UIDMapper(uid_root=uid_root, uid_salt=uid_salt)

    pseud_prefix = str(policy.get("pseudonym_prefix", "P")).strip() or "P"

    processed = 0
    skipped_non_dicom = 0
    skipped_already = 0
    failed = 0

    for path in cfg.input_root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(cfg.input_root)
        out_path = cfg.output_root / rel

        ds_head = read_dicom_safe(path)
        if ds_head is None:
            skipped_non_dicom += 1
            continue

        pid = str(ds_head.get("PatientID", "")).strip()
        if is_already_processed(pid, pseud_prefix):
            skipped_already += 1
            continue

        try:
            pseudo = pseudonymise_one_file(
                in_file=path,
                out_file=out_path,
                db_path=cfg.db_path,
                policy=policy,
                uid_mapper=uid_mapper,
                pseud_prefix=pseud_prefix,
                pseud_salt=pseud_salt,
                date_salt=date_salt,
            )
            if pseudo:
                processed += 1
            else:
                skipped_already += 1
        except Exception as e:
            failed += 1
            print("FAIL:", path, e)

    print("\nDone")
    print("Processed:", processed)
    print("Skipped non-DICOM:", skipped_non_dicom)
    print("Skipped already processed/invalid:", skipped_already)
    print("Failed:", failed)
    print("DB:", cfg.db_path)
    print("Output:", cfg.output_root)


# -----------------------------
# RESTORE (CONTROLLED RE-IDENTIFICATION PROCESS FROM DB)
"""This is used to restore identifiers from the Sqlite database"""
# -----------------------------
def restore_one_file(pseudo_file: Path, restored_out: Path, db_path: Path, pseud_prefix: str = "P") -> None:
    ds = pydicom.dcmread(str(pseudo_file))
    pseudo_id = str(ds.get("PatientID", "")).strip()

    if not pseudo_id or not pseudo_id.startswith(pseud_prefix): #check if pseudo complete
        raise ValueError(f"File does not look pseudonymised (PatientID={pseudo_id}): {pseudo_file}")

    row = get_real_by_pseudo(db_path, pseudo_id)
    if not row:
        raise ValueError(f"No mapping found in DB for pseudo_id={pseudo_id}")

    real_id, name, dob = row
    ds.PatientID = real_id
    ds.PatientName = name
    if "PatientBirthDate" in ds:
        ds.PatientBirthDate = dob

    restored_out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(restored_out), write_like_original=False)
    print(f"Restored {pseudo_id} -> {real_id}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DICOM pseudonymisation v2 (policy-driven, deterministic UID remap, SQLite mapping).")
    p.add_argument("--input", default=str(DEFAULT_INPUT_ROOT), help="Input folder root.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT), help="Output folder root.")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite mapping DB path.")
    p.add_argument("--policy", required=True, help="Policy YAML/JSON path.")
    return p


def main():
    args = build_argparser().parse_args()

    cfg = RunConfig(
        input_root=Path(args.input),
        output_root=Path(args.output),
        db_path=Path(args.db),
        policy_path=Path(args.policy),
    )

    init_db(cfg.db_path)
    print("DB PATH:", cfg.db_path)
    print("INPUT:", cfg.input_root)
    print("OUTPUT:", cfg.output_root)
    print("POLICY:", cfg.policy_path)

    bulk_pseudonymise(cfg)


if __name__ == "__main__":
    main()




