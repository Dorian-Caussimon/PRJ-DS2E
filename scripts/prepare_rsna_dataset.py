from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

RSNA_CLASS_TO_PROJECT_LABEL = {
    "Normal": "normal",
    "Lung Opacity": "suspected_opacity",
    "No Lung Opacity / Not Normal": "uncertain",
}


def _kaggle_executable() -> str:
    local_kaggle = Path(sys.executable).with_name("kaggle.exe")
    if local_kaggle.exists():
        return str(local_kaggle)
    kaggle = shutil.which("kaggle")
    if kaggle:
        return kaggle
    raise FileNotFoundError("Kaggle CLI not found. Install it with: python -m pip install -r requirements-data.txt")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_rsna_classes(class_info_csv: Path) -> dict[str, str]:
    with class_info_csv.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        mapping = {}
        for row in rows:
            patient_id = row["patientId"]
            rsna_class = row["class"]
            project_label = RSNA_CLASS_TO_PROJECT_LABEL.get(rsna_class)
            if project_label is None:
                continue
            mapping[patient_id] = project_label
        return mapping


def _find_downloaded_file(rsna_dir: Path, competition_file: str) -> Path | None:
    relative_path = Path(competition_file)
    candidates = [
        rsna_dir / relative_path,
        rsna_dir / relative_path.name,
        rsna_dir / f"{relative_path.name}.zip",
    ]
    return next((path for path in candidates if path.exists()), None)


def _extract_if_zip(path: Path, target: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        target_name = target.as_posix()
        target_basename = target.name
        member = next(
            (
                name
                for name in names
                if name.replace("\\", "/") == target_name or Path(name).name == target_basename
            ),
            None,
        )
        if member is None and len(names) == 1:
            member = names[0]
        if member is None:
            raise FileNotFoundError(f"Zip archive does not contain expected file {target.name}: {path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def _download_competition_file(
    rsna_dir: Path,
    competition_file: str,
    competition: str,
    force_download: bool,
) -> Path:
    target = rsna_dir / Path(competition_file)
    existing = _find_downloaded_file(rsna_dir, competition_file)
    if existing is not None and not force_download:
        extracted = _extract_if_zip(existing, target)
        if extracted != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted.replace(target)
            extracted = target
        return extracted

    rsna_dir.mkdir(parents=True, exist_ok=True)
    command = [
        _kaggle_executable(),
        "competitions",
        "download",
        competition,
        "-f",
        competition_file,
        "-p",
        str(rsna_dir),
    ]
    if force_download:
        command.append("-o")

    subprocess.run(command, check=True, env=os.environ.copy())

    downloaded = _find_downloaded_file(rsna_dir, competition_file)
    if downloaded is None:
        raise FileNotFoundError(f"Kaggle download succeeded but file was not found: {competition_file}")

    downloaded = _extract_if_zip(downloaded, target)
    if downloaded != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(target)
        downloaded = target
    return downloaded


def _sample_balanced(
    patient_to_label: dict[str, str],
    cases_per_class: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for label in ("normal", "suspected_opacity", "uncertain"):
        patient_ids = [patient_id for patient_id, value in patient_to_label.items() if value == label]
        patient_ids.sort()
        rng.shuffle(patient_ids)
        selected.extend((patient_id, label) for patient_id in patient_ids[:cases_per_class])
    return selected


def _dicom_to_uint8(dicom_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(dicom_path)
    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    low, high = np.percentile(pixels, (0.5, 99.5))
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())
    if high <= low:
        return np.zeros_like(pixels, dtype=np.uint8)

    pixels = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    return (pixels * 255).astype(np.uint8)


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "image_path", "source", "label", "split", "quality", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_rsna_sample(
    rsna_dir: Path,
    image_out_dir: Path,
    manifest_path: Path,
    cases_per_class: int,
    seed: int,
    split: str,
    download_missing: bool,
    competition: str,
    force_download: bool,
) -> dict[str, object]:
    class_info_csv = rsna_dir / "stage_2_detailed_class_info.csv"
    dicom_dir = rsna_dir / "stage_2_train_images"

    if not class_info_csv.exists():
        if download_missing:
            _download_competition_file(
                rsna_dir=rsna_dir,
                competition_file="stage_2_detailed_class_info.csv",
                competition=competition,
                force_download=force_download,
            )
        else:
            raise FileNotFoundError(f"Missing RSNA class info CSV: {class_info_csv}")

    patient_to_label = _read_rsna_classes(class_info_csv)
    selected = _sample_balanced(patient_to_label, cases_per_class=cases_per_class, seed=seed)

    image_out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    skipped: list[str] = []

    for index, (patient_id, label) in enumerate(selected, start=1):
        dicom_path = dicom_dir / f"{patient_id}.dcm"
        if not dicom_path.exists():
            if download_missing:
                try:
                    dicom_path = _download_competition_file(
                        rsna_dir=rsna_dir,
                        competition_file=f"stage_2_train_images/{patient_id}.dcm",
                        competition=competition,
                        force_download=force_download,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError):
                    skipped.append(patient_id)
                    continue
            else:
                skipped.append(patient_id)
                continue

        image = Image.fromarray(_dicom_to_uint8(dicom_path)).convert("RGB")
        png_path = image_out_dir / f"RSNA_{index:04d}_{label}_{patient_id}.png"
        image.save(png_path)

        rows.append(
            {
                "case_id": f"RSNA_{index:04d}_{patient_id}",
                "image_path": _repo_relative(png_path),
                "source": "rsna_pneumonia_detection_challenge",
                "label": label,
                "split": split,
                "quality": "good",
                "notes": f"mapped from RSNA class to {label}; original_patient_id={patient_id}",
            }
        )

    _write_csv(manifest_path, rows)
    counts = {label: sum(row["label"] == label for row in rows) for label in RSNA_CLASS_TO_PROJECT_LABEL.values()}
    return {
        "manifest_path": _repo_relative(manifest_path),
        "image_out_dir": _repo_relative(image_out_dir),
        "rows": len(rows),
        "counts": counts,
        "skipped_missing_dicoms": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small RSNA real-data sample for this project.")
    parser.add_argument("--rsna-dir", type=Path, default=ROOT / "data_external" / "rsna-pneumonia")
    parser.add_argument(
        "--image-out-dir",
        type=Path,
        default=ROOT / "data_external" / "rsna-pneumonia" / "processed_images",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data_external" / "rsna-pneumonia" / "real_cases_rsna_sample.csv",
    )
    parser.add_argument("--cases-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="real_sample")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--competition", default="rsna-pneumonia-detection-challenge")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    summary = prepare_rsna_sample(
        rsna_dir=args.rsna_dir,
        image_out_dir=args.image_out_dir,
        manifest_path=args.manifest_path,
        cases_per_class=args.cases_per_class,
        seed=args.seed,
        split=args.split,
        download_missing=args.download_missing,
        competition=args.competition,
        force_download=args.force_download,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
