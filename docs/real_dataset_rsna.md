# RSNA real dataset workflow

This project keeps real medical data outside Git. The synthetic dataset remains the default smoke-test
dataset. RSNA data is prepared into a separate CSV manifest and PNG image folder under
`data_external/`, which is ignored by Git.

## Manual prerequisites

1. Create or log into a Kaggle account.
2. Open the RSNA Pneumonia Detection Challenge page and accept the competition/data rules:
   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
3. Create Kaggle API credentials from:
   https://www.kaggle.com/settings/api

Do not commit Kaggle tokens, downloaded archives, DICOM files, converted images, or SQLite evidence
databases.

This repository uses a local Kaggle config directory ignored by Git:

```powershell
New-Item -ItemType Directory -Force data_external\.kaggle
$env:KAGGLE_CONFIG_DIR=(Resolve-Path data_external\.kaggle).Path
```

## Download

From the repository root:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-data.txt
.\.venv\Scripts\kaggle competitions files rsna-pneumonia-detection-challenge
```

If the `kaggle` command says authentication is missing, use one of the Kaggle-supported methods:

```powershell
$env:KAGGLE_CONFIG_DIR=(Resolve-Path data_external\.kaggle).Path
.\.venv\Scripts\kaggle auth login
```

or set a token for the current PowerShell session:

```powershell
$env:KAGGLE_API_TOKEN="paste_your_kaggle_token_here"
```

## Prepare a small project-compatible sample

```powershell
$env:KAGGLE_CONFIG_DIR=(Resolve-Path data_external\.kaggle).Path
.\.venv\Scripts\python scripts\prepare_rsna_dataset.py --cases-per-class 10 --download-missing
```

This creates:

- `data_external/rsna-pneumonia/processed_images/*.png`
- `data_external/rsna-pneumonia/real_cases_rsna_sample.csv`

Label mapping:

| RSNA class | Project label |
|---|---|
| `Normal` | `normal` |
| `Lung Opacity` | `suspected_opacity` |
| `No Lung Opacity / Not Normal` | `uncertain` |

The last mapping is deliberately conservative: the image is abnormal/not normal, but not necessarily
the opacity class targeted by this prototype.

## Evaluate without touching the synthetic smoke test

```powershell
.\.venv\Scripts\python eval\run_evaluation.py --mode toy `
  --cases-csv data_external\rsna-pneumonia\real_cases_rsna_sample.csv `
  --out-dir eval\outputs_rsna `
  --db-path medical_ai_evidence_rsna.sqlite
```

For a quick run:

```powershell
.\.venv\Scripts\python eval\run_evaluation.py --mode baseline `
  --cases-csv data_external\rsna-pneumonia\real_cases_rsna_sample.csv `
  --limit 3 `
  --out-dir eval\outputs_rsna_smoke `
  --db-path medical_ai_evidence_rsna_smoke.sqlite
```

Outputs to inspect:

- `eval/outputs_rsna/baseline_predictions.csv`
- `eval/outputs_rsna/improved_predictions.csv`
- `eval/outputs_rsna/before_after_summary.csv`
- `medical_ai_evidence_rsna.sqlite`

## Model access note

The repository currently calls `google/medgemma-4b-it` through Transformers. To obtain non-fallback
predictions, you must accept the Hugging Face model terms and authenticate locally:

```powershell
.\.venv\Scripts\huggingface-cli login
$env:HF_TOKEN="hf_your_token_here"
```

On Windows with an NVIDIA GPU, install the CUDA PyTorch wheel explicitly. Installing plain `torch`
from PyPI may install a CPU-only build:

```powershell
.\.venv\Scripts\python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If Hugging Face returns `403 gated repo`, the token is valid but the account has not been granted
access to the exact MedGemma repository. Accept/request access on both pages if needed:

- https://huggingface.co/google/medgemma-4b-it
- https://huggingface.co/google/medgemma-4b-pt

Without model access or enough compute, the code falls back to `uncertain`. That still validates the
guardrails, logs, JSON schema and warning contract, but it is not evidence of model performance.
