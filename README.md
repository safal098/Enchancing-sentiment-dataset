# Enhancing Sentiment

Fine-tune and compare multilingual transformer models for three-class sentiment classification on Nepali text. The project includes a Jupyter training workflow, prepared train/validation/test spreadsheets, saved evaluation reports, and plots from previous runs.

## Models and task

The training pipeline supports these Hugging Face checkpoints:

| Short name | Model checkpoint |
| --- | --- |
| `xlmr` | `xlm-roberta-base` |
| `muril` | `google/muril-base-cased` |
| `mbert` | `bert-base-multilingual-cased` |

The target labels are `Positive`, `Neutral`, and `Negative`. The notebook maps them to integer IDs 0, 1, and 2, respectively. Each model uses its own tokenizer and can be trained as a separate run. Optional Optuna search is available through the `run_model` options.

## Repository contents

```text
.
|-- README.md
|-- Nepali_sentiment_training_MultiModel (1).ipynb  # Main notebook workflow
|-- Nepali_sentiment_training_MultiModel (1).py     # Exported notebook script
|-- Devanagari Dataset/
|   |-- train_devanagari.xlsx
|   |-- val_devanagari.xlsx
|   |-- test_devanagari.xlsx
|-- English Dataset/
|   |-- Train-English.xlsx
|   |-- val-English.xlsx
|   |-- Test-English.xlsx
|-- tokenizers/                                     # Saved tokenizer assets
`-- models/                                         # Reports, metrics, and plots
	|-- mbert_mbert23/
	|-- muril_muril23/
	`-- xlm-roberta-base_xlmr23/
```

Each existing model-result directory contains some or all of `metrics.json`, `classification_report.csv`, `training_history.csv`, `test_predictions.csv`, `error_analysis.csv`, `best_hyperparameters.csv`, and a `plots/` directory. The plots include a loss curve, confusion matrix, ROC curve, and calibration chart.

## Data format and preprocessing

The notebook reads the three Excel files from the project root. It looks for the columns `Nepali_Translation` and `Sentiment`; if those names are absent, it falls back to the fourth and third columns, respectively. Rows with missing values are dropped, sentiment text is normalized, and duplicate text is removed within and across splits. Cross-split duplicates are removed in train, validation, test priority order, and the notebook checks for leakage.

Accepted sentiment labels are `Positive`, `Neutral`, and `Negative`. The loader also corrects the known misspellings `Postive` and `Netural`. Check the workbook columns and label values if loading fails or rows are unexpectedly removed.

## Setup on Windows

Clone the repository and create an isolated environment:

```powershell
git clone https://github.com/safal098/Enhancing-Sentiment.git
Set-Location Enhancing-Sentiment
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the build appropriate for your hardware. The standard pip command is shown below; GPU users should select a build matching their CUDA setup. Then install the notebook's other dependencies:

```powershell
python -m pip install torch
python -m pip install transformers accelerate evaluate optuna pandas numpy scikit-learn matplotlib seaborn openpyxl ipykernel
```

The notebook checks for CUDA and can run on CPU, although transformer training will be substantially slower without a compatible GPU. Package versions are not pinned in this repository.

## Run training

1. Open `Nepali_sentiment_training_MultiModel (1).ipynb` in VS Code and select the `.venv` Python kernel.
2. In the configuration cell, update `TRAIN_PATH`, `VAL_PATH`, `TEST_PATH`, and `D_ROOT` to the location where you cloned the repository. The checked-in script currently contains paths for `D:\Nepali sentiment`.
3. Run the setup and data-preparation cells in order. The notebook calls Hugging Face Hub `login()`; authenticate there if prompted. Never put a Hub token in the notebook or commit it.
4. Run only the model-training cell(s) you need. For example:

   ```python
   run_model("xlmr", run_name="xlmr_baseline", use_optuna=False)
   run_model("muril", run_name="muril_baseline", use_optuna=False)
   run_model("mbert", run_name="mbert_baseline", use_optuna=False)
   ```

   The notebook contains multiple experiment cells. Training all of them can take substantial time and disk space. Set `use_optuna=True` and choose `n_trials` when you want hyperparameter search.

`run_model` defaults to `clean_old=True`, which removes that run's existing model, checkpoint, and tokenizer output directories before training. Use a new `run_name` to preserve earlier runs, or set `clean_old=False` when appropriate.

The `.py` file is an exported notebook script and still contains IPython-specific setup calls and machine-specific paths. The notebook is the intended entry point; the script is not a portable standalone command without adapting those sections.

## Outputs

For a run named `xlmr_baseline`, the pipeline writes results under directories similar to:

```text
models/xlm-roberta-base_xlmr_baseline/
├── metrics.json
├── classification_report.csv
├── training_history.csv
├── test_predictions.csv
├── error_analysis.csv
├── best_hyperparameters.csv
├── model/                         # Trained model weights
└── plots/
	├── calibration_test.png
	├── confusion_matrix.png
	├── loss_curve.png
	└── roc_test.png

tokenizers/xlm-roberta-base_xlmr_baseline/
training_checkpoints/xlmr_baseline/
```

The metrics include accuracy, balanced accuracy, macro/weighted F1, Matthews correlation, ROC-AUC where available, calibration error, and bootstrap confidence intervals. `error_analysis.csv` contains misclassified test examples, so treat it as dataset content when sharing results.

## Git and large artifacts

The repository tracks the notebooks, code, input spreadsheets, tokenizer files, and lightweight reports/plots. The `.gitignore` excludes the local Hugging Face cache (including credentials), training checkpoints, and trained model-weight directories. Those files can be several gigabytes; they are not present in the GitHub repository. Confirm you have permission to redistribute the included datasets and generated examples before making the repository public.
