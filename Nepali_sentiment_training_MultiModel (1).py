#!/usr/bin/env python
# coding: utf-8

# In[33]:


import gc
import torch
import shutil
import os
import glob
from pathlib import Path

# 1. Clear GPU Memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU Memory Cleared. Currently allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 2. Run Garbage Collection
gc.collect()

# 3. Redirect HF cache to a local folder instead of Google Drive
HF_CACHE_DIR = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

# 4. Clean up local disk (Temporary results directories)
for results_dir in glob.glob('./results_*'):
    if os.path.exists(results_dir):
        print(f"Removing temporary directory: {results_dir}")
        shutil.rmtree(results_dir)

if os.path.exists('./runs'):
    shutil.rmtree('./runs')

print(f"Memory and local disk cleanup complete. HF Cache redirected to: {os.environ['HF_HOME']}")


# # Devnagiri Text — Sentiment Classification (XLM-R / MuRIL / mBERT)
# 
# This notebook fine-tunes and compares **three** multilingual transformer models — `xlm-roberta-base`, `google/muril-base-cased`, and `bert-base-multilingual-cased` — on the `English_Translation` column for **3-class sentiment classification** (Positive / Neutral / Negative).
# 
# 
# 
# 
# ### One pipeline, three independent runs
# All three models are trained through **one shared function**, `run_sentiment_pipeline(model_key, model_name)`. Each call is fully self-contained (own tokenizer, own model, own trainer, own save directory) — so you run whichever model you need, in any order, without the others being trained first:
# ```python
# results['XLM-R'] = run_sentiment_pipeline('xlmr',  'xlm-roberta-base')
# results['MuRIL'] = run_sentiment_pipeline('muril', 'google/muril-base-cased')
# results['mBERT'] = run_sentiment_pipeline('mbert', 'bert-base-multilingual-cased')
# ```
# ```
# Dataset  : Combined_Common_File.xlsx
# Input col: input_text (English_Translation column, mapped from source file)
# Label col: sentiment  (Positive / Neutral / Negative)
# ```

# ## 0. Mount Drive & GPU Check

# In[34]:


# Local-device notebook: Colab Drive mount is not required.
print('Local notebook environment active; Google Drive mount skipped.')


# In[35]:


import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU detected — training will be very slow!")


# ## 1. Imports

# In[36]:


get_ipython().system('pip install -q evaluate')


# In[37]:


# Install missing libraries
get_ipython().system('pip install -q evaluate')

import os
import gc
import shutil
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import evaluate
import warnings
warnings.filterwarnings('ignore')

print("All imports OK")


# In[38]:


get_ipython().system('pip install -q optuna')


# ## 2. Configuration

# In[39]:


from huggingface_hub import login

login()


# In[40]:


# ── Paths ─────────────────────────────────────────────────────────────────────

# Local data source
TRAIN_PATH = r"D:\Nepali sentiment\train_devanagari.xlsx"
VAL_PATH = r"D:\Nepali sentiment\val_devanagari.xlsx"
TEST_PATH = r"D:\Nepali sentiment\test_devanagari.xlsx"

# ============================================================
# D: DRIVE STORAGE
# ============================================================

D_ROOT = r'D:\Nepali sentiment'

# Main directories
MODEL_ROOT = os.path.join(D_ROOT, 'models')
TOKENIZER_ROOT = os.path.join(D_ROOT, 'tokenizers')
RESULTS_ROOT = os.path.join(D_ROOT, 'results')
TRAINING_ROOT = os.path.join(D_ROOT, 'training_checkpoints')

# Hugging Face cache
HF_ROOT = os.path.join(D_ROOT, 'hf_cache')

os.environ['HF_HOME'] = HF_ROOT
os.environ['HF_HUB_CACHE'] = os.path.join(HF_ROOT, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_ROOT, 'transformers')

# Try a Hub login only if a valid token is available locally.
# If the environment is offline or the token cannot be validated, we still continue
# # with the local cache and already downloaded model files.
# try:
#     from huggingface_hub import login
#     hf_token = os.environ.get('HF_TOKEN')
#     if hf_token:
#         login(token=hf_token, add_to_git_credential=False)
#         print('Hugging Face login successful using HF_TOKEN.')
#     else:
#         print('HF_TOKEN not found; continuing in offline/cache mode.')
# except Exception as exc:
#     print(f'Hugging Face login skipped: {exc}')
#     print('Continuing with local cache / already-downloaded models.')

# Create all directories
for folder in [
    D_ROOT,
    MODEL_ROOT,
    TOKENIZER_ROOT,
    RESULTS_ROOT,
    TRAINING_ROOT,
    HF_ROOT,
    os.environ['HF_HUB_CACHE'],
    os.environ['TRANSFORMERS_CACHE'],
]:
    os.makedirs(folder, exist_ok=True)

print("=" * 70)
print("D: DRIVE STORAGE CONFIGURED")
print("=" * 70)
print(f"Root directory       : {D_ROOT}")
print(f"Models               : {MODEL_ROOT}")
print(f"Tokenizers           : {TOKENIZER_ROOT}")
print(f"Results              : {RESULTS_ROOT}")
print(f"Checkpoints          : {TRAINING_ROOT}")
print(f"HuggingFace cache    : {HF_ROOT}")
print("=" * 70)


# ── Label mapping ─────────────────────────────────────────────────────────────

# Paper (Thapa et al. 2023) uses Positive / Neutral / Negative

SENT_MAP = {
    'Positive': 0,
    'Neutral': 1,
    'Negative': 2,
}

ID2LABEL = {v: k for k, v in SENT_MAP.items()}
N_LABELS = len(SENT_MAP)


# ── Shared training hyper-parameters ──────────────────────────────────────────

MAX_LEN             = 256
SEED                = 42
EPOCHS              = 8
BATCH_TRAIN         = 16
BATCH_EVAL          = 32
LR                  = 1e-5
WEIGHT_DECAY        = 0.15
WARMUP_RATIO        = 0.1
GRAD_ACCUM          = 2
EARLY_STOP_PATIENCE = 2
SAVE_TOTAL_LIMIT    = 1


# ── The 3 models being compared ──────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        'key': 'xlmr',
        'label': 'XLM-RoBERTa',
        'checkpoint': 'xlm-roberta-base',

        # Final trained model → D:
        'save_dir': os.path.join(
            MODEL_ROOT,
            'xlm-roberta-base_model'
        ),

        # Tokenizer → D:
        'tok_dir': os.path.join(
            TOKENIZER_ROOT,
            'xlm-roberta-base_tokenizer'
        ),

        # Trainer checkpoints/results → D:
        'output_dir': os.path.join(
            TRAINING_ROOT,
            'xlmr'
        ),
    },

    {
        'key': 'mbert',
        'label': 'mBERT',
        'checkpoint': 'bert-base-multilingual-cased',

        'save_dir': os.path.join(
            MODEL_ROOT,
            'mbert_sentiment_model'
        ),

        'tok_dir': os.path.join(
            TOKENIZER_ROOT,
            'mbert_sentiment_tokenizer'
        ),

        'output_dir': os.path.join(
            TRAINING_ROOT,
            'mbert'
        ),
    },

    {
        'key': 'muril',
        'label': 'MuRIL',
        'checkpoint': 'google/muril-base-cased',

        'save_dir': os.path.join(
            MODEL_ROOT,
            'muril_sentiment_model'
        ),

        'tok_dir': os.path.join(
            TOKENIZER_ROOT,
            'muril_sentiment_tokenizer'
        ),

        'output_dir': os.path.join(
            TRAINING_ROOT,
            'muril'
        ),
    },
]


print("\nModels configured:")
for c in MODEL_CONFIGS:
    print(
        f"  - {c['label']:<15} → {c['checkpoint']}"
    )
    print(
        f"      Model     : {c['save_dir']}"
    )
    print(
        f"      Tokenizer : {c['tok_dir']}"
    )
    print(
        f"      Training  : {c['output_dir']}"
    )


# ```markdown
# ## 2.1. Data Consolidation
# Combining multiple Excel sources into a single dataset for training and validation.
# ```

# ## 3. Data Loading & Preprocessing

# In[41]:


import pandas as pd
import os


def prepare_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file at {path}.")

    df = pd.read_excel(path) if path.endswith(('.xlsx', '.xls')) else pd.read_csv(path)

    target_text_col = 'Nepali_Translation'
    target_label_col = 'Sentiment'

    if target_text_col in df.columns and target_label_col in df.columns:
        df = df[[target_text_col, target_label_col]].copy()
    else:
        df = df.iloc[:, [3, 2]].copy()

    df.columns = ['input_text', 'sentiment_raw']
    df = df.dropna().reset_index(drop=True)

    df['sentiment_raw'] = (
        df['sentiment_raw']
        .astype(str)
        .str.strip()
        .str.capitalize()
        .replace({'Postive': 'Positive', 'Netural': 'Neutral'})
    )

    df['label'] = df['sentiment_raw'].map(SENT_MAP)
    df = df[df['label'].notna()].reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    # Remove duplicate texts within each split before comparing splits.
    return df.drop_duplicates(subset=['input_text'], keep='first').reset_index(drop=True)


# ── Load datasets ──────────────────────────────────────────────
print("Loading specific datasets...")
train_df = prepare_dataset(TRAIN_PATH)
val_df   = prepare_dataset(VAL_PATH)
test_df  = prepare_dataset(TEST_PATH)

print("\nBefore cross-split dedup:")
print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Cross-split deduplication ──────────────────────────────────
# Priority: train > val > test.
train_texts = set(train_df['input_text'].astype(str))
val_df = val_df[
    ~val_df['input_text'].astype(str).isin(train_texts)
].reset_index(drop=True)

train_val_texts = train_texts | set(val_df['input_text'].astype(str))
test_df = test_df[
    ~test_df['input_text'].astype(str).isin(train_val_texts)
].reset_index(drop=True)

print("\nAfter cross-split dedup:")
print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Verify zero leakage ────────────────────────────────────────
t_set = set(train_df['input_text'].astype(str))
v_set = set(val_df['input_text'].astype(str))
e_set = set(test_df['input_text'].astype(str))
leakage = {
    'train_val': len(t_set & v_set),
    'train_test': len(t_set & e_set),
    'val_test': len(v_set & e_set),
}
print(f"  Leakage check: {leakage}")
assert all(value == 0 for value in leakage.values()), "Leakage still present!"
print("  ✅ Zero leakage confirmed.\n")

print("Successfully loaded datasets with zero cross-split leakage.")


# ## 4. Train / Validation / Test Split
# 
# Stratified 80/10/10 split on the label column — computed once and shared by every model so the comparison across XLM-R / MuRIL / mBERT is apples-to-apples.

# In[42]:


# The datasets were already loaded separately above by prepare_dataset().
# Do not split an undefined combined dataframe here.
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Safety check: if the same raw text appears across splits, report it.
train_texts = set(train_df['input_text'].astype(str))
val_texts = set(val_df['input_text'].astype(str))
test_texts = set(test_df['input_text'].astype(str))
leakage = {
    'train_val': len(train_texts & val_texts),
    'train_test': len(train_texts & test_texts),
    'val_test': len(val_texts & test_texts),
}

print(f"Train : {len(train_df)} rows")
print(f"Val   : {len(val_df)} rows")
print(f"Test  : {len(test_df)} rows")
print(f"Cross-split overlap after deduplication: {leakage}")
print("\nTrain distribution:")
print(train_df['sentiment_raw'].value_counts())
print("\nVal distribution:")
print(val_df['sentiment_raw'].value_counts())
print("\nTest distribution:")
print(test_df['sentiment_raw'].value_counts())


# ## 5. Dataset Class
# 
# Tokenization is model-specific (each model has its own tokenizer/vocab), so it happens inside `run_sentiment_pipeline` below rather than as a separate global step — but the `Dataset` wrapper itself is identical for every model.

# In[43]:


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)

print("SentimentDataset defined.")


# ## 6. Class Weights
# 
# Computed once from `train_df` and reused by every model — the label distribution doesn't change between models, only the tokenizer/architecture does.

# In[44]:


cw = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_df['label'].tolist(),
)
class_weights = torch.tensor(cw, dtype=torch.float)
print(f"Class weights (Pos / Neu / Neg): {class_weights.numpy().round(4)}")


# ## 7. Custom Trainer with Class-Weighted Loss
# 
# We override `compute_loss` to pass `class_weights` to `CrossEntropyLoss`, giving heavier penalty to misclassified minority examples. Same trainer class used for all three models.

# In[45]:


import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Get the probability of the true class
        pt = torch.exp(-ce_loss)
        # Apply the focal loss modulating factor
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_weights = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_weights * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class EpochLossPrinter(TrainerCallback):
    """Prints training and validation loss at the end of each epoch."""

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)

        # Get training loss from the current epoch
        train_loss = None
        val_loss = None
        val_f1 = None

        for log in state.log_history:
            if "loss" in log and "eval_loss" not in log:
                if abs(log.get("epoch", 0) - epoch) < 0.01:
                    train_loss = log["loss"]
            if "eval_loss" in log:
                if abs(log.get("epoch", 0) - epoch) < 0.01:
                    val_loss = log["eval_loss"]
                    val_f1 = log.get("eval_macro_f1", None)

        print(f"\n  📊 Epoch {epoch}/{args.num_train_epochs}")
        if train_loss is not None:
            print(f"     Train Loss : {train_loss:.4f}")
        if val_loss is not None:
            print(f"     Val Loss   : {val_loss:.4f}")
        if val_f1 is not None:
            print(f"     Val F1     : {val_f1:.4f}")

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute focal loss instead of cross_entropy
        loss = self.criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss


# In[46]:


class CustomSentimentHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        # Multi-Sample Dropout: Average logits from 5 different dropout masks
        logits_list = [self.classifier(dropout(pooled_output)) for dropout in self.dropouts]
        return torch.stack(logits_list, dim=0).mean(dim=0)

class CustomSentimentModel(nn.Module):
    def __init__(self, checkpoint, num_labels, id2label, label2id):
        super().__init__()
        from transformers import AutoModel
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        # Load backbone
        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.backbone.config.hidden_size

        # Custom Head
        self.classifier = CustomSentimentHead(hidden_size, num_labels)

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing to the backbone transformer."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)

        # Use pooled output (usually [CLS] token representation)
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]

        logits = self.classifier(pooled_output)

        loss = None
        # Note: Loss calculation is handled by the WeightedTrainer using FocalLoss

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# ## 8. Metrics
# 
# We track **accuracy** and **macro-F1** during training, and produce a full `classification_report` at the end — same metrics reported in Thapa et al. (2023).

# In[49]:


from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )),
        "micro_f1": float(f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )),
    }


# In[50]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def plot_multiclass_roc(all_labels, all_probs, label_names, title='ROC Curve'):
    """
    Plots a multi-class ROC curve using the One-vs-Rest strategy.
    """
    n_classes = len(label_names)
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(all_labels, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 6))
    colors = ['#4C72B0', '#55A868', '#C44E52']

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'ROC {label_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()


# ## 9. Pipeline Helpers
# 
# Two small utilities the training function relies on:
# - `get_precision_kwargs()` checks what the *current* GPU actually supports instead of hardcoding `bf16=True` — a T4 (common on free-tier Colab) doesn't support bf16 and would raise an error on it, while an A100/L4 does.
# - `free_memory()` clears GPU cache and runs garbage collection between models, so the second and third model don't fight the first one for GPU memory.
# 
# 

# In[51]:


def get_precision_kwargs() -> dict:
    """
    Picks the fastest safe mixed-precision setting for the current hardware.
    Tesla T4 (free Colab) does NOT support bf16.
    """
    if not torch.cuda.is_available():
        return {'fp16': False, 'bf16': False}

    # Explicit check for compute capability: bf16 requires >= 8.0 (Ampere+)
    # T4 is 7.5 (Turing), so it must use fp16.
    if torch.cuda.get_device_capability()[0] >= 8:
        try:
            if torch.cuda.is_bf16_supported():
                return {'fp16': False, 'bf16': True}
        except Exception:
            pass

    return {'fp16': True, 'bf16': False}

print("Precision setting for this runtime:", get_precision_kwargs())


# In[52]:


def get_optimizer_params(model, learning_rate, weight_decay, layer_decay=0.9):
    """
    Groups model parameters with layer-wise learning rate decay.
    Lower layers get a smaller learning rate.
    """
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = []

    # 1. Classification head (Maximum LR)
    head_params = [p for n, p in model.named_parameters() if 'classifier' in n or 'pooler' in n]
    if head_params:
        optimizer_grouped_parameters.append({
            'params': head_params,
            'lr': learning_rate,
            'weight_decay': weight_decay
        })

    # 2. Transformer Layers (Decaying LR)
    # Most architectures (BERT, RoBERTa, MuRIL) use 'layer.i' in their naming convention
    for layer_idx in range(11, -1, -1):
        layer_lr = learning_rate * (layer_decay ** (11 - layer_idx))

        optimizer_grouped_parameters.extend([
            {
                'params': [p for n, p in model.named_parameters() if f'layer.{layer_idx}.' in n and not any(nd in n for nd in no_decay)],
                'lr': layer_lr,
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if f'layer.{layer_idx}.' in n and any(nd in n for nd in no_decay)],
                'lr': layer_lr,
                'weight_decay': 0.0
            }
        ])

    # 3. Embeddings (Lowest LR)
    embed_params = [p for n, p in model.named_parameters() if 'embeddings' in n]
    if embed_params:
        optimizer_grouped_parameters.append({
            'params': embed_params,
            'lr': learning_rate * (layer_decay ** 12),
            'weight_decay': weight_decay
        })

    return optimizer_grouped_parameters


# ## 10. Unified Training Pipeline
# 
# This is the piece that was missing for mBERT and MuRIL: **one function**, used for every model, that tokenizes, builds datasets, initializes the model, trains with `WeightedTrainer`, evaluates, plots a confusion matrix, saves to Drive, and cleans up — always in that order, so evaluation and saving always happen *before* anything gets deleted.

# In[53]:


# """
# RESEARCH-GRADE TWO-PHASE SENTIMENT PIPELINE (v6.0)
# ===================================================
# Merges everything from v5.2, plus hardening fixes:

#   [FIX]  EpochLossPrinter is now DEFINED (v5.2 used it but never defined it)
#   [FIX]  compute_class_weight import added (v5.2 forgot it in run_model)
#   [FIX]  metric_for_best_model="macro_f1" + greater_is_better=True
#          (v5.2 silently tracked eval_loss, breaking early-stopping intent)
#   [FIX]  layer_decay now passed to the PHASE-2 trainer (v5.2 dropped it)
#   [FIX]  "Final training completed" no longer printed BEFORE training starts
#   [FIX]  duplicate config assignment lines in run_model removed
#   [FIX]  show_comparison prints best run BEFORE returning
#   [FIX]  dataloader_num_workers defaults to 0 (deadlock-proof, configurable)
#   [FIX]  best_checkpoint cast to str for JSON serialization
#   [FIX]  matplotlib forced to "Agg" backend (headless-safe, no plot hangs)

#   [NEW]  Data QA modes: qa_mode = "strict" | "warn" | "auto_fix"
#          auto_fix removes duplicate texts + cross-split leakage automatically
#   [NEW]  Disk-space pre-flight check (prevents corrupted checkpoints)
#   [NEW]  save_only_model=True default -> much smaller checkpoints
#   [NEW]  Sampled token-length analysis (fast; configurable sample size)
#   [NEW]  Step-level logging (logging_steps) -> training never looks "frozen"
#   [NEW]  faulthandler watchdog -> prints stack trace if training stalls
#   [NEW]  debug_mode -> tiny subsets for 2-minute smoke tests
#   [NEW]  run_all_models() convenience runner

# Assumes the following exist earlier in your notebook/module:
#     Constants:
#         MAX_LEN, SEED, EPOCHS, EARLY_STOP_PATIENCE,
#         GRAD_ACCUM, BATCH_TRAIN, BATCH_EVAL,
#         LR, WEIGHT_DECAY, N_LABELS, ID2LABEL, SENT_MAP
#     Classes:
#         SentimentDataset, CustomSentimentModel, WeightedTrainer
#     Functions:
#         get_optimizer_params(model, lr, weight_decay, layer_decay=...),
#         get_precision_kwargs(), compute_metrics(eval_pred)
# """

# import os
# import gc
# import sys
# import json
# import shutil
# import random
# import faulthandler
# from datetime import datetime

# import numpy as np
# import pandas as pd
# import torch
# import optuna
# import transformers
# import sklearn

# import matplotlib
# matplotlib.use("Agg")   # v6: headless-safe backend (prevents plot hangs)
# import matplotlib.pyplot as plt
# import seaborn as sns

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     EarlyStoppingCallback,
#     DataCollatorWithPadding,
#     TrainerCallback,                      # v6: needed by EpochLossPrinter
# )

# from sklearn.metrics import (
#     roc_curve, auc, confusion_matrix, classification_report,
#     f1_score, accuracy_score, matthews_corrcoef, average_precision_score,
#     roc_auc_score, balanced_accuracy_score
# )
# from sklearn.preprocessing import label_binarize
# from sklearn.utils.class_weight import compute_class_weight   # v6 FIX: was missing


# # ================================================================
# # 0. REPRODUCIBILITY, ENVIRONMENT & UTILITIES
# # ================================================================

# def _log(msg=""):
#     print(msg, flush=True)                # v6: flush -> no "silent hang" confusion

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def get_environment_info():
#     return {
#         "python": sys.version.split()[0],
#         "torch": torch.__version__,
#         "transformers": transformers.__version__,
#         "numpy": np.__version__,
#         "pandas": pd.__version__,
#         "sklearn": sklearn.__version__,
#         "cuda_available": torch.cuda.is_available(),
#         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
#         "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else "N/A",
#         "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
#         "precision_mode": "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
#     }

# def _clear_memory():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()

# def check_disk_space(path, min_free_gb=3.0):
#     """v6 NEW: fail fast instead of corrupting checkpoints mid-run."""
#     free_gb = shutil.disk_usage(path).free / 1e9
#     if free_gb < min_free_gb:
#         raise RuntimeError(
#             f"❌ Only {free_gb:.1f} GB free at '{path}' (need >= {min_free_gb} GB). "
#             f"Free disk space before training."
#         )
#     _log(f"  💾 Disk space OK: {free_gb:.1f} GB free at {path}")
#     return free_gb

# def _watchdog_arm(minutes):
#     """v6 NEW: if training stalls, dump the Python stack so you can see WHERE."""
#     if minutes and minutes > 0:
#         faulthandler.dump_traceback_later(minutes * 60, repeat=True)
#         _log(f"  ⏱️  Watchdog armed: stack dump every {minutes} min if stalled.")

# def _watchdog_disarm():
#     faulthandler.cancel_dump_traceback_later()


# # ================================================================
# # 1. DATA QA (strict / warn / auto_fix)   — v6 UPGRADED
# # ================================================================

# def validate_data(train_df, val_df, test_df, text_col="input_text"):
#     """Report-only validation. Never raises. Returns a report dict."""
#     report = {}
#     report["missing"] = {s: int(df[text_col].isna().sum())
#                          for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df])}

#     report["duplicates"] = {}
#     report["duplicate_label_conflicts"] = {}
#     for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
#         dup_mask = df.duplicated(subset=[text_col], keep=False)
#         dup_df = df[dup_mask]
#         report["duplicates"][s] = int(len(dup_df))
#         if dup_df.empty:
#             report["duplicate_label_conflicts"][s] = 0
#         else:
#             report["duplicate_label_conflicts"][s] = int((dup_df.groupby(text_col)["label"].nunique() > 1).sum())

#     report["label_conflicts"] = {}
#     for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
#         report["label_conflicts"][s] = int((df.groupby(text_col)["label"].nunique() > 1).sum())

#     sets = {s: set(df[text_col].astype(str))
#             for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df])}
#     report["leakage"] = {
#         "train_val": len(sets["train"] & sets["val"]),
#         "train_test": len(sets["train"] & sets["test"]),
#         "val_test": len(sets["val"] & sets["test"]),
#     }
#     report["class_distribution"] = {s: df["label"].value_counts().to_dict()
#                                     for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df])}
#     return report

# def _qa_violations(report):
#     """Human-readable list of blocking violations."""
#     v = []
#     if any(x > 0 for x in report["missing"].values()):
#         v.append(f"Missing values: {report['missing']}")
#     if any(x > 0 for x in report["duplicate_label_conflicts"].values()):
#         v.append(f"Duplicate texts with CONFLICTING labels: {report['duplicate_label_conflicts']}")
#     if any(x > 0 for x in report["label_conflicts"].values()):
#         v.append(f"Label conflicts: {report['label_conflicts']}")
#     if any(x > 0 for x in report["leakage"].values()):
#         v.append(f"Cross-split leakage: {report['leakage']}")
#     return v

# def auto_fix_data(train_df, val_df, test_df, text_col="input_text"):
#     """v6 NEW: automatically clean duplicates + leakage. Returns fixed dfs + fix log."""
#     fixes = {}

#     # 1. Drop missing text
#     for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
#         n_miss = int(df[text_col].isna().sum())
#         if n_miss:
#             fixes[f"dropped_missing_{name}"] = n_miss
#     train_df = train_df.dropna(subset=[text_col])
#     val_df   = val_df.dropna(subset=[text_col])
#     test_df  = test_df.dropna(subset=[text_col])

#     # 2. Drop within-split duplicate texts (keep first occurrence)
#     for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
#         n_dup = int(df.duplicated(subset=[text_col], keep="first").sum())
#         if n_dup:
#             fixes[f"dropped_duplicate_texts_{name}"] = n_dup
#     train_df = train_df.drop_duplicates(subset=[text_col], keep="first")
#     val_df   = val_df.drop_duplicates(subset=[text_col], keep="first")
#     test_df  = test_df.drop_duplicates(subset=[text_col], keep="first")

#     # 3. Remove leakage: val/test are more sacred than train
#     val_test_texts = set(val_df[text_col].astype(str)) | set(test_df[text_col].astype(str))
#     leak_train = train_df[train_df[text_col].astype(str).isin(val_test_texts)]
#     if len(leak_train):
#         fixes["dropped_leaked_from_train"] = int(len(leak_train))
#         train_df = train_df[~train_df[text_col].astype(str).isin(val_test_texts)]

#     test_texts = set(test_df[text_col].astype(str))
#     leak_val = val_df[val_df[text_col].astype(str).isin(test_texts)]
#     if len(leak_val):
#         fixes["dropped_leaked_from_val"] = int(len(leak_val))
#         val_df = val_df[~val_df[text_col].astype(str).isin(test_texts)]

#     return (train_df.reset_index(drop=True),
#             val_df.reset_index(drop=True),
#             test_df.reset_index(drop=True),
#             fixes)

# def run_data_qa(train_df, val_df, test_df, mode="auto_fix", text_col="input_text"):
#     """
#     v6 NEW dispatcher.
#     mode:
#       'strict'   -> raise on ANY violation (v5.2 behavior)
#       'warn'     -> print violations, continue unchanged
#       'auto_fix' -> automatically remove duplicates + leakage, then re-validate
#     """
#     report = validate_data(train_df, val_df, test_df, text_col)
#     violations = _qa_violations(report)

#     if not violations:
#         _log("  ✅ Data QA passed — no violations found")
#         return train_df, val_df, test_df, report

#     for msg in violations:
#         _log(f"  ⚠️  {msg}")

#     if mode == "strict":
#         raise ValueError("❌ Data QA failed in strict mode. Fix the data or use qa_mode='auto_fix'.")
#     if mode == "warn":
#         _log("  ⚠️  qa_mode='warn' — continuing WITHOUT fixing the issues above.")
#         return train_df, val_df, test_df, report

#     # auto_fix
#     _log("  🔧 qa_mode='auto_fix' — cleaning data automatically...")
#     train_df, val_df, test_df, fixes = auto_fix_data(train_df, val_df, test_df, text_col)
#     for k, v in fixes.items():
#         _log(f"     - {k}: {v} rows")
#     report = validate_data(train_df, val_df, test_df, text_col)
#     _log(f"  ✅ Data re-validated | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
#     return train_df, val_df, test_df, report


# # ================================================================
# # 2. EMPIRICAL TOKEN LENGTH & TRUNCATION RATE   — v6: sampled
# # ================================================================

# def analyze_token_lengths(texts, tokenizer, current_max_len, sample_size=None, seed=42):
#     texts = list(texts)
#     sampled = False
#     if sample_size and len(texts) > sample_size:
#         rng = np.random.RandomState(seed)
#         idx = rng.choice(len(texts), sample_size, replace=False)
#         texts = [texts[i] for i in idx]
#         sampled = True

#     lengths = [len(tokenizer.encode(str(x), add_special_tokens=True)) for x in texts]
#     truncation_rate = float(np.mean(np.array(lengths) > current_max_len) * 100)

#     _log("\n  [Token Length Distribution]" + (f"  (sampled {len(texts)} texts)" if sampled else ""))
#     for p in [90, 95, 97, 98, 99, 99.5, 100]:
#         _log(f"    {p:>5.1f}% -> {np.percentile(lengths, p):.0f} tokens")
#     _log(f"    Current MAX_LEN = {current_max_len}")
#     _log(f"    Truncation Rate = {truncation_rate:.2f}%")
#     return lengths, truncation_rate


# # ================================================================
# # 3. PAIRED BOOTSTRAP CI
# # ================================================================

# def compute_bootstrap(y_true, y_pred, target_names, n_bootstrap=1000, ci=0.95, seed=42):
#     n = len(y_true)
#     rng = np.random.RandomState(seed)
#     acc_s, mac_s, wei_s, mcc_s, bal_s = [], [], [], [], []
#     pc_f1_s = {cls: [] for cls in target_names}

#     for _ in range(n_bootstrap):
#         idx = rng.randint(0, n, n)
#         yt, yp = y_true[idx], y_pred[idx]
#         acc_s.append(accuracy_score(yt, yp))
#         mac_s.append(f1_score(yt, yp, average="macro", zero_division=0))
#         wei_s.append(f1_score(yt, yp, average="weighted", zero_division=0))
#         mcc_s.append(matthews_corrcoef(yt, yp))
#         bal_s.append(balanced_accuracy_score(yt, yp))
#         pc = f1_score(yt, yp, average=None, labels=list(range(len(target_names))), zero_division=0)
#         for i, cls in enumerate(target_names):
#             pc_f1_s[cls].append(pc[i])

#     def get_ci(scores):
#         s = np.array(scores)
#         return float(np.mean(s)), float(np.percentile(s, (1-ci)/2*100)), float(np.percentile(s, (1+ci)/2*100))

#     return {
#         "accuracy": get_ci(acc_s), "macro_f1": get_ci(mac_s),
#         "weighted_f1": get_ci(wei_s), "mcc": get_ci(mcc_s), "balanced_accuracy": get_ci(bal_s),
#         "per_class_f1": {cls: get_ci(scores) for cls, scores in pc_f1_s.items()}
#     }


# # ================================================================
# # 4. CALIBRATION & ECE (MEAN CLASS ECE)
# # ================================================================

# def compute_ece_and_plot(y_true, y_probs, target_names, save_path, n_bins=10):
#     fig, axes = plt.subplots(1, len(target_names), figsize=(5 * len(target_names), 5))
#     if len(target_names) == 1: axes = [axes]
#     bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
#     ece_scores = {}

#     for i, cls in enumerate(target_names):
#         y_bin = (y_true == i).astype(int)
#         prob_pos = y_probs[:, i]
#         if y_bin.sum() == 0:
#             axes[i].text(0.5, 0.5, "Not in test set", ha="center", va="center")
#             ece_scores[cls] = None
#             continue
#         ece, bin_accs, bin_confs = 0.0, [], []
#         for b in range(n_bins):
#             if b == n_bins - 1:
#                 mask = (prob_pos >= bin_edges[b]) & (prob_pos <= bin_edges[b+1])
#             else:
#                 mask = (prob_pos >= bin_edges[b]) & (prob_pos < bin_edges[b+1])
#             count = int(mask.sum())
#             if count > 0:
#                 acc, conf = float(y_bin[mask].mean()), float(prob_pos[mask].mean())
#                 ece += (count / len(y_bin)) * abs(acc - conf)
#                 bin_accs.append(acc); bin_confs.append(conf)
#             else:
#                 bin_accs.append(np.nan); bin_confs.append(np.nan)
#         ece_scores[cls] = float(ece)

#         valid = ~np.isnan(bin_accs)
#         axes[i].plot(np.array(bin_confs)[valid], np.array(bin_accs)[valid], "s-", label=cls)
#         axes[i].plot([0, 1], [0, 1], "k--", label="Perfect")
#         axes[i].set_title(f"{cls}\nECE = {ece:.4f}")
#         axes[i].set_xlabel("Confidence"); axes[i].set_ylabel("Accuracy")
#         axes[i].set_xlim(0, 1); axes[i].set_ylim(0, 1); axes[i].grid(True, alpha=0.5)

#     plt.suptitle("Probability Calibration (TEST SET)", fontsize=13, y=1.02)
#     plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

#     valid_ece = [v for v in ece_scores.values() if v is not None]
#     mean_class_ece = float(np.mean(valid_ece)) if valid_ece else None
#     return ece_scores, mean_class_ece


# # ================================================================
# # 5. LABEL PREPARATION
# # ================================================================

# def _prepare_labels(label_series: pd.Series) -> list:
#     if pd.api.types.is_numeric_dtype(label_series):
#         return label_series.astype(int).tolist()
#     try:
#         return label_series.astype(int).tolist()
#     except Exception:
#         return label_series.map(SENT_MAP).astype(int).tolist()


# # ================================================================
# # 5b. EPOCH LOSS PRINTER CALLBACK   — v6 FIX: was missing in v5.2
# # ================================================================

# class EpochLossPrinter(TrainerCallback):
#     """Prints train/eval metrics as they arrive (with flush)."""
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if not logs:
#             return
#         ep = f"{state.epoch:.2f}" if state.epoch is not None else "?"
#         if "loss" in logs:
#             _log(f"    [epoch {ep}] train loss = {logs['loss']:.4f} | "
#                  f"lr = {logs.get('learning_rate', float('nan')):.2e}")
#         if "eval_loss" in logs:
#             _log(f"    [epoch {ep}] eval loss = {logs['eval_loss']:.4f} | "
#                  f"eval macro-F1 = {logs.get('eval_macro_f1', float('nan')):.4f}")


# # ================================================================
# # 6. LLRD + WEIGHTED TRAINER (with Optuna-safe optimizer reset)
# # ================================================================

# class LLRDWeightedTrainer(WeightedTrainer):
#     def __init__(self, *args, use_llrd=False, layer_decay=0.9, **kwargs):
#         self.use_llrd = use_llrd
#         self.layer_decay = layer_decay
#         super().__init__(*args, **kwargs)

#     def _reset_optimizer_and_scheduler(self):
#         self.optimizer = None
#         self.lr_scheduler = None

#     def _hp_search_setup(self, trial):
#         self._reset_optimizer_and_scheduler()
#         try:
#             super()._hp_search_setup(trial)
#         except AttributeError:
#             params = self.hp_space(trial)
#             for key, value in params.items():
#                 setattr(self.args, key, value)

#     def train(self, *args, **kwargs):
#         trial = kwargs.get("trial", None)
#         if trial is None and len(args) >= 2:
#             trial = args[1]
#         if trial is not None:
#             self._reset_optimizer_and_scheduler()
#         return super().train(*args, **kwargs)

#     def create_optimizer(self):
#         if getattr(self, "optimizer", None) is not None:
#             return self.optimizer
#         if not self.use_llrd:
#             optimizer = super().create_optimizer()
#             if getattr(self, "optimizer", None) is None:
#                 self.optimizer = optimizer
#             return self.optimizer
#         grouped_params = get_optimizer_params(
#             self.model, self.args.learning_rate,
#             self.args.weight_decay, layer_decay=self.layer_decay,
#         )
#         self.optimizer = torch.optim.AdamW(
#             grouped_params,
#             lr=self.args.learning_rate,
#             weight_decay=self.args.weight_decay,
#         )
#         return self.optimizer


# # ================================================================
# # 7. MAIN TWO-PHASE RESEARCH PIPELINE (v6)
# # ================================================================

# def run_research_pipeline_v6(
#     config: dict,
#     train_df: pd.DataFrame,
#     val_df: pd.DataFrame,
#     test_df: pd.DataFrame,
#     class_weights: torch.Tensor,
#     report_to: str = "none",
#     n_trials: int = 5,
# ) -> dict:
#     """
#     Two-phase research pipeline (v6):
#       Phase 0: Data QA (strict / warn / auto_fix)
#       Phase 1: Optional Optuna hyperparameter search (train + val only)
#       Phase 2: Final training with best hyperparameters (train + val)
#       Phase 3: Independent TEST evaluation with full diagnostics
#     """
#     set_seed(SEED)

#     # ── READ CONFIG ──
#     key        = config["key"]
#     label      = config["label"]
#     checkpoint = config["checkpoint"]
#     save_dir   = config["save_dir"]
#     output_dir = config["output_dir"]
#     tok_dir    = config["tok_dir"]

#     use_llrd         = config.get("use_llrd", False)
#     layer_decay      = config.get("layer_decay", 0.9)
#     use_optuna       = config.get("use_optuna", True)
#     use_custom_model = config.get("use_custom_model", False)
#     search_epochs    = config.get("search_epochs", 4)
#     default_warmup   = config.get("warmup_ratio", 0.1)
#     batch_candidates = config.get("batch_size_candidates", [16, 32])
#     wd_min, wd_max   = config.get("weight_decay_range", (0.01, 0.20))
#     save_total_limit = config.get("save_total_limit", 1)

#     # ── v6 NEW CONFIG ──
#     qa_mode              = config.get("qa_mode", "auto_fix")          # strict | warn | auto_fix
#     num_workers          = config.get("num_workers", 0)               # 0 = deadlock-proof
#     save_only_model      = config.get("save_only_model", True)        # smaller checkpoints
#     token_analysis_sample= config.get("token_analysis_sample", 5000)  # None = full scan
#     debug_mode           = config.get("debug_mode", False)            # tiny smoke test
#     debug_train_n        = config.get("debug_train_n", 256)
#     debug_eval_n         = config.get("debug_eval_n", 64)
#     watchdog_minutes     = config.get("watchdog_minutes", 30)
#     logging_steps        = config.get("logging_steps", 25)
#     min_free_gb          = config.get("min_free_gb", 3.0)

#     # ── DIRECTORIES ──
#     plots_dir = os.path.join(save_dir, "plots")
#     model_dir = os.path.join(save_dir, "model")
#     for d in [save_dir, tok_dir, output_dir, plots_dir, model_dir]:
#         os.makedirs(d, exist_ok=True)

#     if os.path.abspath(save_dir) == os.path.abspath(output_dir):
#         raise ValueError("save_dir and output_dir must be different.")

#     _log("\n" + "="*80 + f"\n  RESEARCH PIPELINE v6: {label}\n" + "="*80)

#     # ── PRE-FLIGHT: DISK SPACE (v6 NEW) ──
#     check_disk_space(output_dir, min_free_gb=min_free_gb)

#     # ── PHASE 0: DATA QA (v6 UPGRADED) ──
#     _log(f"\n[0] Data Quality Checks (mode={qa_mode})")
#     train_df, val_df, test_df, data_report = run_data_qa(
#         train_df, val_df, test_df, mode=qa_mode
#     )

#     # ── DEBUG MODE (v6 NEW): tiny subsets for smoke tests ──
#     if debug_mode:
#         _log(f"\n[DEBUG MODE] Subsampling: train={debug_train_n}, val/test={debug_eval_n}")
#         train_df = train_df.sample(min(debug_train_n, len(train_df)), random_state=SEED).reset_index(drop=True)
#         val_df   = val_df.sample(min(debug_eval_n, len(val_df)),     random_state=SEED).reset_index(drop=True)
#         test_df  = test_df.sample(min(debug_eval_n, len(test_df)),   random_state=SEED).reset_index(drop=True)

#     # ── PHASE 1: TOKENIZATION & LENGTH ANALYSIS ──
#     _log("\n[1] Tokenization & Length Analysis")
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     _, truncation_rate = analyze_token_lengths(
#         train_df["input_text"], tokenizer, MAX_LEN, sample_size=token_analysis_sample
#     )

#     # 🔴 Dynamic padding: padding=False here, DataCollator handles it per-batch
#     train_enc = tokenizer(train_df["input_text"].astype(str).tolist(), truncation=True, padding=False, max_length=MAX_LEN)
#     val_enc   = tokenizer(val_df["input_text"].astype(str).tolist(),   truncation=True, padding=False, max_length=MAX_LEN)
#     test_enc  = tokenizer(test_df["input_text"].astype(str).tolist(),  truncation=True, padding=False, max_length=MAX_LEN)

#     train_labels = _prepare_labels(train_df["label"])
#     val_labels   = _prepare_labels(val_df["label"])
#     test_labels  = _prepare_labels(test_df["label"])

#     train_ds = SentimentDataset(train_enc, train_labels)
#     val_ds   = SentimentDataset(val_enc,   val_labels)
#     test_ds  = SentimentDataset(test_enc,  test_labels)

#     labels       = list(range(N_LABELS))
#     target_names = [str(ID2LABEL[i]) for i in sorted(ID2LABEL.keys())]

#     _log(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

#     # ── MODEL INIT ──
#     def model_init(trial=None):
#         if use_custom_model:
#             return CustomSentimentModel(checkpoint, N_LABELS, ID2LABEL, SENT_MAP)
#         return AutoModelForSequenceClassification.from_pretrained(
#             checkpoint, num_labels=N_LABELS, id2label=ID2LABEL, label2id=SENT_MAP
#         )

#     # ── DATA COLLATOR (dynamic padding) ──
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

#     # ============================================================
#     # PHASE 1: OPTUNA SEARCH
#     # ============================================================
#     best_hparams = {}
#     best_objective = None

#     if use_optuna:
#         _log("\n" + "-"*80)
#         _log("  PHASE 1: OPTUNA HYPERPARAMETER SEARCH")
#         _log("-"*80)

#         search_args = TrainingArguments(
#             output_dir=output_dir,
#             optim="adamw_torch",
#             num_train_epochs=search_epochs,
#             per_device_train_batch_size=BATCH_TRAIN,
#             per_device_eval_batch_size=BATCH_EVAL,
#             learning_rate=LR,
#             weight_decay=WEIGHT_DECAY,
#             warmup_ratio=default_warmup,
#             gradient_accumulation_steps=GRAD_ACCUM,
#             seed=SEED,
#             eval_strategy="epoch",
#             save_strategy="no",
#             load_best_model_at_end=False,
#             gradient_checkpointing=True,
#             gradient_checkpointing_kwargs={"use_reentrant": False},
#             logging_strategy="steps",
#             logging_steps=logging_steps,
#             logging_first_step=True,
#             disable_tqdm=False,
#             report_to="none",
#             tf32=torch.cuda.is_available(),
#             dataloader_pin_memory=True,
#             dataloader_num_workers=num_workers,        # v6 FIX: 0 by default
#             **get_precision_kwargs(),
#         )

#         search_trainer = LLRDWeightedTrainer(
#             model_init=model_init,
#             args=search_args,
#             train_dataset=train_ds,
#             eval_dataset=val_ds,
#             compute_metrics=compute_metrics,
#             class_weights=class_weights,
#             data_collator=data_collator,
#             use_llrd=use_llrd,
#             layer_decay=layer_decay,
#             callbacks=[EpochLossPrinter()],
#         )

#         def hp_space(trial):
#             return {
#                 "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
#                 "weight_decay":  trial.suggest_float("weight_decay",  wd_min, wd_max),
#                 "warmup_ratio":  trial.suggest_float("warmup_ratio",  0.05, 0.20),
#                 "per_device_train_batch_size": trial.suggest_categorical(
#                     "per_device_train_batch_size", batch_candidates),
#             }

#         _watchdog_arm(watchdog_minutes)
#         try:
#             try:
#                 best_run = search_trainer.hyperparameter_search(
#                     direction="maximize",
#                     backend="optuna",
#                     hp_space=hp_space,
#                     n_trials=n_trials,
#                     compute_objective=lambda m: m["eval_macro_f1"],
#                     sampler=optuna.samplers.TPESampler(seed=SEED),
#                 )
#             except TypeError:
#                 best_run = search_trainer.hyperparameter_search(
#                     direction="maximize",
#                     backend="optuna",
#                     hp_space=hp_space,
#                     n_trials=n_trials,
#                     compute_objective=lambda m: m["eval_macro_f1"],
#                 )
#         finally:
#             _watchdog_disarm()

#         best_hparams   = dict(best_run.hyperparameters)
#         best_objective = best_run.objective

#         _log(f"\n  BEST OPTUNA Macro-F1 : {best_objective:.4f}")
#         for k, v in best_hparams.items():
#             _log(f"    {k:<35}: {v}")

#         del search_trainer
#         _clear_memory()
#     else:
#         best_hparams = {
#             "learning_rate": LR,
#             "weight_decay": WEIGHT_DECAY,
#             "per_device_train_batch_size": BATCH_TRAIN,
#             "warmup_ratio": default_warmup,
#         }

#     # ============================================================
#     # PHASE 2: FINAL TRAINING
#     # ============================================================
#     _log("\n" + "-"*80)
#     _log("  PHASE 2: FINAL TRAINING")
#     _log("-"*80)

#     final_args = TrainingArguments(
#         output_dir=output_dir,
#         optim="adamw_torch",
#         num_train_epochs=EPOCHS,
#         per_device_train_batch_size=best_hparams.get("per_device_train_batch_size", BATCH_TRAIN),
#         per_device_eval_batch_size=BATCH_EVAL,
#         learning_rate=best_hparams.get("learning_rate", LR),
#         weight_decay=best_hparams.get("weight_decay", WEIGHT_DECAY),
#         warmup_ratio=best_hparams.get("warmup_ratio", default_warmup),
#         gradient_accumulation_steps=GRAD_ACCUM,
#         seed=SEED,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         save_total_limit=save_total_limit,
#         save_only_model=save_only_model,               # v6 NEW: smaller checkpoints
#         load_best_model_at_end=True,
#         metric_for_best_model="macro_f1",              # v6 FIX: was defaulting to eval_loss
#         greater_is_better=True,                        # v6 FIX
#         logging_strategy="steps",                      # v6: never looks frozen
#         logging_steps=logging_steps,
#         logging_first_step=True,
#         disable_tqdm=False,
#         # gradient_checkpointing=True,
#         gradient_checkpointing_kwargs={"use_reentrant": False},
#         report_to=report_to,
#         tf32=torch.cuda.is_available(),
#         dataloader_pin_memory=True,
#         dataloader_num_workers=num_workers,            # v6 FIX: 0 by default
#         **get_precision_kwargs(),
#     )

#     final_trainer = LLRDWeightedTrainer(
#         model_init=model_init,
#         args=final_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         compute_metrics=compute_metrics,
#         class_weights=class_weights,
#         data_collator=data_collator,
#         use_llrd=use_llrd,
#         layer_decay=layer_decay,                       # v6 FIX: v5.2 dropped this
#         callbacks=[
#             EpochLossPrinter(),
#             EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE),
#         ],
#     )

#     _log("\n  ⏳ Starting final training...")          # v6 FIX: print BEFORE training
#     _watchdog_arm(watchdog_minutes)
#     try:
#         final_trainer.train()
#     finally:
#         _watchdog_disarm()
#     _log("\n  ✅ Final training completed. Best checkpoint loaded.")

#     # ── BEST CHECKPOINT INFO ──
#     best_epoch      = getattr(final_trainer.state, "best_epoch", None)
#     best_metric     = getattr(final_trainer.state, "best_metric", None)
#     best_checkpoint = getattr(final_trainer.state, "best_model_checkpoint", None)
#     global_step     = getattr(final_trainer.state, "global_step", None)

#     # ── TRAIN vs VAL DIAGNOSTICS ──
#     _log("\n[Train vs Validation Diagnostics]")
#     tr_m = final_trainer.evaluate(train_ds, metric_key_prefix="train")
#     vl_m = final_trainer.evaluate(val_ds,   metric_key_prefix="val")
#     tr_f1, vl_f1 = tr_m.get("train_macro_f1", 0), vl_m.get("val_macro_f1", 0)
#     train_val_gap = tr_f1 - vl_f1

#     if train_val_gap > 0.05:
#         gap_status = "⚠️ DIAGNOSTIC: Potential overfitting signal (Gap > 0.05)"
#     elif train_val_gap < -0.01:
#         gap_status = "⚠️ DIAGNOSTIC: Potential underfitting signal (Gap < -0.01)"
#     else:
#         gap_status = "✅ DIAGNOSTIC: No large train-validation performance gap detected"

#     _log(f"  Train Macro-F1: {tr_f1:.4f}")
#     _log(f"  Val   Macro-F1: {vl_f1:.4f}")
#     _log(f"  Gap (F1)      : {train_val_gap:+.4f}")
#     _log(f"  Status        : {gap_status}")

#     # ============================================================
#     # PHASE 3: INDEPENDENT TEST EVALUATION
#     # ============================================================
#     _log("\n" + "-"*80)
#     _log("  PHASE 3: INDEPENDENT TEST EVALUATION")
#     _log("-"*80)

#     test_pred = final_trainer.predict(test_ds)
#     logits = test_pred.predictions
#     if isinstance(logits, tuple):
#         logits = logits[0]
#     logits = np.asarray(logits, dtype=np.float32)
#     y_true = np.asarray(test_pred.label_ids, dtype=np.int64)

#     # Handle binary single-logit case safely
#     if N_LABELS == 2 and logits.ndim == 2 and logits.shape[1] == 1:
#         prob_pos = torch.sigmoid(torch.tensor(logits[:, 0])).numpy()
#         y_probs = np.column_stack([1 - prob_pos, prob_pos])
#         y_pred  = (prob_pos >= 0.5).astype(int)
#     elif N_LABELS == 2 and logits.ndim == 1:
#         prob_pos = torch.sigmoid(torch.tensor(logits)).numpy()
#         y_probs = np.column_stack([1 - prob_pos, prob_pos])
#         y_pred  = (prob_pos >= 0.5).astype(int)
#     else:
#         y_pred  = np.argmax(logits, axis=-1)
#         y_probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

#     # ── METRICS ──
#     report = classification_report(
#         y_true, y_pred, labels=labels, target_names=target_names,
#         output_dict=True, zero_division=0
#     )
#     report_df = pd.DataFrame(report).transpose()

#     per_class_f1 = {
#         cls: float(report[cls]["f1-score"]) if cls in report else 0.0
#         for cls in target_names
#     }

#     mcc = matthews_corrcoef(y_true, y_pred)
#     balanced_acc = balanced_accuracy_score(y_true, y_pred)
#     test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
#     val_test_gap = vl_f1 - test_f1

#     y_bin = label_binarize(y_true, classes=labels)
#     roc_auc, pr_auc = {}, {}
#     for i in range(N_LABELS):
#         if len(np.unique(y_bin[:, i])) < 2: continue
#         fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
#         roc_auc[target_names[i]] = float(auc(fpr, tpr))
#         pr_auc[target_names[i]] = float(average_precision_score(y_bin[:, i], y_probs[:, i]))

#     try:
#         macro_roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
#     except ValueError:
#         macro_roc_auc = None

#     _log(f"\n  Test Accuracy       : {report['accuracy']:.4f}")
#     _log(f"  Balanced Accuracy   : {balanced_acc:.4f}")
#     _log(f"  Test Macro-F1       : {test_f1:.4f}")
#     _log(f"  Test Weighted-F1    : {report['weighted avg']['f1-score']:.4f}")
#     _log(f"  MCC                 : {mcc:.4f}")
#     _log(f"  Macro ROC-AUC       : {macro_roc_auc:.4f}" if macro_roc_auc else "  Macro ROC-AUC       : N/A")
#     _log(f"  Val-Test Gap        : {val_test_gap:+.4f}")

#     # ── BOOTSTRAP CI ──
#     _log("\n[Bootstrap CI]")
#     boot = compute_bootstrap(y_true, y_pred, target_names)

#     # ── CALIBRATION ──
#     _log("\n[Calibration]")
#     ece_scores, mean_class_ece = compute_ece_and_plot(
#         y_true, y_probs, target_names,
#         save_path=os.path.join(plots_dir, "calibration_test.png")
#     )

#     # ── ERROR ANALYSIS ──
#     _log("\n[Error Analysis]")
#     lengths = [len(tokenizer.encode(str(test_df["input_text"].iloc[i]), add_special_tokens=True))
#                for i in range(len(y_true))]
#     errors = []
#     for i in range(len(y_true)):
#         if y_true[i] != y_pred[i]:
#             errors.append({
#                 "text": test_df["input_text"].iloc[i],
#                 "true": target_names[y_true[i]],
#                 "pred": target_names[y_pred[i]],
#                 "confidence": float(y_probs[i, y_pred[i]]),
#                 "token_length": lengths[i]
#             })
#     errors_df = pd.DataFrame(errors)
#     if not errors_df.empty:
#         errors_df = errors_df.sort_values("confidence", ascending=False)
#     errors_df.to_csv(os.path.join(save_dir, "error_analysis.csv"), index=False)
#     _log(f"  Saved {len(errors_df)} mistakes to error_analysis.csv")

#     # ── VISUALIZATIONS ──
#     _log("\n[Saving Visualizations]")

#     # Loss curves
#     history = final_trainer.state.log_history
#     hist_df = pd.DataFrame(history)
#     tl = [l['loss'] for l in history if 'loss' in l and 'eval_loss' not in l]
#     te = [l['epoch'] for l in history if 'loss' in l and 'eval_loss' not in l]
#     vl_loss = [l['eval_loss'] for l in history if 'eval_loss' in l]
#     ve = [l['epoch'] for l in history if 'eval_loss' in l]

#     plt.figure(figsize=(10, 5))
#     if tl: plt.plot(te, tl, label="Train Loss", alpha=0.7)
#     if vl_loss: plt.plot(ve, vl_loss, label="Val Loss", marker="o", linewidth=2)
#     plt.title(f"Loss Curves: {label}")
#     plt.xlabel("Epoch"); plt.ylabel("Loss")
#     plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=150)
#     plt.close()

#     # Confusion matrix
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     plt.figure(figsize=(7, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=target_names, yticklabels=target_names)
#     plt.title(f"Confusion Matrix (TEST): {label}")
#     plt.xlabel("Predicted"); plt.ylabel("True")
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, "confusion_matrix_test.png"), dpi=150)
#     plt.close()

#     # ROC curves
#     plt.figure(figsize=(8, 6))
#     for i in range(N_LABELS):
#         if len(np.unique(y_bin[:, i])) < 2: continue
#         fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
#         class_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=2, label=f"{target_names[i]} (AUC={class_auc:.3f})")
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.xlabel("FPR"); plt.ylabel("TPR")
#     plt.title(f"ROC Curves (TEST): {label}")
#     plt.legend(loc="lower right")
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, "roc_test.png"), dpi=150)
#     plt.close()

#     # ── SAVE ARTIFACTS ──
#     _log("\n[Saving Artifacts]")

#     if hasattr(final_trainer.model, "backbone") and \
#        hasattr(final_trainer.model.backbone, "gradient_checkpointing_disable"):
#         final_trainer.model.backbone.gradient_checkpointing_disable()
#     for p in final_trainer.model.parameters():
#         p.data = p.data.contiguous()
#     tokenizer.save_pretrained(tok_dir)
#     final_trainer.save_model(model_dir)

#     # Save config.json
#     with open(os.path.join(save_dir, "config.json"), "w") as f:
#         json.dump(config, f, indent=2, default=str)

#     # Save metrics.json  (v6 FIX: Path -> str for JSON)
#     metrics_out = {
#         "test_n": int(len(y_true)),
#         "class_distributions": data_report["class_distribution"],
#         "accuracy": float(report["accuracy"]),
#         "balanced_accuracy": float(balanced_acc),
#         "macro_f1": float(test_f1),
#         "weighted_f1": float(report["weighted avg"]["f1-score"]),
#         "mcc": float(mcc),
#         "macro_roc_auc": float(macro_roc_auc) if macro_roc_auc is not None else None,
#         "mean_class_ece": mean_class_ece,
#         "truncation_rate": truncation_rate,
#         "per_class_f1": per_class_f1,
#         "roc_auc": roc_auc,
#         "pr_auc": pr_auc,
#         "train_f1": float(tr_f1),
#         "val_f1": float(vl_f1),
#         "test_f1": float(test_f1),
#         "train_val_gap": float(train_val_gap),
#         "val_test_gap": float(val_test_gap),
#         "gap_status": gap_status,
#         "best_epoch": best_epoch,
#         "best_val_metric": best_metric,
#         "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
#         "global_step": global_step,
#         "best_hparams": best_hparams,
#         "best_optuna_objective": best_objective,
#         "bootstrap_ci": boot,
#         "ece_scores": ece_scores,
#         "data_validation": {k: v for k, v in data_report.items() if k != "class_distribution"},
#         "environment": get_environment_info(),
#     }
#     with open(os.path.join(save_dir, "metrics.json"), "w") as f:
#         json.dump(metrics_out, f, indent=2, default=str)

#     # Save CSVs
#     report_df.to_csv(os.path.join(save_dir, "classification_report.csv"))
#     hist_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

#     preds_df = pd.DataFrame({"true_label": y_true, "predicted_label": y_pred})
#     for i, cls in enumerate(target_names):
#         preds_df[f"prob_{cls}"] = y_probs[:, i]
#     preds_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

#     if best_hparams:
#         best_params_df = pd.DataFrame({
#             "parameter": list(best_hparams.keys()),
#             "value": list(best_hparams.values())
#         })
#         best_params_df.to_csv(os.path.join(save_dir, "best_hyperparameters.csv"), index=False)

#     _log(f"  Model       → {model_dir}")
#     _log(f"  Tokenizer   → {tok_dir}")
#     _log(f"  Plots       → {plots_dir}")
#     _log(f"  Config      → {save_dir}/config.json")
#     _log(f"  Metrics     → {save_dir}/metrics.json")

#     # ── CLEANUP ──
#     del final_trainer
#     _clear_memory()

#     _log("\n" + "="*80)
#     _log("  PIPELINE COMPLETED SUCCESSFULLY")
#     _log("="*80)

#     return {
#         "label": label,
#         "best_hparams": best_hparams,
#         "best_optuna_objective": best_objective,
#         "test_accuracy": float(report["accuracy"]),
#         "test_macro_f1": test_f1,
#         "test_weighted_f1": float(report["weighted avg"]["f1-score"]),
#         "balanced_accuracy": balanced_acc,
#         "mcc": mcc,
#         "macro_roc_auc": macro_roc_auc,
#         "mean_class_ece": mean_class_ece,
#         "per_class_f1": per_class_f1,
#         "roc_auc": roc_auc,
#         "pr_auc": pr_auc,
#         "train_f1": tr_f1,
#         "val_f1": vl_f1,
#         "train_val_gap": train_val_gap,
#         "val_test_gap": val_test_gap,
#         "gap_status": gap_status,
#         "bootstrap_ci": boot,
#         "ece_scores": ece_scores,
#         "classification_report": report_df,
#         "history": hist_df,
#         "predictions_df": preds_df,
#         "errors_df": errors_df,
#         "data_validation": {k: v for k, v in data_report.items() if k != "class_distribution"},
#     }


# # ================================================================
# # 8. CUSTOM HYPERPARAMETER RUNNER (VALIDATION-BASED SELECTION)
# # ================================================================
# if 'run_history' not in globals(): run_history = []
# if 'all_results' not in globals(): all_results = {}

# BASE_CONFIGS = {
#     "muril": {"key": "muril", "label": "MuRIL", "checkpoint": "google/muril-base-cased"},
#     "xlmr":  {"key": "xlmr",  "label": "XLM-RoBERTa", "checkpoint": "xlm-roberta-base"},
#     "mbert": {"key": "mbert", "label": "mBERT", "checkpoint": "bert-base-multilingual-cased"},
# }

# DEFAULT_HPARAMS = {
#     "learning_rate": 2e-5, "weight_decay": 0.15, "warmup_ratio": 0.1,
#     "batch_size": 16, "use_llrd": True, "layer_decay": 0.9,
#     "use_custom_model": False, "use_optuna": True, "search_epochs": 4,
#     "batch_size_candidates": [16, 32], "weight_decay_range": (0.01, 0.20),
#     "save_total_limit": 1,
#     # ── v6 additions ──
#     "qa_mode": "auto_fix",          # strict | warn | auto_fix
#     "num_workers": 0,               # 0 = deadlock-proof
#     "save_only_model": True,        # smaller checkpoints, fewer disk failures
#     "token_analysis_sample": 5000,  # None = scan every row (slow)
#     "debug_mode": False,            # True = tiny smoke test
#     "watchdog_minutes": 30,         # 0 = disable stall watchdog
# }

# def run_model(model_key, run_name=None, clean_old=True, n_trials=5, **custom_hparams):
#     """
#     Run a single model with optional custom hyperparameters.

#     Examples:
#         run_model("xlmr")
#         run_model("xlmr", learning_rate=3e-5)
#         run_model("muril", run_name="muril_no_llrd", use_llrd=False)
#         run_model("mbert", debug_mode=True, use_optuna=False)   # 2-min smoke test
#     """
#     if model_key not in BASE_CONFIGS:
#         raise ValueError(f"Unknown model '{model_key}'. Available: {list(BASE_CONFIGS.keys())}")

#     cfg = BASE_CONFIGS[model_key].copy()
#     cfg.update(DEFAULT_HPARAMS)
#     cfg.update(custom_hparams)
#     if run_name is None: run_name = model_key
#     cfg["key"], cfg["save_dir"] = run_name, f"./results_{run_name}"
#     cfg["output_dir"], cfg["tok_dir"] = f"./runs/{run_name}", f"./tokenizers_{run_name}"

#     if clean_old:
#         for d in [cfg["save_dir"], cfg["output_dir"], cfg["tok_dir"]]:
#             if os.path.exists(d): shutil.rmtree(d)

#     global class_weights
#     if 'class_weights' not in globals():
#         if 'train_df' not in globals():
#             raise ValueError("train_df is not defined. Load the dataset before calling run_model().")
#         classes = np.unique(train_df['label'].astype(int).to_numpy())
#         cw = compute_class_weight(
#             class_weight='balanced',
#             classes=classes,
#             y=train_df['label'].astype(int).to_numpy(),
#         )
#         class_weights = torch.tensor(cw, dtype=torch.float)
#         _log(f"Class weights (Pos / Neu / Neg): {class_weights.numpy().round(4)}")

#     _log(f"\n{'#'*80}\n  RUNNING: {cfg['label']} | Run: {run_name}\n{'#'*80}")

#     result = run_research_pipeline_v6(
#         config=cfg,
#         train_df=train_df,
#         val_df=val_df,
#         test_df=test_df,
#         class_weights=class_weights,
#         report_to="none",
#         n_trials=n_trials,
#     )
#     all_results[run_name] = result

#     run_history.append({
#         "run_name": run_name, "model": model_key,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
#         "val_macro_f1": result.get("val_f1", None),
#         "test_macro_f1": result["test_macro_f1"],
#         "test_accuracy": result["test_accuracy"],
#         "mcc": result.get("mcc", None),
#         "balanced_accuracy": result.get("balanced_accuracy", None),
#         "mean_class_ece": result.get("mean_class_ece", None),
#         "gap_status": result.get("gap_status", "")
#     })
#     _log(f"\n✅ {run_name} Done | Val Macro-F1: {result.get('val_f1', 0):.4f} | "
#          f"Test Macro-F1: {result['test_macro_f1']:.4f}")
#     return result

# def run_all_models(model_keys=("xlmr", "mbert", "muril"), n_trials=5, **custom_hparams):
#     """v6 NEW: run several models back-to-back; failures don't stop the batch."""
#     for k in model_keys:
#         try:
#             run_model(k, n_trials=n_trials, **custom_hparams)
#         except Exception as e:
#             _log(f"❌ {k} FAILED: {e}")
#             _clear_memory()
#     return show_comparison()

# def show_comparison():
#     """Show comparison table. Selects best by VALIDATION score, not test score."""
#     if not run_history:
#         _log("No runs completed yet.")
#         return None
#     df = pd.DataFrame(run_history)

#     _log("\n" + "=" * 120)
#     _log("RUN COMPARISON TABLE")
#     _log("=" * 120)
#     _log(df[["run_name", "model", "learning_rate", "val_macro_f1", "test_macro_f1",
#              "balanced_accuracy", "mean_class_ece", "gap_status"]].to_string(index=False))

#     # 🔴 CRITICAL: Select best configuration by VALIDATION score to prevent test-set leakage
#     best_idx = df["val_macro_f1"].idxmax()
#     best = df.iloc[best_idx]

#     _log(f"\n🏆 Best Configuration (Selected by Validation): {best['run_name']}")
#     _log(f"   Val Macro-F1 = {best['val_macro_f1']:.4f} | Test Macro-F1 = {best['test_macro_f1']:.4f}")
#     return df


# In[54]:


"""
RESEARCH-GRADE SENTIMENT PIPELINE — v6.1 (SPEED-OPTIMIZED, SINGLE SEED)
=========================================================================

Changes from v6:
  [SPEED]  Single seed only (no multi-seed loop)
  [SPEED]  gradient_checkpointing OFF by default (configurable)
  [SPEED]  dataloader_num_workers=0 (avoids Windows/Jupyter deadlock)
  [SPEED]  Train diagnostics on SUBSET (default 1000 samples)
  [SPEED]  save_only_model=True (smaller checkpoints, faster I/O)
  [SPEED]  save_total_limit=1 (less disk writes)
  [SPEED]  batch_size=32, grad_accum=1 (configurable)
  [SPEED]  Sampled token-length analysis (5000 samples)
  [SPEED]  tf32=True + auto bf16/fp16
  [SPEED]  logging_strategy="steps" for live progress visibility
  [KEEP]   Dynamic padding via DataCollatorWithPadding
  [KEEP]   Early stopping, best-checkpoint loading
  [KEEP]   Bootstrap CI, ECE, ROC, error analysis (on TEST only)

Assumes the following exist earlier in your notebook/module:
    Constants:
        MAX_LEN, SEED, EPOCHS, EARLY_STOP_PATIENCE,
        GRAD_ACCUM, BATCH_TRAIN, BATCH_EVAL,
        LR, WEIGHT_DECAY, N_LABELS, ID2LABEL, SENT_MAP
    Classes:
        SentimentDataset, CustomSentimentModel, WeightedTrainer
    Functions:
        get_optimizer_params(model, lr, weight_decay, layer_decay=...),
        get_precision_kwargs(), compute_metrics(eval_pred)
"""

import os
import gc
import sys
import json
import shutil
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import optuna
import transformers
import sklearn
import math
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorWithPadding,
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    f1_score, accuracy_score, matthews_corrcoef,
    average_precision_score, roc_auc_score, balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize


# ================================================================
# 0. REPRODUCIBILITY & ENVIRONMENT
# ================================================================

def set_seed(seed):
    """Set ALL random seeds for full reproducibility (single seed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_environment_info():
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else "N/A",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "precision_mode": "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16",
    }


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _log(msg=""):
    """Print with flush to avoid silent-hang appearance in notebooks."""
    print(msg, flush=True)


# ================================================================
# 1. DATA CLEANING / MAPPING / STRATIFIED SPLIT
# ================================================================

def prepare_dataset(df, text_col="input_text", label_col="sentiment_raw",
                    seed=42, test_size=0.1):
    _log("\n" + "=" * 80)
    _log("  DATA PREPARATION")
    _log("=" * 80)

    initial = len(df)
    df = df.dropna(subset=[text_col]).drop_duplicates(subset=[text_col]).reset_index(drop=True)
    _log(f"  Rows after cleaning: {len(df)} (removed {initial - len(df)})")

    if "label" not in df.columns or df["label"].dtype == object:
        df["label"] = df[label_col].map(SENT_MAP)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed,
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=(test_size / (1.0 - test_size)),
        stratify=train_val_df["label"], random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    _log(f"\n  Train : {len(train_df)} rows")
    _log(f"  Val   : {len(val_df)} rows")
    _log(f"  Test  : {len(test_df)} rows")
    _log(f"\n  Train dist: {train_df['label'].value_counts().to_dict()}")
    _log(f"  Val   dist: {val_df['label'].value_counts().to_dict()}")
    _log(f"  Test  dist: {test_df['label'].value_counts().to_dict()}")

    classes = np.unique(train_df["label"].to_numpy())
    cw = compute_class_weight(class_weight="balanced", classes=classes,
                              y=train_df["label"].to_numpy())
    class_weights = torch.tensor(cw, dtype=torch.float)
    _log(f"\n  ⚖️ Class weights: {class_weights.numpy().round(4)}")

    return train_df, val_df, test_df, class_weights


# ================================================================
# 2. STRICT DATA VALIDATION
# ================================================================

def validate_data(train_df, val_df, test_df, text_col="input_text", strict=True):
    report = {}
    report["missing"] = {
        s: int(df[text_col].isna().sum())
        for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df])
    }
    report["duplicates"] = {}
    report["duplicate_label_conflicts"] = {}
    for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        dup_mask = df.duplicated(subset=[text_col], keep=False)
        dup_df = df[dup_mask].copy()
        report["duplicates"][s] = int(len(dup_df))
        if dup_df.empty:
            report["duplicate_label_conflicts"][s] = 0
            continue
        conflicting = dup_df.groupby(text_col)["label"].nunique()
        report["duplicate_label_conflicts"][s] = int((conflicting > 1).sum())

    report["label_conflicts"] = {}
    for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        lc = df.groupby(text_col)["label"].nunique()
        report["label_conflicts"][s] = int((lc > 1).sum())

    sets = {s: set(df[text_col].astype(str))
            for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df])}
    report["leakage"] = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    report["class_distribution"] = {
        s: df["label"].value_counts().to_dict()
        for s, df in zip(["train", "val", "test"], [train_df, val_df, test_df])
    }

    if strict:
        if any(v > 0 for v in report["missing"].values()):
            raise ValueError(f"❌ Missing values: {report['missing']}")
        if any(v > 0 for v in report["duplicate_label_conflicts"].values()):
            raise ValueError(f"❌ Duplicate label conflicts: {report['duplicate_label_conflicts']}")
        if any(v > 0 for v in report["leakage"].values()):
            raise ValueError(f"❌ Cross-split leakage: {report['leakage']}")
        if any(v > 0 for v in report["label_conflicts"].values()):
            raise ValueError(f"❌ Label conflicts: {report['label_conflicts']}")
    return report


# ================================================================
# 3. TOKEN LENGTH ANALYSIS (SAMPLED FOR SPEED)
# ================================================================

def analyze_token_lengths(texts, tokenizer, current_max_len, sample_size=5000):
    texts = list(texts)
    sampled = False
    if sample_size and len(texts) > sample_size:
        idx = np.random.RandomState(42).choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in idx]
        sampled = True

    lengths = [len(tokenizer.encode(str(x), add_special_tokens=True)) for x in texts]
    truncation_rate = float(np.mean(np.array(lengths) > current_max_len) * 100)

    _log("\n  [Token Length Distribution]" + (f"  (sampled {len(texts)})" if sampled else ""))
    for p in [90, 95, 97, 98, 99, 99.5, 100]:
        _log(f"    {p:>5.1f}% → {np.percentile(lengths, p):.0f} tokens")
    _log(f"    Current MAX_LEN = {current_max_len}")
    _log(f"    Truncation Rate = {truncation_rate:.2f}%")
    return lengths, truncation_rate


# ================================================================
# 4. PAIRED BOOTSTRAP CI
# ================================================================

def compute_bootstrap(y_true, y_pred, target_names, n_bootstrap=1000, ci=0.95, seed=42):
    n = len(y_true)
    rng = np.random.RandomState(seed)
    acc_s, mac_s, wei_s, mcc_s, bal_s = [], [], [], [], []
    pc_f1_s = {cls: [] for cls in target_names}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        acc_s.append(accuracy_score(yt, yp))
        mac_s.append(f1_score(yt, yp, average="macro", zero_division=0))
        wei_s.append(f1_score(yt, yp, average="weighted", zero_division=0))
        mcc_s.append(matthews_corrcoef(yt, yp))
        bal_s.append(balanced_accuracy_score(yt, yp))
        pc = f1_score(yt, yp, average=None, labels=list(range(len(target_names))), zero_division=0)
        for i, cls in enumerate(target_names):
            pc_f1_s[cls].append(pc[i])

    def get_ci(scores):
        s = np.array(scores)
        return (float(np.mean(s)),
                float(np.percentile(s, (1 - ci) / 2 * 100)),
                float(np.percentile(s, (1 + ci) / 2 * 100)))

    return {
        "accuracy": get_ci(acc_s), "macro_f1": get_ci(mac_s),
        "weighted_f1": get_ci(wei_s), "mcc": get_ci(mcc_s),
        "balanced_accuracy": get_ci(bal_s),
        "per_class_f1": {cls: get_ci(scores) for cls, scores in pc_f1_s.items()},
    }


# ================================================================
# 5. CALIBRATION & ECE
# ================================================================

def compute_ece_and_plot(y_true, y_probs, target_names, save_path, n_bins=10):
    fig, axes = plt.subplots(1, len(target_names), figsize=(5 * len(target_names), 5))
    if len(target_names) == 1:
        axes = [axes]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_scores = {}

    for i, cls in enumerate(target_names):
        y_bin = (y_true == i).astype(int)
        prob_pos = y_probs[:, i]
        if y_bin.sum() == 0:
            axes[i].text(0.5, 0.5, "Not in test set", ha="center", va="center")
            ece_scores[cls] = None
            continue
        ece, bin_accs, bin_confs = 0.0, [], []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (prob_pos >= lo) & (prob_pos <= hi if b == n_bins - 1 else prob_pos < hi)
            count = mask.sum()
            if count > 0:
                acc, conf = float(y_bin[mask].mean()), float(prob_pos[mask].mean())
                ece += (count / len(y_bin)) * abs(acc - conf)
                bin_accs.append(acc)
                bin_confs.append(conf)
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
        ece_scores[cls] = float(ece)
        valid = ~np.isnan(bin_accs)
        axes[i].plot(np.array(bin_confs)[valid], np.array(bin_accs)[valid], "s-", label=cls)
        axes[i].plot([0, 1], [0, 1], "k--", label="Perfect")
        axes[i].set_title(f"{cls}\nECE = {ece:.4f}")
        axes[i].set_xlabel("Confidence")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.5)

    plt.suptitle("Probability Calibration (TEST SET)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    valid_ece = [v for v in ece_scores.values() if v is not None]
    mean_class_ece = float(np.mean(valid_ece)) if valid_ece else None
    return ece_scores, mean_class_ece


# ================================================================
# 6. LABEL PREPARATION
# ================================================================

def _prepare_labels(label_series: pd.Series) -> list:
    if pd.api.types.is_numeric_dtype(label_series):
        return label_series.astype(int).tolist()
    try:
        return label_series.astype(int).tolist()
    except Exception:
        return label_series.map(SENT_MAP).astype(int).tolist()


# ================================================================
# 7. EPOCH METRICS PRINTER CALLBACK
# ================================================================

class EpochMetricsPrinter(TrainerCallback):
    """Prints epoch-level metrics in old-style format."""

    def __init__(self):
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs and "eval_loss" not in logs:
            self.last_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or state.epoch is None:
            return
        epoch = int(round(state.epoch))
        total_epochs = int(args.num_train_epochs)
        _log("\n" + "=" * 62)
        _log(f"  📊 EPOCH {epoch}/{total_epochs}")
        _log("=" * 62)
        if self.last_train_loss is not None:
            _log(f"  Train Loss          : {self.last_train_loss:.6f}")
        else:
            _log("  Train Loss          : N/A")
        val_loss = metrics.get("eval_loss")
        val_acc  = metrics.get("eval_accuracy")
        val_f1   = metrics.get("eval_macro_f1")
        if val_loss is not None:
            _log(f"  Validation Loss     : {val_loss:.6f}")
        if val_acc is not None:
            _log(f"  Validation Accuracy : {val_acc:.6f}")
        if val_f1 is not None:
            _log(f"  Validation Macro-F1 : {val_f1:.6f}")
        _log("=" * 62 + "\n")


# ================================================================
# 8. LLRD + WEIGHTED TRAINER
# ================================================================

class LLRDWeightedTrainer(WeightedTrainer):
    def __init__(self, *args, use_llrd=False, layer_decay=0.9, **kwargs):
        self.use_llrd = use_llrd
        self.layer_decay = layer_decay
        super().__init__(*args, **kwargs)

    def _reset_optimizer_and_scheduler(self):
        self.optimizer = None
        self.lr_scheduler = None

    def _hp_search_setup(self, trial):
        self._reset_optimizer_and_scheduler()
        try:
            super()._hp_search_setup(trial)
        except AttributeError:
            params = self.hp_space(trial)
            for key, value in params.items():
                setattr(self.args, key, value)

    def train(self, *args, **kwargs):
        trial = kwargs.get("trial", None)
        if trial is None and len(args) >= 2:
            trial = args[1]
        if trial is not None:
            self._reset_optimizer_and_scheduler()
        return super().train(*args, **kwargs)

    def create_optimizer(self):
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer
        if not self.use_llrd:
            optimizer = super().create_optimizer()
            if getattr(self, "optimizer", None) is None:
                self.optimizer = optimizer
            return self.optimizer
        grouped_params = get_optimizer_params(
            self.model, self.args.learning_rate,
            self.args.weight_decay, layer_decay=self.layer_decay,
        )
        self.optimizer = torch.optim.AdamW(
            grouped_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


# ================================================================
# 9. MAIN PIPELINE v6.1 (SPEED-OPTIMIZED, SINGLE SEED)
# ================================================================

def run_pipeline_v6(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    test_df: pd.DataFrame, class_weights: torch.Tensor,
                    report_to: str = "none", n_trials: int = 5) -> dict:

    # 🔴 SINGLE SEED — set once, used everywhere
    set_seed(SEED)

    # ── READ CONFIG ──
    key        = config["key"]
    label      = config["label"]
    checkpoint = config["checkpoint"]
    save_dir   = config["save_dir"]
    output_dir = config["output_dir"]
    tok_dir    = config["tok_dir"]

    use_llrd         = config.get("use_llrd", False)
    layer_decay      = config.get("layer_decay", 0.9)
    use_optuna       = config.get("use_optuna", False)
    use_custom_model = config.get("use_custom_model", False)
    search_epochs    = config.get("search_epochs", 4)
    default_warmup   = config.get("warmup_ratio", 0.1)
    batch_candidates = config.get("batch_size_candidates", [16, 32])
    wd_min, wd_max   = config.get("weight_decay_range", (0.01, 0.20))
    save_total_limit = config.get("save_total_limit", 1)  # 🔴 SPEED: reduced from 2

    # 🔴 SPEED CONFIG
    grad_checkpointing  = config.get("gradient_checkpointing", False)  # OFF by default
    train_diag_size     = config.get("train_diagnostic_size", 1000)    # subset for diagnostics
    num_workers         = config.get("num_workers", 0)                 # avoid deadlock
    batch_train         = config.get("batch_size", BATCH_TRAIN)
    grad_accum          = config.get("grad_accum", GRAD_ACCUM)

    # ── DIRECTORIES ──
    plots_dir = os.path.join(save_dir, "plots")
    model_dir = os.path.join(save_dir, "model")
    for d in [save_dir, tok_dir, output_dir, plots_dir, model_dir]:
        os.makedirs(d, exist_ok=True)

    if os.path.abspath(save_dir) == os.path.abspath(output_dir):
        raise ValueError("save_dir and output_dir must be different.")

    _log("\n" + "=" * 80)
    _log(f"  PIPELINE v6.1 (SPEED-OPTIMIZED): {label}")
    _log("=" * 80)
    _log(f"  Seed            : {SEED} (single)")
    _log(f"  Optuna          : {use_optuna}")
    _log(f"  LLRD            : {use_llrd}")
    _log(f"  Grad Ckpt       : {grad_checkpointing}")
    _log(f"  Batch/Accum     : {batch_train}/{grad_accum}")
    _log(f"  Learning Rate   : {config.get('learning_rate', LR)}")

    # ============================================================
    # STEP 1: STRICT QA
    # ============================================================
    _log("\n[1] Strict Data QA")
    data_report = validate_data(train_df, val_df, test_df, strict=True)
    _log("  ✅ Passed strict QA")

    # ============================================================
    # STEP 2: TOKENIZATION
    # ============================================================
    _log("\n[2] Tokenization & Length Analysis")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    _, truncation_rate = analyze_token_lengths(
        train_df["input_text"], tokenizer, MAX_LEN, sample_size=5000,
    )

    # 🔴 Dynamic padding: padding=False here, DataCollator pads per-batch
    train_enc = tokenizer(train_df["input_text"].astype(str).tolist(),
                          truncation=True, padding=False, max_length=MAX_LEN)
    val_enc   = tokenizer(val_df["input_text"].astype(str).tolist(),
                          truncation=True, padding=False, max_length=MAX_LEN)
    test_enc  = tokenizer(test_df["input_text"].astype(str).tolist(),
                          truncation=True, padding=False, max_length=MAX_LEN)

    train_labels = _prepare_labels(train_df["label"])
    val_labels   = _prepare_labels(val_df["label"])
    test_labels  = _prepare_labels(test_df["label"])

    train_ds = SentimentDataset(train_enc, train_labels)
    val_ds   = SentimentDataset(val_enc, val_labels)
    test_ds  = SentimentDataset(test_enc, test_labels)

    labels       = list(range(N_LABELS))
    target_names = [str(ID2LABEL[i]) for i in sorted(ID2LABEL.keys())]
    _log(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── MODEL INIT ──
    def model_init(trial=None):
        if use_custom_model:
            return CustomSentimentModel(checkpoint, N_LABELS, ID2LABEL, SENT_MAP)
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=N_LABELS, id2label=ID2LABEL, label2id=SENT_MAP,
        )

    # ── Dynamic padding collator ──
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
    )

    # ============================================================
    # STEP 3: OPTUNA PHASE 1 (optional)
    # ============================================================
    best_hparams = {}
    best_objective = None

    if use_optuna:
        _log("\n" + "-" * 80)
        _log("  PHASE 1: OPTUNA SEARCH (Train + Val only)")
        _log("-" * 80)

        search_args = TrainingArguments(
            output_dir=output_dir,
            optim="adamw_torch",
            num_train_epochs=search_epochs,
            per_device_train_batch_size=batch_train,
            per_device_eval_batch_size=BATCH_EVAL,
            learning_rate=config.get("learning_rate", LR),
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=default_warmup,
            gradient_accumulation_steps=grad_accum,
            seed=SEED,
            eval_strategy="epoch",
            save_strategy="no",              # 🔴 SPEED: no checkpoints during search
            load_best_model_at_end=False,
            gradient_checkpointing=grad_checkpointing,
            logging_strategy="epoch",
            logging_steps=10,
            disable_tqdm=True,
            report_to="none",
            tf32=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=num_workers,
            **get_precision_kwargs(),
        )

        search_trainer = LLRDWeightedTrainer(
            model_init=model_init, args=search_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=compute_metrics, class_weights=class_weights,
            data_collator=data_collator, use_llrd=use_llrd, layer_decay=layer_decay,
            callbacks=[EpochMetricsPrinter()],
        )

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
                "weight_decay": trial.suggest_float("weight_decay", wd_min, wd_max),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.20),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", batch_candidates),
            }

        try:
            best_run = search_trainer.hyperparameter_search(
                direction="maximize", backend="optuna", hp_space=hp_space,
                n_trials=n_trials, compute_objective=lambda m: m["eval_macro_f1"],
                sampler=optuna.samplers.TPESampler(seed=SEED),
            )
        except TypeError:
            best_run = search_trainer.hyperparameter_search(
                direction="maximize", backend="optuna", hp_space=hp_space,
                n_trials=n_trials, compute_objective=lambda m: m["eval_macro_f1"],
            )

        best_hparams   = dict(best_run.hyperparameters)
        best_objective = best_run.objective
        _log(f"\n  BEST OPTUNA Macro-F1: {best_objective:.4f}")
        for k, v in best_hparams.items():
            _log(f"    {k:<35}: {v}")

        del search_trainer
        _clear_memory()

    else:
        _log("\n  Optuna disabled → Using fixed hyperparameters.")
        best_hparams = {
            "learning_rate": config.get("learning_rate", LR),
            "weight_decay": config.get("weight_decay", WEIGHT_DECAY),
            "warmup_ratio": config.get("warmup_ratio", default_warmup),
            "per_device_train_batch_size": config.get("batch_size", BATCH_TRAIN),
        }
        for k, v in best_hparams.items():
            _log(f"    {k:<35}: {v}")

    # ============================================================
    # STEP 4: FINAL TRAINING (SINGLE SEED)
    # ============================================================
    _log("\n" + "-" * 80)
    _log("  PHASE 2: FINAL TRAINING (single seed)")
    _log("-" * 80)

    final_args = TrainingArguments(
        output_dir=output_dir,
        optim="adamw_torch",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=best_hparams.get("per_device_train_batch_size", batch_train),
        per_device_eval_batch_size=BATCH_EVAL,
        learning_rate=best_hparams.get("learning_rate", LR),
        weight_decay=best_hparams.get("weight_decay", WEIGHT_DECAY),
        warmup_ratio=best_hparams.get("warmup_ratio", default_warmup),
        gradient_accumulation_steps=grad_accum,
        seed=SEED,                          # 🔴 SINGLE SEED
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,  # 🔴 SPEED: 1 checkpoint max
        save_only_model=True,               # 🔴 SPEED: skip optimizer/scheduler state
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",           # 🔴 SPEED: visible progress
        logging_steps=25,
        logging_first_step=True,
        disable_tqdm=False,
        gradient_checkpointing=grad_checkpointing,  # 🔴 SPEED: OFF by default
        report_to=report_to,
        tf32=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=num_workers,  # 🔴 SPEED: 0 avoids deadlock
        **get_precision_kwargs(),
    )

    final_trainer = LLRDWeightedTrainer(
        model_init=model_init, args=final_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, class_weights=class_weights,
        data_collator=data_collator, use_llrd=use_llrd, layer_decay=layer_decay,
        callbacks=[
            EpochMetricsPrinter(),
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE),
        ],
    )

    final_trainer.train()
    _log("\n  ✅ Training complete. Best checkpoint loaded.")

    # ── BEST CHECKPOINT INFO ──
    best_epoch      = getattr(final_trainer.state, "best_epoch", None)
    best_metric     = getattr(final_trainer.state, "best_metric", None)
    best_checkpoint = getattr(final_trainer.state, "best_model_checkpoint", None)
    global_step     = getattr(final_trainer.state, "global_step", None)

    # ============================================================
    # STEP 5: TRAIN vs VAL DIAGNOSTICS (SUBSET FOR SPEED)
    # ============================================================
    _log("\n[Train vs Validation Diagnostics]")

    # 🔴 SPEED: Evaluate only a SUBSET of training data
    train_diag_size = min(train_diag_size, len(train_ds))
    train_diag_ds = torch.utils.data.Subset(train_ds, list(range(train_diag_size)))

    tr_m = final_trainer.evaluate(train_diag_ds, metric_key_prefix="train")
    vl_m = final_trainer.evaluate(val_ds, metric_key_prefix="val")

    tr_f1 = tr_m.get("train_macro_f1", 0)
    vl_f1 = vl_m.get("val_macro_f1", 0)
    train_val_gap = tr_f1 - vl_f1

    if train_val_gap > 0.05:
        gap_status = "⚠️ Potential overfitting (Gap > 0.05)"
    elif train_val_gap < -0.01:
        gap_status = "⚠️ Potential underfitting (Gap < -0.01)"
    else:
        gap_status = "✅ Healthy generalization"

    _log(f"  Train Macro-F1 ({train_diag_size} samples): {tr_f1:.4f}")
    _log(f"  Val   Macro-F1 : {vl_f1:.4f}")
    _log(f"  Gap            : {train_val_gap:+.4f}")
    _log(f"  Status         : {gap_status}")

    # ============================================================
    # STEP 6: INDEPENDENT TEST EVALUATION
    # ============================================================
    _log("\n" + "-" * 80)
    _log("  INDEPENDENT TEST EVALUATION")
    _log("-" * 80)

    test_pred = final_trainer.predict(test_ds)
    logits = test_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float32)
    y_true = np.asarray(test_pred.label_ids, dtype=np.int64)

    if N_LABELS == 2 and logits.ndim == 2 and logits.shape[1] == 1:
        prob_pos = torch.sigmoid(torch.tensor(logits[:, 0])).numpy()
        y_probs = np.column_stack([1 - prob_pos, prob_pos])
        y_pred  = (prob_pos >= 0.5).astype(int)
    elif N_LABELS == 2 and logits.ndim == 1:
        prob_pos = torch.sigmoid(torch.tensor(logits)).numpy()
        y_probs = np.column_stack([1 - prob_pos, prob_pos])
        y_pred  = (prob_pos >= 0.5).astype(int)
    else:
        y_pred  = np.argmax(logits, axis=-1)
        y_probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

    # ── METRICS ──
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    per_class_f1 = {
        cls: float(report[cls]["f1-score"]) if cls in report else 0.0
        for cls in target_names
    }

    mcc          = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    test_f1      = f1_score(y_true, y_pred, average="macro", zero_division=0)
    val_test_gap = vl_f1 - test_f1

    y_bin = label_binarize(y_true, classes=labels)
    roc_auc, pr_auc = {}, {}
    for i in range(N_LABELS):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[target_names[i]] = float(auc(fpr, tpr))
        pr_auc[target_names[i]]  = float(average_precision_score(y_bin[:, i], y_probs[:, i]))

    try:
        macro_roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        macro_roc_auc = None

    _log(f"\n  Accuracy          : {report['accuracy']:.4f}")
    _log(f"  Balanced Accuracy : {balanced_acc:.4f}")
    _log(f"  Macro-F1          : {test_f1:.4f}")
    _log(f"  Weighted-F1       : {report['weighted avg']['f1-score']:.4f}")
    _log(f"  MCC               : {mcc:.4f}")
    if macro_roc_auc is not None:
        _log(f"  Macro ROC-AUC     : {macro_roc_auc:.4f}")
    _log(f"  Val→Test Gap      : {val_test_gap:+.4f}")

    # ── BOOTSTRAP CI ──
    _log("\n[Bootstrap CI (n=1000)]")
    boot = compute_bootstrap(y_true, y_pred, target_names)
    for metric, (mean, lo, hi) in boot.items():
        if metric != "per_class_f1":
            _log(f"  {metric:<20}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # ── CALIBRATION ──
    _log("\n[Calibration / ECE]")
    ece_scores, mean_class_ece = compute_ece_and_plot(
        y_true, y_probs, target_names,
        save_path=os.path.join(plots_dir, "calibration_test.png"),
    )
    for cls, ece in ece_scores.items():
        if ece is not None:
            s = "✅" if ece < 0.05 else ("🟠" if ece < 0.10 else "⚠️")
            _log(f"  {cls:<15}: ECE = {ece:.4f} {s}")
    if mean_class_ece is not None:
        _log(f"  Mean Class ECE  : {mean_class_ece:.4f}")

    # ── GENERALIZATION SUMMARY ──
    _log("\n" + "=" * 80)
    _log("  GENERALIZATION ANALYSIS")
    _log("=" * 80)
    _log(f"  Train F1  : {tr_f1:.4f}")
    _log(f"  Val   F1  : {vl_f1:.4f}")
    _log(f"  Test  F1  : {test_f1:.4f}")
    _log(f"  Train→Val : {train_val_gap:+.4f}")
    _log(f"  Val→Test  : {val_test_gap:+.4f}")
    _log(f"  Status    : {gap_status}")

    # ── ERROR ANALYSIS ──
    _log("\n[Error Analysis]")
    lengths = [
        len(tokenizer.encode(str(test_df["input_text"].iloc[i]), add_special_tokens=True))
        for i in range(len(y_true))
    ]
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                "text": test_df["input_text"].iloc[i],
                "true": target_names[y_true[i]],
                "pred": target_names[y_pred[i]],
                "confidence": float(y_probs[i, y_pred[i]]),
                "token_length": lengths[i],
            })
    errors_df = pd.DataFrame(errors).sort_values("confidence", ascending=False)
    errors_df.to_csv(os.path.join(save_dir, "error_analysis.csv"), index=False)
    _log(f"  Saved {len(errors_df)} errors → error_analysis.csv")

    # ── VISUALIZATIONS ──
    _log("\n[Saving Plots]")
    history = final_trainer.state.log_history
    hist_df = pd.DataFrame(history)

    # Loss curve
    tl = [l["loss"] for l in history if "loss" in l and "eval_loss" not in l]
    te = [l["epoch"] for l in history if "loss" in l and "eval_loss" not in l]
    vl_loss = [l["eval_loss"] for l in history if "eval_loss" in l]
    ve = [l["epoch"] for l in history if "eval_loss" in l]

    plt.figure(figsize=(10, 5))
    if tl:
        plt.plot(te, tl, marker="o", label="Train Loss")
    if vl_loss:
        plt.plot(ve, vl_loss, marker="o", label="Validation Loss")
    plt.title(f"Loss Curves: {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix (TEST): {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(N_LABELS):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{target_names[i]} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curves (TEST): {label}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_test.png"), dpi=150)
    plt.close()

    # ============================================================
    # SAVE ARTIFACTS
    # ============================================================
    _log("\n[Saving Artifacts]")

    if hasattr(final_trainer.model, "backbone") and \
       hasattr(final_trainer.model.backbone, "gradient_checkpointing_disable"):
        final_trainer.model.backbone.gradient_checkpointing_disable()

    for p in final_trainer.model.parameters():
        p.data = p.data.contiguous()

    final_trainer.save_model(model_dir)
    tokenizer.save_pretrained(tok_dir)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    metrics_out = {
        "test_n": len(y_true),
        "class_distributions": data_report["class_distribution"],
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(test_f1),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "mcc": float(mcc),
        "macro_roc_auc": float(macro_roc_auc) if macro_roc_auc is not None else None,
        "mean_class_ece": mean_class_ece,
        "truncation_rate": float(truncation_rate),
        "per_class_f1": per_class_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "train_f1": float(tr_f1),
        "val_f1": float(vl_f1),
        "test_f1": float(test_f1),
        "train_val_gap": float(train_val_gap),
        "val_test_gap": float(val_test_gap),
        "gap_status": gap_status,
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "global_step": global_step,
        "best_hparams": best_hparams,
        "bootstrap_ci": boot,
        "ece_scores": ece_scores,
        "data_validation": {k: v for k, v in data_report.items() if k != "class_distribution"},
        "environment": get_environment_info(),
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    report_df.to_csv(os.path.join(save_dir, "classification_report.csv"))
    hist_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    preds_df = pd.DataFrame({"true_label": y_true, "predicted_label": y_pred})
    for i, cls in enumerate(target_names):
        preds_df[f"prob_{cls}"] = y_probs[:, i]
    preds_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

    if best_hparams:
        pd.DataFrame({
            "parameter": list(best_hparams.keys()),
            "value": list(best_hparams.values()),
        }).to_csv(os.path.join(save_dir, "best_hyperparameters.csv"), index=False)

    _log(f"  Model     → {model_dir}")
    _log(f"  Tokenizer → {tok_dir}")
    _log(f"  Plots     → {plots_dir}")

    del final_trainer
    _clear_memory()

    _log("\n" + "=" * 80)
    _log("  PIPELINE v6.1 COMPLETED SUCCESSFULLY")
    _log("=" * 80)

    return {
        "label": label,
        "best_hparams": best_hparams,
        "test_accuracy": float(report["accuracy"]),
        "test_macro_f1": float(test_f1),
        "test_weighted_f1": float(report["weighted avg"]["f1-score"]),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
        "macro_roc_auc": macro_roc_auc,
        "mean_class_ece": mean_class_ece,
        "per_class_f1": per_class_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "train_f1": float(tr_f1),
        "val_f1": float(vl_f1),
        "train_val_gap": float(train_val_gap),
        "val_test_gap": float(val_test_gap),
        "gap_status": gap_status,
        "bootstrap_ci": boot,
        "ece_scores": ece_scores,
        "classification_report": report_df,
        "history": hist_df,
        "predictions_df": preds_df,
        "errors_df": errors_df,
    }


# ================================================================
# 10. RUNNER
# ================================================================

if "run_history" not in globals():
    run_history = []
if "all_results" not in globals():
    all_results = {}

BASE_CONFIGS = {
    "muril": {"key": "muril", "label": "MuRIL", "checkpoint": "google/muril-base-cased"},
    "xlmr":  {"key": "xlmr",  "label": "XLM-RoBERTa", "checkpoint": "xlm-roberta-base"},
    "mbert": {"key": "mbert", "label": "mBERT", "checkpoint": "bert-base-multilingual-cased"},
}

DEFAULT_HPARAMS = {
    "learning_rate": 2e-5,
    "weight_decay": 0.15,
    "warmup_ratio": 0.1,
    "batch_size": 32,                     # 🔴 SPEED: larger batch
    "grad_accum": 1,                     # 🔴 SPEED: no accumulation needed
    "use_llrd": False,
    "layer_decay": 0.9,
    "use_custom_model": False,
    "use_optuna": False,
    "search_epochs": 5,
    "batch_size_candidates": [16, 32],
    "weight_decay_range": (0.01, 0.20),
    "save_total_limit": 1,               # 🔴 SPEED
    "gradient_checkpointing": False,     # 🔴 SPEED: OFF by default
    "train_diagnostic_size": 1000,       # 🔴 SPEED: subset for diagnostics
    "num_workers": 0,                    # 🔴 SPEED: avoids deadlock
}


def run_model(model_key, run_name=None, clean_old=True, n_trials=5, **custom_hparams):
    if model_key not in BASE_CONFIGS:
        raise ValueError(f"Unknown '{model_key}'. Available: {list(BASE_CONFIGS.keys())}")

    cfg = BASE_CONFIGS[model_key].copy()
    cfg.update(DEFAULT_HPARAMS)
    cfg.update(custom_hparams)

    if run_name is None:
        run_name = model_key
    cfg["key"] = run_name

    # Output directories
    if model_key == "xlmr":
        cfg["save_dir"]   = os.path.join(MODEL_ROOT, f"xlm-roberta-base_{run_name}")
        cfg["output_dir"] = os.path.join(TRAINING_ROOT, run_name)
        cfg["tok_dir"]    = os.path.join(TOKENIZER_ROOT, f"xlm-roberta-base_{run_name}")
    elif model_key == "mbert":
        cfg["save_dir"]   = os.path.join(MODEL_ROOT, f"mbert_{run_name}")
        cfg["output_dir"] = os.path.join(TRAINING_ROOT, run_name)
        cfg["tok_dir"]    = os.path.join(TOKENIZER_ROOT, f"mbert_{run_name}")
    elif model_key == "muril":
        cfg["save_dir"]   = os.path.join(MODEL_ROOT, f"muril_{run_name}")
        cfg["output_dir"] = os.path.join(TRAINING_ROOT, run_name)
        cfg["tok_dir"]    = os.path.join(TOKENIZER_ROOT, f"muril_{run_name}")
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    if clean_old:
        for d in [cfg["save_dir"], cfg["output_dir"], cfg["tok_dir"]]:
            if os.path.exists(d):
                shutil.rmtree(d)

    # Class weights
    global class_weights
    if "class_weights" not in globals():
        if "train_df" not in globals():
            raise ValueError("train_df not defined. Run prepare_dataset() first.")
        classes = np.unique(train_df["label"].astype(int).to_numpy())
        cw = compute_class_weight("balanced", classes=classes,
                                  y=train_df["label"].astype(int).to_numpy())
        class_weights = torch.tensor(cw, dtype=torch.float)
        _log(f"Class weights: {class_weights.numpy().round(4)}")

    _log(f"\n{'#' * 80}")
    _log(f"  RUNNING: {cfg['label']} | Run: {run_name}")
    _log(f"  Model output : {cfg['save_dir']}")
    _log(f"  Checkpoints  : {cfg['output_dir']}")
    _log(f"  Tokenizer    : {cfg['tok_dir']}")
    _log(f"{'#' * 80}")

    result = run_pipeline_v6(
        config=cfg, train_df=train_df, val_df=val_df, test_df=test_df,
        class_weights=class_weights, report_to="none", n_trials=n_trials,
    )

    all_results[run_name] = result

    run_history.append({
        "run_name": run_name,
        "model": model_key,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "learning_rate": cfg["learning_rate"],
        "use_optuna": cfg["use_optuna"],
        "use_llrd": cfg["use_llrd"],
        "val_macro_f1": result.get("val_f1"),
        "test_macro_f1": result["test_macro_f1"],
        "test_accuracy": result["test_accuracy"],
        "mcc": result.get("mcc"),
        "balanced_accuracy": result.get("balanced_accuracy"),
        "mean_class_ece": result.get("mean_class_ece"),
        "gap_status": result.get("gap_status", ""),
    })

    _log(f"\n✅ {run_name} | Val F1: {result.get('val_f1', 0):.4f} "
         f"| Test F1: {result['test_macro_f1']:.4f}")
    return result


# ================================================================
# 11. COMPARISON
# ================================================================

def show_comparison():
    if not run_history:
        _log("No runs completed yet.")
        return None

    df = pd.DataFrame(run_history)
    cols = ["run_name", "model", "learning_rate", "use_optuna", "use_llrd",
            "val_macro_f1", "test_macro_f1", "balanced_accuracy",
            "mean_class_ece", "gap_status"]

    _log("\n" + "=" * 130)
    _log("RUN COMPARISON")
    _log("=" * 130)
    _log(df[cols].to_string(index=False))

    best_idx = df["val_macro_f1"].idxmax()
    best = df.iloc[best_idx]
    _log(f"\n🏆 Best (by Validation): {best['run_name']}")
    _log(f"   Val F1  = {best['val_macro_f1']:.4f}")
    _log(f"   Test F1 = {best['test_macro_f1']:.4f}")
    return df


# In[55]:


run_model(
    "xlmr",
    use_optuna=True,
    n_trials=5,
    search_epochs=5,
    use_llrd=True,
    layer_decay=0.85,
)


# In[28]:


# Fastest approach: Just use the best HPs directly, no searching
run_model(
    "xlmr",
    run_name="xlmr_fixed_best_hps",
    use_optuna=False,  # <--- Turn off Optuna to save time
    use_llrd=False,
    learning_rate=4.33e-06,
    weight_decay=0.190636,
    warmup_ratio=0.159799,
    batch_size=16,
)


# In[30]:


run_model(
    "muril",
    use_optuna=True,
    n_trials=5,
    search_epochs=5,
    use_llrd=True,
    layer_decay=0.85,
)


# In[20]:


# Run mBERT with your specific best hyperparameters
run_model(
    "mbert",
    run_name="mbert_best_hps",      # Custom name to identify this run
    clean_old=True,                 # Clears previous mBERT runs to save disk space
    use_optuna=False,               # Disable Optuna search since we have fixed HPs
    use_llrd=True,                  # Keep Layer-wise Learning Rate Decay enabled (set to False if you want to test without it)
    learning_rate=1.0952662748632564e-05,
    weight_decay=0.036503833523887946,
    warmup_ratio=0.09382169728028272,
    batch_size=32                   # This maps to per_device_train_batch_size in the pipeline
)


# In[31]:


run_model(
    "mbert",
    use_optuna=True,
    n_trials=5,
    search_epochs=5,
    use_llrd=True,
    layer_decay=0.85,
)


# In[ ]:


# Configuration for XLM-RoBERTa with best hyperparameters
xlmr_config = {
    'key': 'xlmr',
    'label': 'XLM-RoBERTa',
    'checkpoint': 'xlm-roberta-base',
    'save_dir': './results_xlmr',
    'output_dir': './runs/xlmr',
    'tok_dir': './tokenizers_xlmr',

    # Disable Optuna since we're using fixed best hyperparameters
    'use_optuna': False,

    # Optional settings
    'use_llrd': False,           # Layer-wise learning rate decay
    'layer_decay': 0.9,
    'use_custom_model': False,  # Use standard AutoModel

    # Best hyperparameters found
    'learning_rate': 4.33e-06,
    'weight_decay': 0.190635718,
    'warmup_ratio': 0.159799091,
    'batch_size': 16,
}

# Run the pipeline
result = run_training_pipeline_optimized(
    config=xlmr_config,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    class_weights=class_weights,
    report_to='none',
)

print(f"\n✅ XLM-RoBERTa Training Complete!")
print(f"Test Macro-F1: {result['test_macro_f1']:.4f}")
print(f"Test Accuracy: {result['test_accuracy']:.4f}")


# In[22]:


# import sys
# import io
# import faulthandler

# def _watchdog_arm(minutes):
#     if minutes and minutes > 0:
#         try:
#             # Jupyter's sys.stderr is an OutStream and lacks fileno().
#             # We attempt to use the original OS-level stderr stream instead.
#             if hasattr(sys, '__stderr__') and sys.__stderr__ is not None:
#                 faulthandler.dump_traceback_later(
#                     minutes * 60, 
#                     repeat=True, 
#                     file=sys.__stderr__
#                 )
#             else:
#                 raise io.UnsupportedOperation("No original stderr available")

#             _log(f"  ⏱️  Watchdog armed: stack dump every {minutes} min if stalled.")

#         except (io.UnsupportedOperation, AttributeError, ValueError, OSError):
#             # Fallback for environments where faulthandler cannot get a valid file descriptor
#             _log(f"  ⏱️  Watchdog disabled: Jupyter/IPython environment does not support faulthandler fileno().")


# In[23]:


run_model("xlmr", run_name="smoke", debug_mode=True)


# In[24]:


from sklearn.utils.class_weight import compute_class_weight

if 'class_weights' not in globals():
    classes = np.unique(train_df['label'].astype(int).to_numpy())
    cw = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_df['label'].astype(int).to_numpy(),
    )
    class_weights = torch.tensor(cw, dtype=torch.float)
    print(f"Class weights (Pos / Neu / Neg): {class_weights.numpy().round(4)}")


# ## 11. Disk Space Cleanup (safety net)
# 
# `run_sentiment_pipeline` already cleans up its own local checkpoint dir after each run; this cell is just a safety net in case a run was interrupted.

# In[25]:


import glob

for directory in glob.glob('./results_*'):
    if os.path.exists(directory):
        print(f"Removing {directory} to free up space...")
        shutil.rmtree(directory)

print("\nDisk cleanup complete.")


# ## 12. Comparison Table Against Thapa et al. (2023)
# 
# Built dynamically from the `results` dict — every entry reflects a model that was actually trained and evaluated in this notebook, not a remembered/hard-coded score. Run only the models you care about above; this table adapts to whatever is in `results`.

# In[26]:


PAPER_RESULTS = {
    'mBERT (paper)' : 0.61,
    'MuRIL (paper)' : 0.65,
    'XLM-R (paper)' : 0.67,
}

OUR_RESULTS = {
    f"{cfg['label']} (ours)": all_results[cfg['key']]['macro_f1']
    for cfg in MODEL_CONFIGS if cfg['key'] in all_results
}

all_scores = {**PAPER_RESULTS, **OUR_RESULTS}

print("\n📊 Macro-F1 Comparison")
print("-" * 50)
for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * int(score * 40)
    print(f"{name:<24} {score:.4f}  {bar}")

summary_df = pd.DataFrame([
    {
        'Model'          : cfg['label'],
        'Checkpoint'     : cfg['checkpoint'],
        'Macro F1 (ours)': all_results[cfg['key']]['macro_f1'],
        'Accuracy (ours)': all_results[cfg['key']]['accuracy'],
    }
    for cfg in MODEL_CONFIGS if cfg['key'] in all_results
]).sort_values('Macro F1 (ours)', ascending=False).reset_index(drop=True)

summary_df


# ```markdown
# ## 12.1. Visualizing Training & Validation Loss
# 
# The following function extracts the loss history from each model's `results` entry and plots them to check for convergence or overfitting.
# ```

# In[ ]:


plot_labels = list(all_scores.keys())
plot_scores = list(all_scores.values())
plot_colors = ['#4C72B0' if '(ours)' in l else '#B0B0B0' for l in plot_labels]

order = np.argsort(plot_scores)[::-1]
plot_labels = [plot_labels[i] for i in order]
plot_scores = [plot_scores[i] for i in order]
plot_colors = [plot_colors[i] for i in order]

plt.figure(figsize=(8, 5))
plt.barh(plot_labels, plot_scores, color=plot_colors)
plt.xlabel('Macro F1')
plt.title('Macro-F1: Paper Baselines vs. Ours')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('macro_f1_comparison.png', dpi=150)
plt.show()


# In[ ]:


def plot_loss_curves(results_dict, smooth=True, window=5):
    if not results_dict:
        print("No results found. Please train at least one model first.")
        return

    num_models = len(results_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), sharey=True)
    if num_models == 1: axes = [axes]

    for i, (model_key, data) in enumerate(results_dict.items()):
        history = data['history']

        train_loss = [log['loss'] for log in history if 'loss' in log]
        train_steps = [log['epoch'] for log in history if 'loss' in log]
        val_loss = [log['eval_loss'] for log in history if 'eval_loss' in log]
        val_epochs = [log['epoch'] for log in history if 'eval_loss' in log]

        ax = axes[i]

        # Original noisy training loss in light color
        ax.plot(train_steps, train_loss, color='blue', alpha=0.2, label='Raw Train Loss')

        if smooth and len(train_loss) > window:
            # Apply moving average to smooth out fluctuations
            smoothed_loss = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            ax.plot(train_steps[window-1:], smoothed_loss, color='blue', linewidth=2, label='Smoothed Train')

        ax.plot(val_epochs, val_loss, label='Validation Loss', color='red', marker='o', linewidth=2)

        ax.set_title(f'Loss Trends: {model_key}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# Re-run with smoothing to see the real trend
plot_loss_curves(results, smooth=True, window=10)


# ## 13. Inference Helper
# 
# Quick function to run inference on new text using any model that's been trained above. Reloads the model/tokenizer from its saved Drive directory (since `run_sentiment_pipeline` frees GPU memory after each run by default).

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained(model_key: str):
    """Reload a trained model + tokenizer from Drive using its entry in `results`."""
    r = results[model_key]
    tok = AutoTokenizer.from_pretrained(r['tok_save'])
    mdl = AutoModelForSequenceClassification.from_pretrained(r['save_dir']).to(device)
    return mdl, tok

def predict_sentiment(texts, model, tokenizer, device='cuda', batch_size=32):
    """
    Predict sentiment for a list of texts.
    Returns list of (label_str, confidence) tuples.
    """
    model.eval()
    results_out = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        for pred, prob in zip(preds, probs):
            results_out.append((ID2LABEL[pred], round(float(prob[pred]), 4)))

    return results_out


# ── Quick smoke test — uses whichever model in `results` scored highest ──────
if results:
    best_key = max(results, key=lambda k: results[k]['macro_f1'])
    demo_model, demo_tokenizer = load_trained(best_key)

    sample_texts  = val_df['input_text'].head(5).tolist()
    sample_labels = val_df['sentiment_raw'].head(5).tolist()

    predictions = predict_sentiment(sample_texts, demo_model, demo_tokenizer, device=device)
    print(f"Using best model so far: {best_key} (macro F1 = {results[best_key]['macro_f1']:.4f})\n")
    print(f"{'Text':<60} {'True':>10} {'Pred':>10} {'Conf':>6}")
    print("-" * 90)
    for text, true_lbl, (pred_lbl, conf) in zip(sample_texts, sample_labels, predictions):
        match = '✓' if true_lbl == pred_lbl else '✗'
        print(f"{str(text)[:58]:<60} {true_lbl:>10} {pred_lbl:>10} {conf:>6.3f} {match}")
else:
    print("No models trained yet — run one of the `run_sentiment_pipeline(...)` cells above first.")

