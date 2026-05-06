"""
Central configuration for the Workforce Intelligence System.
All hyperparameters, paths, and model settings in one place.
"""
import os
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

# --- NER Model (Agent 1) ---
# v3-v6: DistilBERT-base-cased (5 entity types, 11 BIO labels)
# v7:    backbone upgraded to BERT/RoBERTa-large on the M3 Ultra Mac Studio
#        with 10 entity types and 21 BIO labels. See ENTITY_SCHEMA_V7.md.
NER_MODEL_NAME = "distilbert-base-cased"  # v6 default; v7 notebook overrides to bert-large/roberta-large

# v6 label list — kept for rollback / backwards-compat. The v6 model
# checkpoint on disk embeds this ordering in its config.json, so loading
# it will use this set regardless of what NER_LABELS points to.
NER_LABELS_V6 = [
    "O",
    "B-SKILL", "I-SKILL",
    "B-CERT", "I-CERT",
    "B-DEGREE", "I-DEGREE",
    "B-EMPLOYER", "I-EMPLOYER",
    "B-YEARS_EXP", "I-YEARS_EXP",
]

# v7 label list — 10 entity types (5 legacy + 5 new: TOOL, INDUSTRY,
# LOCATION, PROJECT, SOFT_SKILL). Ordered so legacy B-/I- pairs stay
# adjacent and the 5 new types append at the tail — makes per-type
# evaluation slices trivial.
NER_LABELS_V7 = [
    "O",
    "B-SKILL", "I-SKILL",
    "B-TOOL", "I-TOOL",
    "B-CERT", "I-CERT",
    "B-DEGREE", "I-DEGREE",
    "B-EMPLOYER", "I-EMPLOYER",
    "B-YEARS_EXP", "I-YEARS_EXP",
    "B-INDUSTRY", "I-INDUSTRY",
    "B-LOCATION", "I-LOCATION",
    "B-PROJECT", "I-PROJECT",
    "B-SOFT_SKILL", "I-SOFT_SKILL",
]

# Canonical label set — defaults to v7 for new training runs. Loading an
# existing v6 checkpoint uses the model's own config.id2label so the two
# worlds coexist cleanly (evaluate_ner.py + ner_model.py derive the label
# map from the loaded model, not from these module-level constants).
NER_LABELS = NER_LABELS_V7
NER_LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: label for i, label in enumerate(NER_LABELS)}
NER_NUM_LABELS = len(NER_LABELS)
# Convenience list of bare entity types (no BIO prefix) derived from NER_LABELS.
NER_ENTITY_TYPES = [l[2:] for l in NER_LABELS if l.startswith("B-")]

NER_MAX_LENGTH = 256
NER_EPOCHS = 10
NER_BATCH_SIZE = 16
NER_LEARNING_RATE = 2e-5  # Small LR to avoid catastrophic forgetting

# --- SBERT Matching (Agent 1) ---
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MATCH_THRESHOLD = 0.60  # Minimum cosine similarity for a "match"

# --- Bi-LSTM Forecasting (Agent 2) ---
LSTM_INPUT_SIZE = 8      # Number of input features per timestep
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True
LSTM_SEQ_LENGTH = 12     # 12-month lookback window
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 1e-3

# --- Attrition Thresholds ---
ATTRITION_HIGH_RISK = 0.70
ATTRITION_MEDIUM_RISK = 0.40

# --- LLM Settings ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.3

# --- Dashboard ---
DASHBOARD_PORT = 8501
