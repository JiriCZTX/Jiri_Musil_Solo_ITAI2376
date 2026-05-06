"""
Agent 1 Deep Learning Model: Ensemble NER for Energy-Sector Resumes.

Combines two complementary transformer architectures:
  1. Fine-tuned DistilBERT (Module 05: Transformers) — token classification
     trained on energy-sector resume data for domain-specific extraction.
  2. GLiNER (Generalist Lightweight NER, NAACL 2024) — zero-shot span
     extraction using a bidirectional transformer that can identify any
     entity type specified at inference time.

The ensemble merges outputs from both models, leveraging DistilBERT's
domain-specific fine-tuning with GLiNER's zero-shot generalization to
achieve robust entity extraction across: SKILL, CERT, DEGREE, EMPLOYER,
YEARS_EXP.

Architecture:
  DistilBERT: pretrained → fine-tuned token classification head
  GLiNER: bidirectional transformer with span-level entity matching
  Ensemble: union-merge with deduplication and post-processing

Course Connection: Module 05 - Transformers / Attention Is All You Need
"""
import json
import re
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    NER_MODEL_NAME, NER_LABELS, NER_LABEL2ID, NER_ID2LABEL,
    NER_NUM_LABELS, NER_MAX_LENGTH, NER_EPOCHS, NER_BATCH_SIZE,
    NER_LEARNING_RATE, MODELS_DIR,
)


class ResumeNERDataset(Dataset):
    """Token-classified resume dataset for DistilBERT NER fine-tuning."""

    def __init__(self, data, tokenizer, max_length=NER_MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Tokenize with word-to-subtoken alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels to subword tokens
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                tag = ner_tags[word_idx] if word_idx < len(ner_tags) else "O"
                label_ids.append(NER_LABEL2ID.get(tag, 0))
            else:
                # Subword continuation: use I- tag if parent is B-
                tag = ner_tags[word_idx] if word_idx < len(ner_tags) else "O"
                if tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                label_ids.append(NER_LABEL2ID.get(tag, 0))
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class NEREngine:
    """Train and run the DistilBERT NER model for resume entity extraction."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(NER_MODEL_NAME)
        self.model = None

    def build_model(self):
        """Initialize DistilBERT with token classification head."""
        self.model = DistilBertForTokenClassification.from_pretrained(
            NER_MODEL_NAME,
            num_labels=NER_NUM_LABELS,
            id2label=NER_ID2LABEL,
            label2id=NER_LABEL2ID,
        ).to(self.device)
        return self.model

    def train(self, train_data, val_data=None, epochs=NER_EPOCHS):
        """
        Fine-tune DistilBERT on labeled resume NER data.

        Uses small learning rate (2e-5) to avoid catastrophic forgetting
        of pretrained representations, as learned in Lab 05.
        """
        if self.model is None:
            self.build_model()

        train_dataset = ResumeNERDataset(train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=NER_BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=NER_LEARNING_RATE,
            weight_decay=0.01,
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        self.model.train()
        history = []

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            history.append(avg_loss)
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Validation
            if val_data:
                metrics = self.evaluate(val_data)
                print(f"  Val F1: {metrics['weighted_f1']:.4f}")

        return history

    def evaluate(self, eval_data):
        """Evaluate NER model and return classification metrics."""
        self.model.eval()
        eval_dataset = ResumeNERDataset(eval_data, self.tokenizer)
        eval_loader = DataLoader(eval_dataset, batch_size=NER_BATCH_SIZE)

        all_preds, all_labels = [], []

        id2label = self._model_id2label()
        labels_list = [v for v in id2label.values() if v != "O"]

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                for pred_seq, label_seq in zip(preds, labels):
                    for p, l in zip(pred_seq, label_seq):
                        if l.item() != -100:  # Skip special tokens
                            all_preds.append(id2label[p.item()])
                            all_labels.append(id2label[l.item()])

        # Filter out 'O' for meaningful entity-level metrics
        entity_preds = [p for p, l in zip(all_preds, all_labels) if l != "O"]
        entity_labels = [l for l in all_labels if l != "O"]

        report = classification_report(
            all_labels, all_preds,
            labels=labels_list,
            output_dict=True, zero_division=0,
        )

        return {
            "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0),
            "report": report,
            "total_predictions": len(all_preds),
            "entity_predictions": len(entity_preds),
        }

    def _model_id2label(self):
        """Return the loaded model's own {int: str} BIO label map.

        HuggingFace serializes id2label with string keys in config.json, so
        we normalize back to ints. This lets v6 (11 labels) and v7 (21
        labels) both work without touching module-level constants.
        """
        raw = self.model.config.id2label
        return {int(k): v for k, v in raw.items()}

    def _model_entity_types(self):
        """Bare entity types (no BIO prefix) embedded in the loaded model."""
        out = []
        seen = set()
        for lbl in self._model_id2label().values():
            if lbl.startswith("B-"):
                et = lbl[2:]
                if et not in seen:
                    seen.add(et)
                    out.append(et)
        return out

    def extract_entities(self, text):
        """
        Extract named entities from raw resume text.

        Returns dict of entity_type -> list of extracted spans. Entity
        types are derived from the loaded model's own config, so a v6 model
        returns 5 types and a v7 model returns 10.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call build_model() or load() first.")

        self.model.eval()
        # Normalize whitespace: collapse newlines, tabs, and extra spaces
        import re
        clean = re.sub(r'\s+', ' ', text).strip()
        # Remove common punctuation that breaks entity spans
        clean = clean.replace(' —', ',').replace('—', ',')
        tokens = clean.split()

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=NER_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)[0]

        id2label = self._model_id2label()
        # Map subword predictions back to word-level
        word_ids = encoding.word_ids()
        word_preds = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in word_preds:
                word_preds[word_id] = id2label[preds[idx].item()]

        # Group consecutive entities into spans. Container is pre-populated
        # with all entity types the loaded model knows about.
        entities = {et: [] for et in self._model_entity_types()}
        current_entity = None
        current_tokens = []

        for word_idx in sorted(word_preds.keys()):
            if word_idx >= len(tokens):
                break
            label = word_preds[word_idx]
            token = tokens[word_idx]

            if label.startswith("B-"):
                if current_entity and current_tokens:
                    etype = current_entity.replace("B-", "").replace("I-", "")
                    if etype in entities:
                        entities[etype].append(" ".join(current_tokens))
                current_entity = label
                current_tokens = [token]
            elif label.startswith("I-") and current_entity:
                current_tokens.append(token)
            else:
                if current_entity and current_tokens:
                    etype = current_entity.replace("B-", "").replace("I-", "")
                    if etype in entities:
                        entities[etype].append(" ".join(current_tokens))
                current_entity = None
                current_tokens = []

        # Flush last entity
        if current_entity and current_tokens:
            etype = current_entity.replace("B-", "").replace("I-", "")
            if etype in entities:
                entities[etype].append(" ".join(current_tokens))

        return entities

    def save(self, path=None):
        """Save model and tokenizer."""
        path = Path(path or MODELS_DIR / "ner")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"NER model saved to {path}")

    def load(self, path=None):
        """Load saved model and tokenizer."""
        path = Path(path or MODELS_DIR / "ner")
        self.model = DistilBertForTokenClassification.from_pretrained(path).to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        print(f"NER model loaded from {path}")
        return self


class GLiNERExtractor:
    """
    Zero-shot NER using GLiNER (NAACL 2024).

    GLiNER uses a bidirectional transformer to extract arbitrary entity
    types at inference time without task specific fine tuning. It
    outperforms large language model baselines on NER benchmarks while
    running on CPU with under 500M parameters.

    Reference: Zaratiana et al., "GLiNER: Generalist Model for Named
    Entity Recognition using Bidirectional Transformer" (NAACL 2024)
    """

    GLINER_MODEL = "urchade/gliner_medium-v2.1"
    GLINER_LOCAL = Path(__file__).parent / "saved" / "gliner"

    # v6 labels — 5 types. Retained so the v5/v6 models still receive the
    # same query surface (GLiNER's zero-shot output depends on the exact
    # label wording).
    LABELS_V6 = [
        "technical skill",
        "professional certification",
        "academic degree",
        "employer or company",
        "years of experience",
    ]
    LABEL_MAP_V6 = {
        "technical skill": "SKILL",
        "professional certification": "CERT",
        "academic degree": "DEGREE",
        "employer or company": "EMPLOYER",
        "years of experience": "YEARS_EXP",
    }

    # v7 labels — 10 types. Wording tuned for GLiNER's zero-shot head:
    # descriptive labels outperform abbreviated ones (see GLiNER paper
    # §4.2). Legacy 5 kept identical so v6-trained spans are comparable.
    LABELS_V7 = [
        "technical skill",
        "software tool or platform",             # TOOL — NEW
        "professional certification",
        "academic degree",
        "employer or company",
        "years of experience",
        "energy industry sector or segment",     # INDUSTRY — NEW
        "work location or geographic basin",     # LOCATION — NEW
        "named project or facility",             # PROJECT — NEW
        "interpersonal or leadership skill",     # SOFT_SKILL — NEW
    ]
    LABEL_MAP_V7 = {
        "technical skill": "SKILL",
        "software tool or platform": "TOOL",
        "professional certification": "CERT",
        "academic degree": "DEGREE",
        "employer or company": "EMPLOYER",
        "years of experience": "YEARS_EXP",
        "energy industry sector or segment": "INDUSTRY",
        "work location or geographic basin": "LOCATION",
        "named project or facility": "PROJECT",
        "interpersonal or leadership skill": "SOFT_SKILL",
    }

    # Default to v7 — the fine-tuned weights in models/saved/gliner/ were
    # trained against this label surface. Callers can override via the
    # `labels_version` kwarg to load() for apples-to-apples v6 comparison.
    LABELS = LABELS_V7
    LABEL_MAP = LABEL_MAP_V7

    # Universities/schools should not be classified as employers
    UNIVERSITY_KEYWORDS = [
        "university", "college", "institute", "school",
        "a&m", "mit", "stanford", "caltech",
    ]

    def __init__(self, labels_version: str = "v7"):
        self.model = None
        self.labels_version = labels_version
        self._configure_labels(labels_version)

    def _configure_labels(self, version: str):
        """Select v6 (5-type) or v7 (10-type) GLiNER label surface."""
        if version == "v6":
            self.LABELS = list(self.LABELS_V6)
            self.LABEL_MAP = dict(self.LABEL_MAP_V6)
        else:
            self.LABELS = list(self.LABELS_V7)
            self.LABEL_MAP = dict(self.LABEL_MAP_V7)

    def load(self, labels_version: str = None):
        """Load GLiNER model — fine-tuned local if available, else pretrained.

        `labels_version` override lets evaluation swap between v6 and v7
        label surfaces for apples-to-apples comparison without reloading
        the weights.
        """
        if labels_version:
            self._configure_labels(labels_version)
        from gliner import GLiNER
        if self.GLINER_LOCAL.exists() and (self.GLINER_LOCAL / "gliner_config.json").exists():
            self.model = GLiNER.from_pretrained(str(self.GLINER_LOCAL))
            print(f"GLiNER model loaded: {self.GLINER_LOCAL} (energy fine-tuned, labels={self.labels_version})")
        else:
            self.model = GLiNER.from_pretrained(self.GLINER_MODEL)
            print(f"GLiNER model loaded: {self.GLINER_MODEL} (pretrained, labels={self.labels_version})")
        return self

    def _is_university(self, text):
        """Check if entity text refers to a university/school."""
        lower = text.lower()
        return any(kw in lower for kw in self.UNIVERSITY_KEYWORDS)

    def _clean_skill(self, text):
        """Remove leading verbs/phrases from skill entities."""
        prefixes = [
            "expertise in ", "proficient in ", "skilled in ",
            "familiar with ", "experienced in ", "led ",
            "managed ", "expert in ", "core ",
        ]
        lower = text.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):]
                break
        return text.strip()

    def extract_entities(self, text, threshold=0.4):
        """Extract entities using GLiNER zero-shot prediction."""
        if self.model is None:
            raise RuntimeError("GLiNER model not loaded. Call load() first.")

        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', text).strip()

        raw_entities = self.model.predict_entities(
            clean, self.LABELS, threshold=threshold
        )

        # Container is derived from the active LABEL_MAP (v6 = 5 types,
        # v7 = 10 types) — a single source of truth.
        entities = {etype: [] for etype in self.LABEL_MAP.values()}

        for ent in raw_entities:
            etype = self.LABEL_MAP.get(ent["label"])
            if not etype:
                continue
            span = ent["text"].strip()

            # Post-processing filters
            if etype == "EMPLOYER" and self._is_university(span):
                continue  # Skip universities misclassified as employers
            if etype == "SKILL":
                span = self._clean_skill(span)
                if len(span) < 2:
                    continue  # Skip single-char or empty skills
            if etype == "TOOL":
                # Tools should be named products, not generic categories.
                # GLiNER occasionally returns phrases like "control systems"
                # as TOOL; keep only concrete product-like spans.
                if len(span) < 2:
                    continue
            if etype == "LOCATION":
                # Reject PII: home addresses typically include digits +
                # street-name patterns. Professional basins/regions don't.
                if re.search(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(Street|St|Ave|Avenue|Road|Rd|Blvd|Boulevard)\b",
                             span):
                    continue
            if not span:
                continue

            entities[etype].append(span)

        # Regex fallback for years of experience (GLiNER sometimes misses)
        if "YEARS_EXP" in entities and not entities["YEARS_EXP"]:
            years_patterns = [
                r'(\d+\+?\s*years?\s+of\s+(?:hands-on\s+)?experience)',
                r'(over\s+\d+\s+years?)',
                r'(\d+-year\s+track\s+record)',
            ]
            for pat in years_patterns:
                match = re.search(pat, clean, re.IGNORECASE)
                if match:
                    entities["YEARS_EXP"].append(match.group(1))
                    break

        return entities


class EnsembleNEREngine:
    """
    Ensemble NER combining fine-tuned DistilBERT with zero-shot GLiNER.

    Strategy: union-merge with deduplication. Both models extract entities
    independently, then results are merged. This leverages DistilBERT's
    domain-specific training (catches energy terms it was fine-tuned on)
    with GLiNER's generalization (catches entities the fine-tuned model
    missed due to limited training data).
    """

    def __init__(self, device=None):
        self.distilbert = NEREngine(device=device)
        self.gliner = GLiNERExtractor()
        self._loaded = False

    def load(self, ner_path=None):
        """Load both models."""
        self.distilbert.load(ner_path)
        self.gliner.load()
        self._loaded = True
        print("Ensemble NER loaded (DistilBERT + GLiNER)")
        return self

    def build_and_train(self, train_data, val_data=None, epochs=NER_EPOCHS):
        """Train the DistilBERT component and load GLiNER."""
        history = self.distilbert.build_model()
        history = self.distilbert.train(train_data, val_data, epochs)
        self.gliner.load()
        self._loaded = True
        return history

    def save(self, path=None):
        """Save the DistilBERT component (GLiNER uses pretrained weights)."""
        self.distilbert.save(path)

    @staticmethod
    def _normalize(text):
        """Lowercase and strip for dedup comparison."""
        return re.sub(r'\s+', ' ', text).lower().strip()

    # Words that are noise if extracted as standalone entities
    NOISE_WORDS = {
        "skills", "skill", "expertise", "core", "tools", "experience",
        "proficient", "familiar", "strong", "key", "other", "additional",
    }

    # Known software tools sometimes misclassified as certifications
    SOFTWARE_TOOLS = {
        "etap", "hysys", "matlab", "python", "autocad", "sap pm",
        "maximo", "skm powertools", "sql", "c++",
    }

    # University keywords for filtering from DEGREE results
    UNIVERSITY_KEYWORDS = [
        "university", "college", "institute", "school",
        "a&m", "mit", "stanford", "caltech",
    ]

    @staticmethod
    def _deduplicate(entities_list):
        """Remove duplicate and substring entities (case-insensitive, keep longest)."""
        if not entities_list:
            return []
        # Sort by length descending so longer spans are kept first
        sorted_ents = sorted(entities_list, key=len, reverse=True)
        seen_normalized = set()
        deduped = []
        for ent in sorted_ents:
            key = re.sub(r'\s+', ' ', ent).lower().strip()
            # Skip if this is a substring of an already-kept entity
            is_substring = any(key in kept for kept in seen_normalized)
            if key not in seen_normalized and not is_substring:
                seen_normalized.add(key)
                deduped.append(ent)
        return deduped

    def _clean_entities(self, entities):
        """Post-processing to improve entity quality.

        Routing rule for misclassified software tools:
          - v7 output has a TOOL type → move SOFTWARE_TOOLS from CERT to TOOL
          - v6 output has no TOOL type → move them to SKILL (legacy behavior)
        """
        cleaned = {etype: [] for etype in entities}
        reclassified_tool_like = []
        target_for_tools = "TOOL" if "TOOL" in entities else "SKILL"

        for etype, vals in entities.items():
            for v in vals:
                v_lower = v.lower().strip()

                # Remove noise words
                if v_lower in self.NOISE_WORDS:
                    continue

                # Move software tools out of CERT (they're TOOLs, not CERTs)
                if etype == "CERT" and v_lower in self.SOFTWARE_TOOLS:
                    reclassified_tool_like.append(v)
                    continue

                # v7: SKILL entries that are actually known TOOLs move to TOOL
                if target_for_tools == "TOOL" and etype == "SKILL" and v_lower in self.SOFTWARE_TOOLS:
                    reclassified_tool_like.append(v)
                    continue

                # Remove university names from DEGREE
                if etype == "DEGREE":
                    if any(kw in v_lower for kw in self.UNIVERSITY_KEYWORDS):
                        continue

                # Skip very short entities (likely noise)
                if len(v.strip()) < 2:
                    continue

                cleaned[etype].append(v)

        # Route reclassified software tools to the right bucket.
        if reclassified_tool_like:
            cleaned.setdefault(target_for_tools, []).extend(reclassified_tool_like)

        return cleaned

    def extract_entities(self, text):
        """
        Extract entities using both models and merge results.

        Returns dict of entity_type -> list of extracted spans. The union
        of types present in EITHER engine's output is iterated, so this
        works for v6 (5 types) and v7 (10 types) without changes.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded.")

        # Get predictions from both models
        distilbert_entities = self.distilbert.extract_entities(text)
        gliner_entities = self.gliner.extract_entities(text)

        # Union of entity types present in either engine's output.
        all_types = set(distilbert_entities) | set(gliner_entities)

        # Union-merge: combine all entities, then deduplicate
        merged = {}
        for etype in all_types:
            combined = distilbert_entities.get(etype, []) + gliner_entities.get(etype, [])
            merged[etype] = self._deduplicate(combined)

        # Post-processing cleanup
        merged = self._clean_entities(merged)

        # Final dedup after cleanup (software tools may have moved to SKILL/TOOL)
        for etype in merged:
            merged[etype] = self._deduplicate(merged[etype])

        return merged

    def evaluate(self, eval_data):
        """Evaluate the DistilBERT component on labeled data."""
        return self.distilbert.evaluate(eval_data)


# =============================================================================
# Per-class owners loaded from disk (no retraining)
# =============================================================================
# These two extractors are the "real" trained owners that RoutedNEREngine wires
# into per_class_router. They each load weights from models/saved/ — nothing
# is retrained or modified here. If a load fails, the corresponding owner is
# set to None and the router degrades gracefully (the type bucket is still
# present in the output dict, populated only by the gazetteer overlay if any).


class _V11ModernBertExtractor:
    """ModernBERT v11 token classifier loaded from ``models/saved/ner_v11/``.

    The v11 weights cover all 10 production entity types (BIO labelling for
    SKILL/TOOL/CERT/DEGREE/EMPLOYER/YEARS_EXP/INDUSTRY/LOCATION/PROJECT/
    SOFT_SKILL — 21 ids in id2label). The model uses ModernBERT, which the
    legacy :class:`NEREngine` cannot load (that class hardcodes
    DistilBertTokenizerFast/DistilBertForTokenClassification). This wrapper
    uses ``transformers.AutoModelForTokenClassification`` so it works without
    retraining or modifying weights.
    """

    PATH = Path(__file__).parent / "saved" / "ner_v11"

    def __init__(self, device: str = "cpu"):
        # MPS / CUDA don't always support every ModernBERT op cleanly; CPU
        # inference on Apple Silicon for one resume is fast enough (~150 ms).
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.PATH))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.PATH))
        self.model.to(self.device).eval()
        return self

    def predict_spans(self, text: str):
        """Run the v11 BIO classifier; return List[Span] with character offsets.

        Whitespace is preserved (we do NOT collapse it) so the offsets we hand
        back land on the original text — important for the router's
        overlap dedup and for any downstream highlighting.
        """
        from models.per_class_router import Span
        if self.model is None or self.tokenizer is None:
            return []
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            return_offsets_mapping=True, max_length=512,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        pred_ids = out.logits.argmax(-1)[0].tolist()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        # Group adjacent B-/I- tokens of the same entity type into spans.
        spans = []
        cur_type = None
        cur_start = None
        cur_end = None
        for tok_idx, lid in enumerate(pred_ids):
            label = id2label.get(lid, "O")
            tok_start, tok_end = offsets[tok_idx]
            if tok_start == tok_end:  # special tokens
                continue
            if label.startswith("B-"):
                if cur_type is not None:
                    spans.append((cur_type, cur_start, cur_end))
                cur_type = label[2:]
                cur_start, cur_end = tok_start, tok_end
            elif label.startswith("I-") and cur_type == label[2:]:
                cur_end = tok_end
            else:
                if cur_type is not None:
                    spans.append((cur_type, cur_start, cur_end))
                cur_type = None
                cur_start = cur_end = None
        if cur_type is not None:
            spans.append((cur_type, cur_start, cur_end))

        return [
            Span(text=text[s:e], type=t, start=s, end=e,
                 score=0.95, source="v11_modernbert")
            for (t, s, e) in spans
            if s is not None and e is not None and 0 <= s < e <= len(text)
        ]


class _CandidateGLiNERExtractor(GLiNERExtractor):
    """10-type GLiNER candidate from
    ``models/saved/gliner_v3_10type_candidate_20260427_225326/model/``.

    Subclass of :class:`GLiNERExtractor` that only changes ``GLINER_LOCAL``.
    Uses the v7 (10-type) label surface, exactly the same one the candidate
    was fine-tuned against (per its training config).
    """

    GLINER_LOCAL = (
        Path(__file__).parent / "saved"
        / "gliner_v3_10type_candidate_20260427_225326" / "model"
    )

    def __init__(self):
        super().__init__(labels_version="v7")

    def predict_spans(self, text: str, threshold: float = 0.4):
        """GLiNER returns offsets natively. Convert to router Span."""
        from models.per_class_router import Span
        if self.model is None:
            return []
        clean = re.sub(r"\s+", " ", text).strip()
        # Use the original text for offsets; the cleaned text alters whitespace.
        # GLiNER returns offsets relative to the input it was given (the
        # cleaned text), so we re-anchor by string search on `text` for each
        # surface form. This is the same approach used elsewhere in this
        # module when an extractor doesn't preserve original-text offsets.
        raw = self.model.predict_entities(clean, self.LABELS, threshold=threshold)
        out = []
        for ent in raw:
            etype = self.LABEL_MAP.get(ent["label"])
            if not etype:
                continue
            span_text = ent["text"].strip()
            if not span_text:
                continue
            pos = text.find(span_text)
            if pos < 0:
                pos = text.lower().find(span_text.lower())
            if pos < 0:
                start, end = 0, 0
            else:
                start, end = pos, pos + len(span_text)
            out.append(Span(
                text=span_text, type=etype, start=start, end=end,
                score=float(ent.get("score", 0.7)),
                source="gliner_v3_10type_candidate",
            ))
        return out


class RoutedNEREngine:
    """
    Wires the per-class router to the actual trained owners on disk.

    Class ownership (after this wiring):

    +------------+------------------------------+----------------------------+
    | Type       | Intended owner               | Actual owner here          |
    +============+==============================+============================+
    | SKILL      | v6 ensemble                  | v6 EnsembleNEREngine       |
    | CERT       | v6 ensemble                  | v6 EnsembleNEREngine       |
    |            |                              | + gazetteer overlay (add)  |
    | DEGREE     | v6 ensemble                  | v6 EnsembleNEREngine       |
    | EMPLOYER   | v6 ensemble                  | v6 EnsembleNEREngine       |
    | YEARS_EXP  | v6 ensemble                  | v6 EnsembleNEREngine       |
    | TOOL       | v11 ModernBERT + gaz overlay | v11 ModernBERT + gazetteer |
    | INDUSTRY   | candidate/Gate 2 + gazetteer | gliner_v3_10type_candidate |
    |            |                              | + gazetteer overlay        |
    | LOCATION   | gazetteer / candidate        | gliner_v3_10type_candidate |
    |            |                              | + gazetteer overlay        |
    | PROJECT    | candidate/Gate 2             | gliner_v3_10type_candidate |
    |            |                              | + gazetteer overlay        |
    | SOFT_SKILL | candidate/Gate 2             | gliner_v3_10type_candidate |
    +------------+------------------------------+----------------------------+

    Loaded model paths (verified at ``load()`` time):

      - models/saved/ner/                       — v6 DistilBERT
      - models/saved/gliner/                    — v6 GLiNER (used by v6 ensemble)
      - models/saved/ner_v11/                   — v11 ModernBERT (10-type)
      - models/saved/gliner_v3_10type_candidate_20260427_225326/model/
                                                — candidate GLiNER (10-type)

    Graceful degradation: if any of the v11 / candidate-GLiNER weights fail
    to load, the corresponding owner is set to ``None`` and the relevant type
    bucket is filled only by the gazetteer overlay (or empty if no overlay).
    No silent substitution: ``load()`` records which owners actually loaded
    on the instance, so the dashboard report can be honest about coverage.

    SKILL / TOOL duplication policy (intentional under per_class_router §1):
    when v6 emits "AutoCAD" as SKILL and v11 emits "AutoCAD" as TOOL, both
    are preserved. The router's overlap dedup is per-type only — it does
    not cross-type-displace. This matches the "preserve legacy SKILL
    identity, add TOOL overlays" framing. To switch to "hard cross-type
    ownership" (TOOL displaces SKILL on the same surface form), a
    post-processing pass would be required; that is NOT implemented here
    and would be a deliberate, separate decision.
    """

    def __init__(self, device=None):
        self._ensemble = EnsembleNEREngine(device=device)
        self._v11 = _V11ModernBertExtractor()
        self._candidate_gliner = _CandidateGLiNERExtractor()
        self._gazetteer = None
        # Per-owner load status for the audit / report
        self.loaded_owners = {
            "v6_ensemble":          False,
            "v11_modernbert_TOOL":  False,
            "candidate_gliner_GATE2": False,
            "gazetteer":            False,
        }
        self._loaded = False

    def load(self, ner_path=None):
        # 1) v6 EnsembleNEREngine — DistilBERT (legacy 5) + GLiNER (v7 labels).
        try:
            self._ensemble.load(ner_path)
            self.loaded_owners["v6_ensemble"] = True
        except Exception as e:
            print(f"[RoutedNEREngine] v6 ensemble load failed: {e!r} — "
                  f"falling back to GLiNER-only inside the ensemble.")
            try:
                self._ensemble.gliner.load()
                self._ensemble._loaded = True
                self.loaded_owners["v6_ensemble"] = True  # gliner-only still counts
            except Exception as e2:
                print(f"[RoutedNEREngine] v6 GLiNER fallback also failed: {e2!r}")

        # 2) v11 ModernBERT — TOOL owner.
        try:
            self._v11.load()
            self.loaded_owners["v11_modernbert_TOOL"] = True
        except Exception as e:
            print(f"[RoutedNEREngine] v11 ModernBERT load failed: {e!r} — "
                  f"TOOL falls back to gazetteer overlay only.")
            self._v11 = None

        # 3) Candidate 10-type GLiNER — Gate 2 owner.
        try:
            self._candidate_gliner.load()
            self.loaded_owners["candidate_gliner_GATE2"] = True
        except Exception as e:
            print(f"[RoutedNEREngine] candidate 10-type GLiNER load failed: "
                  f"{e!r} — INDUSTRY/LOCATION/PROJECT/SOFT_SKILL fall back "
                  f"to gazetteer overlay only.")
            self._candidate_gliner = None

        # 4) Gazetteer (always loads; CPU-only regex matcher).
        try:
            from models.gazetteer_matcher import build_default_matcher
            self._gazetteer = build_default_matcher()
            self.loaded_owners["gazetteer"] = True
        except Exception as e:
            print(f"[RoutedNEREngine] gazetteer load failed: {e!r}")
            self._gazetteer = None

        self._loaded = True
        print("RoutedNEREngine loaded:", self.loaded_owners)
        return self

    # ------- adapters -----------------------------------------------------
    @staticmethod
    def _ensemble_spans_for_types(ents_dict, *, text, types, source_tag):
        """Translate an EnsembleNEREngine dict slice into router Spans.
        Offsets are recovered by case-insensitive string-search on ``text``.
        """
        from models.per_class_router import Span
        out = []
        for etype in types:
            for txt in ents_dict.get(etype, []):
                if not txt:
                    continue
                pos = text.find(txt)
                if pos < 0:
                    pos = text.lower().find(txt.lower())
                if pos < 0:
                    start, end = 0, 0
                else:
                    start, end = pos, pos + len(txt)
                out.append(Span(text=txt, type=etype, start=start, end=end,
                                score=0.90, source=source_tag))
        return out

    # ------- public API ---------------------------------------------------
    def extract_entities(self, text):
        """Route the four owners + gazetteer through ``per_class_router.route``.

        Returns ``Dict[str, List[str]]`` keyed by every type in
        ``per_class_router.ALL_TYPES``. Empty types appear with ``[]``.
        """
        if not self._loaded:
            raise RuntimeError("RoutedNEREngine not loaded. Call load() first.")
        from models.per_class_router import (
            RoutingConfig, route, GATE2_TYPES, LEGACY5_TYPES, V11_OWNED, ALL_TYPES,
        )

        # Cache each model's output for this text — extractor closures reuse
        # the cached result per call so no extractor runs twice.
        ensemble_ents = self._ensemble.extract_entities(text)
        v11_spans = (self._v11.predict_spans(text)
                     if self._v11 is not None else [])
        cand_spans = (self._candidate_gliner.predict_spans(text)
                      if self._candidate_gliner is not None else [])

        def _legacy_extractor(_t):
            # v6 ensemble owns legacy 5 — filter to those types only.
            return self._ensemble_spans_for_types(
                ensemble_ents, text=text, types=LEGACY5_TYPES,
                source_tag="v6_ensemble",
            )

        def _tool_extractor(_t):
            # v11 ModernBERT owns TOOL.
            return [s for s in v11_spans if s.type in V11_OWNED]

        def _gate2_factory(target):
            def _fn(_t, _target=target):
                # candidate 10-type GLiNER owns each Gate-2 type.
                return [s for s in cand_spans if s.type == _target]
            return _fn

        def _gazetteer_extractor(t):
            if self._gazetteer is None:
                return []
            return list(self._gazetteer.match(t))

        cfg = RoutingConfig(
            v6_extractor=_legacy_extractor,
            v11_tool_extractor=_tool_extractor,
            gate2_winners={c: _gate2_factory(c) for c in GATE2_TYPES},
            gazetteer_matcher=_gazetteer_extractor,
        )
        spans = route(text, cfg)

        # Pre-populate every ALL_TYPES key so the dashboard can iterate
        # without KeyError; deduplicate per type case-insensitively.
        result = {t: [] for t in ALL_TYPES}
        seen = {t: set() for t in ALL_TYPES}
        for s in spans:
            if s.type not in result:
                result[s.type] = []
                seen[s.type] = set()
            key = s.text.strip()
            if not key or key.lower() in seen[s.type]:
                continue
            seen[s.type].add(key.lower())
            result[s.type].append(key)
        return result


def _routed_self_test() -> dict:
    """Mock-based smoke test (fast, no DL weights). Verifies the dict shape
    and per-class fan-out logic. The non-mock real-load smoke test is
    :func:`_routed_real_self_test`."""
    eng = RoutedNEREngine()

    class _MockEnsemble:
        def load(self, ner_path=None):
            return self
        def extract_entities(self, text):
            return {
                "SKILL": ["HAZOP", "P&ID review"],
                "CERT": ["API 510"],
                "DEGREE": ["M.S. Chemical Engineering"],
                "EMPLOYER": ["Chevron"],
                "YEARS_EXP": ["12 years"],
                "TOOL": ["HYSYS", "Python"],   # ignored: TOOL comes from v11
                "INDUSTRY": ["oil and gas"],   # ignored: INDUSTRY from candidate
                "LOCATION": [],
                "PROJECT": [],
                "SOFT_SKILL": [],
            }
        gliner = type("_G", (), {"load": staticmethod(lambda: None)})()
        _loaded = True

    eng._ensemble = _MockEnsemble()
    eng._v11 = None
    eng._candidate_gliner = None
    eng._gazetteer = None
    eng._loaded = True

    sample = (
        "Carlos Mendez — Senior Process Engineer. 12 years at Chevron. "
        "API 510, HAZOP, P&ID review. Tools: HYSYS, Python. "
        "M.S. Chemical Engineering, oil and gas operations."
    )
    out = eng.extract_entities(sample)
    return {
        "uses_mocks": True,
        "n_types_in_output": len(out),
        "types_present": sorted([t for t, v in out.items() if v]),
        "types_empty": sorted([t for t, v in out.items() if not v]),
        "sample_counts": {t: len(v) for t, v in out.items()},
    }


def _routed_real_self_test() -> dict:
    """Real-load smoke test. Loads v6 + v11 + candidate GLiNER + gazetteer
    from disk and runs Carlos's resume through ``extract_entities``. Slow
    (~15-30 s on first call) but exercises the actual production wiring.

    Run via:
        TOKENIZERS_PARALLELISM=false python -c \
        "from models.ner_model import _routed_real_self_test as t; \
         import json; print(json.dumps(t(), indent=2))"

    Or, by default when the module is run directly (the ``__main__`` block
    runs the mock test for speed and points at this function for real).
    """
    eng = RoutedNEREngine()
    eng.load()
    sample = (
        "Carlos Mendez — Senior Process Engineer. "
        "12 years of experience in oil and gas operations. "
        "Currently at Chevron as Lead Process Engineer. "
        "Expertise in P&ID review, HAZOP facilitation, and process safety management. "
        "Led turnaround execution for a 150,000 bpd refinery unit. "
        "Proficient in HYSYS, AutoCAD, and Python for process simulation. "
        "Certifications: PE license, API 570 certified, Six Sigma Black Belt. "
        "Education: M.S. in Chemical Engineering from Texas A&M University. "
        "Previously at Shell for 5 years in upstream operations."
    )
    out = eng.extract_entities(sample)
    return {
        "uses_mocks": False,
        "loaded_owners": eng.loaded_owners,
        "types_present": sorted([t for t, v in out.items() if v]),
        "types_empty": sorted([t for t, v in out.items() if not v]),
        "sample_counts": {t: len(v) for t, v in out.items()},
        "sample_output": out,
    }


if __name__ == "__main__":
    import json as _json
    import os as _os
    # Default: mock smoke test (fast). Pass ROUTED_REAL=1 to run the real
    # load test that actually pulls v11 + candidate GLiNER from disk.
    if _os.getenv("ROUTED_REAL") == "1":
        print(_json.dumps(_routed_real_self_test(), indent=2))
    else:
        print(_json.dumps(_routed_self_test(), indent=2))
