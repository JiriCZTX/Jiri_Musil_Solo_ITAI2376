"""
Agent 2 Deep Learning Model: Bi-LSTM for Workforce Attrition Forecasting.

Uses a Bidirectional LSTM to predict:
  1. Individual attrition probability (binary classification)
  2. Department-level headcount forecast (regression)

Architecture:
  Multi-feature input (8 features) → Bi-LSTM (2 layers, 128 hidden)
  → Dropout → Dual output heads (classification + regression)

Why Bi-LSTM:
  - Forward pass: captures how past events lead to attrition
  - Backward pass: incorporates knowledge of planned future events
    (e.g., scheduled plant shutdown, bonus payout dates)
  - LSTM gates (input, forget, output) solve the vanishing gradient
    problem that RNNs suffer from — critical for temporal patterns
    spanning 12+ months (retirement cohorts, seasonal cycles)
  - Lab 05 results: LSTM (79.45%) beat Vanilla RNN (70.30%) by 10 points
    on AG News, demonstrating the value of memory gates

Course Connection: Module 03 - RNNs/LSTMs (sequence modeling, vanishing gradients)
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, LSTM_BIDIRECTIONAL, LSTM_SEQ_LENGTH,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE, MODELS_DIR,
)


class BiLSTMAttrition(nn.Module):
    """
    Bidirectional LSTM with dual output heads for workforce forecasting.

    The three LSTM gates control information flow:
    - Input gate:  decides what new information to store
    - Forget gate: decides what to discard from cell state
    - Output gate: decides what to expose as hidden state

    Bidirectional processing captures both:
    - Forward: historical patterns leading to attrition
    - Backward: future context (scheduled events, known exits)
    """

    def __init__(self, input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT,
                 bidirectional=LSTM_BIDIRECTIONAL):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Bi-LSTM layers with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions

        # Attention mechanism for temporal weighting
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Head 1: Binary classification (will this person leave in 6 months?)
        # Emits raw logits — paired with BCEWithLogitsLoss at training time
        # (numerically stable) and torch.sigmoid at inference time. The
        # earlier nn.Sigmoid() here collided with BCEWithLogitsLoss's built-in
        # sigmoid, which double-squashed outputs and caused bimodal saturation.
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 1),
        )

        # Head 2: Regression (predicted headcount delta)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """
        Forward pass through the Bi-LSTM.

        Args:
            x: (batch, seq_len, input_size) tensor of monthly features

        Returns:
            attrition_logit: (batch, 1) raw logit (NOT a probability).
                             Callers must apply torch.sigmoid at inference.
                             Training uses BCEWithLogitsLoss which handles
                             the sigmoid internally.
            headcount_delta: (batch, 1) predicted headcount change
        """
        # LSTM forward pass: processes sequence in both directions
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden * num_directions)

        # Attention-weighted context vector
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Dual heads — classification returns raw logits
        attrition_logit = self.classification_head(context)
        headcount_delta = self.regression_head(context)

        return attrition_logit, headcount_delta


class WorkforceSequenceDataset(Dataset):
    """Dataset for creating sequential monthly windows from workforce data."""

    def __init__(self, features, attrition_labels, headcount_deltas, seq_length=LSTM_SEQ_LENGTH):
        self.seq_length = seq_length
        self.features = features
        self.attrition_labels = attrition_labels
        self.headcount_deltas = headcount_deltas

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        # Target is the label at the end of the sequence
        y_class = self.attrition_labels[idx + self.seq_length - 1]
        y_reg = self.headcount_deltas[idx + self.seq_length - 1]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor([y_class]),
            torch.FloatTensor([y_reg]),
        )


class ForecastingEngine:
    """Train and run the Bi-LSTM workforce forecasting model."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            "tenure_years", "comp_ratio", "satisfaction", "performance",
            "engagement", "age", "bls_energy_index", "headcount"
        ]

    def build_model(self):
        """Initialize the Bi-LSTM model."""
        self.model = BiLSTMAttrition(
            input_size=len(self.feature_columns),
        ).to(self.device)
        return self.model

    def prepare_department_data(self, individual_df, monthly_df, department):
        """
        Prepare sequential data for a specific department.
        Merges individual-level (per-employee monthly records) and
        department-level monthly aggregates into the 8-feature sequence
        the Bi-LSTM consumes.

        Performance and age are aggregated from `individual_df` on the
        fly — the Bi-LSTM was previously fed hardcoded constants for
        these two features, which silenced ~25% of its input signal.
        """
        # Department-level monthly data
        dept_monthly = monthly_df[monthly_df["department"] == department].sort_values("month")

        if len(dept_monthly) < LSTM_SEQ_LENGTH + 1:
            return None, None, None

        # Use department-aggregated features
        feature_data = dept_monthly[["avg_tenure", "avg_comp_ratio", "avg_satisfaction"]].copy()
        feature_data.columns = ["tenure_years", "comp_ratio", "satisfaction"]

        # Per-month, per-department averages of performance and age,
        # drawn from the individual monthly records when available. This
        # unhardcodes the two placeholder constants that used to be
        # baked into the input stream.
        perf_series = None
        age_series = None
        if individual_df is not None and len(individual_df) > 0:
            dept_ind = individual_df[individual_df["department"] == department]
            if not dept_ind.empty:
                monthly_perf = (dept_ind.groupby("month")["performance"]
                                .mean().round(3))
                monthly_age = (dept_ind.groupby("month")["age"]
                               .mean().round(2))
                # Align to dept_monthly['month'] order
                perf_series = dept_monthly["month"].map(monthly_perf)
                age_series = dept_monthly["month"].map(monthly_age)
                # Fallback for any month without per-employee records
                perf_series = perf_series.fillna(monthly_perf.mean()
                                                 if len(monthly_perf) else 3.2)
                age_series = age_series.fillna(monthly_age.mean()
                                                if len(monthly_age) else 40.0)
        if perf_series is None:
            perf_series = pd.Series([3.2] * len(dept_monthly),
                                    index=dept_monthly.index)
        if age_series is None:
            age_series = pd.Series([40.0] * len(dept_monthly),
                                   index=dept_monthly.index)

        feature_data["performance"] = perf_series.values
        feature_data["engagement"] = dept_monthly["avg_engagement"].values
        feature_data["age"] = age_series.values
        feature_data["bls_energy_index"] = dept_monthly["bls_energy_index"].values
        feature_data["headcount"] = dept_monthly["headcount"].values

        # Compute attrition rate as binary (above/below threshold)
        dept_monthly_attrition = dept_monthly["departures"] / dept_monthly["headcount"].clip(lower=1)
        attrition_labels = (dept_monthly_attrition > dept_monthly_attrition.median()).astype(float).values

        # Headcount delta for regression
        headcount_deltas = dept_monthly["headcount"].diff().fillna(0).values

        # Scale features
        features_scaled = self.scaler.fit_transform(feature_data.values)

        return features_scaled, attrition_labels, headcount_deltas

    def train(self, individual_df, monthly_df, epochs=LSTM_EPOCHS,
              val_fraction=0.15, early_stop_patience=10, verbose=True):
        """
        Train the Bi-LSTM on all departments' temporal data.

        Upgrades over v1 trainer:
          - Per-department stratified train/val split (tail-N months held out
            per department so val sequences are temporally disjoint from
            training).
          - Best-epoch checkpointing by val AUC-ROC (falls back to val loss
            when AUC is degenerate).
          - Early stopping when val AUC stops improving for `patience` epochs.
          - pos_weight=1.0: the label is a dept-level above/below-median
            split, which is ~50/50 by construction, so class weighting is
            unnecessary. v1 used 5.0 which biased predictions to positive.
          - BCEWithLogitsLoss consumes raw logits from the (now-sigmoid-less)
            classification head.
          - Reports calibration (Brier score) alongside discrimination (AUC).
        """
        if self.model is None:
            self.build_model()

        # Build train/val sequences, holding out the last `val_fraction`
        # months of each department as validation. This keeps training and
        # validation temporally disjoint and preserves the per-dept balance.
        train_feats, train_cls, train_reg = [], [], []
        val_feats, val_cls, val_reg = [], [], []

        for dept in monthly_df["department"].unique():
            features, attrition, headcount = self.prepare_department_data(
                individual_df, monthly_df, dept
            )
            if features is None:
                continue
            n = len(features)
            n_val = max(LSTM_SEQ_LENGTH + 1, int(n * val_fraction))
            if n - n_val < LSTM_SEQ_LENGTH + 1:
                # Not enough for a clean split — use all for training.
                train_feats.append(features)
                train_cls.append(attrition)
                train_reg.append(headcount)
                continue
            train_feats.append(features[:n - n_val])
            train_cls.append(attrition[:n - n_val])
            train_reg.append(headcount[:n - n_val])
            val_feats.append(features[n - n_val - LSTM_SEQ_LENGTH:])
            val_cls.append(attrition[n - n_val - LSTM_SEQ_LENGTH:])
            val_reg.append(headcount[n - n_val - LSTM_SEQ_LENGTH:])

        train_features = np.concatenate(train_feats)
        train_attrition = np.concatenate(train_cls)
        train_headcount = np.concatenate(train_reg)
        train_ds = WorkforceSequenceDataset(
            train_features, train_attrition, train_headcount
        )
        train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE,
                                  shuffle=True)

        have_val = len(val_feats) > 0
        if have_val:
            val_features = np.concatenate(val_feats)
            val_attrition = np.concatenate(val_cls)
            val_headcount = np.concatenate(val_reg)
            val_ds = WorkforceSequenceDataset(
                val_features, val_attrition, val_headcount
            )
            val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE,
                                    shuffle=False)

        # pos_weight=1.0: the label is median-balanced within each dept, so
        # the dataset-wide positive ratio is ~0.5 — no weighting needed.
        pos_weight = torch.tensor([1.0]).to(self.device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse_loss = nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=LSTM_LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        history = {"loss": [], "cls_loss": [], "reg_loss": [],
                   "val_loss": [], "val_auc": [], "val_f1": [],
                   "val_brier": []}
        best_auc = -1.0
        best_loss = float("inf")
        best_state = None
        best_epoch = -1
        no_improve = 0

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            total_loss = total_cls = total_reg = 0.0
            for x_batch, y_cls, y_reg in train_loader:
                x_batch = x_batch.to(self.device)
                y_cls = y_cls.to(self.device)
                y_reg = y_reg.to(self.device)

                logit, headcount_delta = self.model(x_batch)
                cls_loss_val = bce_loss(logit, y_cls)
                reg_loss_val = mse_loss(headcount_delta, y_reg)
                loss = cls_loss_val + 0.1 * reg_loss_val

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_cls += cls_loss_val.item()
                total_reg += reg_loss_val.item()

            avg_train = total_loss / max(len(train_loader), 1)
            history["loss"].append(avg_train)
            history["cls_loss"].append(total_cls / max(len(train_loader), 1))
            history["reg_loss"].append(total_reg / max(len(train_loader), 1))

            # --- Validate ---
            val_auc = val_f1 = val_brier = None
            avg_val = None
            if have_val:
                self.model.eval()
                val_loss_sum = 0.0
                val_probs, val_labels = [], []
                with torch.no_grad():
                    for x_batch, y_cls, y_reg in val_loader:
                        x_batch = x_batch.to(self.device)
                        y_cls = y_cls.to(self.device)
                        y_reg = y_reg.to(self.device)
                        logit, hd = self.model(x_batch)
                        vcls = bce_loss(logit, y_cls)
                        vreg = mse_loss(hd, y_reg)
                        val_loss_sum += (vcls + 0.1 * vreg).item()
                        probs = torch.sigmoid(logit).cpu().numpy().flatten()
                        val_probs.extend(probs)
                        val_labels.extend(y_cls.cpu().numpy().flatten())
                avg_val = val_loss_sum / max(len(val_loader), 1)
                if len(set(val_labels)) > 1:
                    val_auc = float(roc_auc_score(val_labels, val_probs))
                    binary = [1 if p > 0.5 else 0 for p in val_probs]
                    val_f1 = float(f1_score(val_labels, binary,
                                            zero_division=0))
                val_brier = float(np.mean(
                    (np.array(val_probs) - np.array(val_labels)) ** 2
                ))
            history["val_loss"].append(avg_val)
            history["val_auc"].append(val_auc)
            history["val_f1"].append(val_f1)
            history["val_brier"].append(val_brier)

            # --- LR scheduler steps on val loss (or train loss if no val) ---
            scheduler.step(avg_val if have_val else avg_train)

            # --- Best-epoch checkpoint ---
            improved = False
            if val_auc is not None and val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_loss = avg_val if avg_val is not None else avg_train
                best_state = {k: v.clone().cpu()
                              for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve = 0
                improved = True
            elif val_auc is None and (
                avg_val is not None and avg_val < best_loss - 1e-4
            ):
                best_loss = avg_val
                best_state = {k: v.clone().cpu()
                              for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve = 0
                improved = True
            else:
                no_improve += 1

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                marker = "  *best*" if improved else ""
                vstr = (f" | val_loss: {avg_val:.4f}  val_auc: {val_auc:.4f}"
                         f"  val_f1: {val_f1:.4f}  brier: {val_brier:.4f}"
                         if val_auc is not None else "")
                print(f"  Epoch {epoch+1:2d}/{epochs} - "
                      f"train_loss: {avg_train:.4f}{vstr}{marker}")

            # --- Early stopping ---
            if have_val and no_improve >= early_stop_patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch+1} "
                          f"(no val-AUC improvement for "
                          f"{early_stop_patience} epochs).")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"  Loaded best weights from epoch {best_epoch} "
                      f"(val_auc={best_auc:.4f}).")

        history["best_epoch"] = best_epoch
        history["best_val_auc"] = best_auc if best_auc > -1 else None
        return history

    def predict_department(self, monthly_df, department):
        """
        Predict attrition risk and headcount forecast for a department.

        Returns dict with risk level, probability, and projected headcount.
        """
        self.model.eval()

        features, _, _ = self.prepare_department_data(
            pd.DataFrame(), monthly_df, department
        )
        if features is None or len(features) < LSTM_SEQ_LENGTH:
            return {"error": f"Insufficient data for {department}"}

        # Use last sequence window
        x = torch.FloatTensor(features[-LSTM_SEQ_LENGTH:]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            attrition_logit, headcount_delta = self.model(x)
            attrition_prob = torch.sigmoid(attrition_logit)

        prob = attrition_prob.item()
        delta = headcount_delta.item()
        current_hc = monthly_df[monthly_df["department"] == department]["headcount"].iloc[-1]

        return {
            "department": department,
            "attrition_probability": round(prob, 4),
            "risk_level": self._prob_to_risk(prob),
            "current_headcount": int(current_hc),
            "projected_headcount_3m": int(current_hc + delta * 3),
            "projected_headcount_6m": int(current_hc + delta * 6),
            "projected_headcount_12m": int(current_hc + delta * 12),
            "monthly_delta": round(delta, 1),
        }

    def predict_all_departments(self, monthly_df):
        """Predict attrition risk for all departments."""
        results = []
        for dept in monthly_df["department"].unique():
            pred = self.predict_department(monthly_df, dept)
            if "error" not in pred:
                results.append(pred)
        results.sort(key=lambda x: x["attrition_probability"], reverse=True)
        return results

    def evaluate(self, monthly_df, individual_df):
        """Evaluate model with AUC-ROC and MAE metrics."""
        self.model.eval()
        all_probs, all_labels, all_pred_hc, all_true_hc = [], [], [], []

        for dept in monthly_df["department"].unique():
            features, attrition, headcount = self.prepare_department_data(
                individual_df, monthly_df, dept
            )
            if features is None:
                continue

            dataset = WorkforceSequenceDataset(features, attrition, headcount)
            loader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE)

            with torch.no_grad():
                for x_batch, y_cls, y_reg in loader:
                    x_batch = x_batch.to(self.device)
                    logit, delta = self.model(x_batch)
                    prob = torch.sigmoid(logit)
                    all_probs.extend(prob.cpu().numpy().flatten())
                    all_labels.extend(y_cls.numpy().flatten())
                    all_pred_hc.extend(delta.cpu().numpy().flatten())
                    all_true_hc.extend(y_reg.numpy().flatten())

        if not all_labels:
            return {"error": "No evaluation data"}

        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
        mae = mean_absolute_error(all_true_hc, all_pred_hc)
        binary_preds = [1 if p > 0.5 else 0 for p in all_probs]
        f1 = f1_score(all_labels, binary_preds, zero_division=0)

        return {
            "auc_roc": round(auc, 4),
            "mae_headcount": round(mae, 2),
            "f1_score": round(f1, 4),
            "n_samples": len(all_labels),
        }

    @staticmethod
    def _prob_to_risk(prob):
        """Convert probability to risk level string."""
        if prob >= 0.70:
            return "CRITICAL"
        elif prob >= 0.50:
            return "HIGH"
        elif prob >= 0.30:
            return "MEDIUM"
        else:
            return "LOW"

    def save(self, path=None):
        """Save model weights and scaler."""
        path = Path(path or MODELS_DIR / "bilstm")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "bilstm_weights.pth")
        print(f"Bi-LSTM model saved to {path}")

    @staticmethod
    def _remap_colab_keys(state_dict):
        """
        Remap Colab v2 weight keys to match local model layer names.

        Colab used abbreviated names and slightly different Sequential structure:
          Colab: attn, cls, reg  (cls/reg have 5 layers: Drop,Linear,ReLU,Linear,Sigmoid)
          Local: attention, classification_head, regression_head (6 layers: +extra Dropout)

        Key index mapping for cls/reg:
          Colab cls.1 (Linear) → local classification_head.1
          Colab cls.3 (Linear) → local classification_head.4  (shifted by extra Dropout at idx 3)
          Same pattern for reg → regression_head
        """
        KEY_MAP = {
            # Attention: same structure, just rename
            "attn.": "attention.",
            # Classification head: rename + reindex
            "cls.0.": "classification_head.0.",   # Dropout (no weights, but just in case)
            "cls.1.": "classification_head.1.",   # Linear → Linear (same index)
            "cls.3.": "classification_head.4.",   # Linear → Linear (shifted past extra Dropout)
            # Regression head: rename + reindex
            "reg.0.": "regression_head.0.",
            "reg.1.": "regression_head.1.",
            "reg.3.": "regression_head.4.",
        }

        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in KEY_MAP.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            remapped[new_key] = value
        return remapped

    def load(self, path=None):
        """Load saved model weights, auto-remapping Colab v2 keys if needed."""
        path = Path(path or MODELS_DIR / "bilstm")
        if self.model is None:
            self.build_model()

        state_dict = torch.load(path / "bilstm_weights.pth", map_location=self.device)

        # Detect if weights are from Colab v2 (uses 'attn' instead of 'attention')
        if any(k.startswith("attn.") for k in state_dict.keys()):
            print("  Detected Colab v2 weights — remapping keys...")
            state_dict = self._remap_colab_keys(state_dict)

        self.model.load_state_dict(state_dict)
        print(f"Bi-LSTM model loaded from {path}")
        return self
