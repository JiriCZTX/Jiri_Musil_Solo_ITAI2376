"""
CrewAI Custom Tools for Agent 2: Workforce Forecasting Agent.

Wraps the Bi-LSTM forecasting model and workforce CSV data as callable
tools that CrewAI can invoke during its ReAct reasoning loop.

Agent 2's intelligence layer synthesizes the Bi-LSTM's raw attrition
signal into decision-grade outputs:

  1.  Driver attribution (gradient-based feature importance)
  2.  Top 3 risk drivers (ranked by contribution)
  3.  Retirement-cliff cohort (age >= 58 / 55-57)
  4.  New-hire churn cohort (tenure < 2y)
  5.  Low-engagement cohort (satisfaction/engagement < 2.5)
  6.  Comp-gap cohort (comp_ratio < 0.90)
  7.  Trend direction (accelerating / stable / decelerating)
  8.  Seasonality peak month
  9.  Lead indicators (3-month satisfaction + engagement momentum)
 10.  Intervention simulator (counterfactual forward pass)
 11.  Replacement cost estimate ($)
 12.  Vacancy days at risk
 13.  Knowledge loss severity (tenure-weighted)
 14.  Internal mobility bench (cross-department candidates)
 15.  Market competition signal (BLS + dept heat)
 16.  Retention plan (top drivers -> interventions)
 17.  Confidence signal (data + model uncertainty)
 18.  Executive briefing (verdict / drivers / action — 3 paragraphs)

The LLM brain uses these tools to produce workforce decisions grounded
in concrete numbers — not model probabilities dressed up in prose.
"""
import json
import math
import numpy as np
import pandas as pd
from crewai.tools import tool
from dataclasses import asdict
from typing import Optional, Dict, Any, List

from industries import get_industry


# Global model and data instances (initialized in main.py / dashboard.py)
_forecast_engine = None
_monthly_df: Optional[pd.DataFrame] = None
_individual_df: Optional[pd.DataFrame] = None
_employee_df: Optional[pd.DataFrame] = None


def set_forecast_engine(engine):
    global _forecast_engine
    _forecast_engine = engine


def set_workforce_data(monthly, individual, employees=None):
    """
    Register the workforce DataFrames the tools will query.

    Backward-compatible: `employees` is optional. When omitted, cohort
    queries degrade gracefully by synthesizing per-employee features
    from `individual` (the monthly long-format per-person records).
    """
    global _monthly_df, _individual_df, _employee_df
    _monthly_df = monthly
    _individual_df = individual
    _employee_df = employees


# =============================================================================
# Domain constants — sourced from the active industry plugin
# =============================================================================
#
# These constants were hardcoded for the Energy vertical in v5. V6 moved them
# into the industries/ plugin layer so the tool logic is domain-agnostic.
# The 5 accessor functions below read the active industry (via get_industry())
# at call time — meaning a sidebar toggle from Energy → Finance switches all
# tool behavior without touching any tool code.
#
# Each helper returns a plain dict (via dataclasses.asdict) so existing
# call sites using ["key"] access patterns continue to work unchanged.
#
# Byte-identity contract: when the active industry is ENERGY, these helpers
# return EXACTLY the same dicts the former hardcoded constants did. Carlos
# Mendez static-mode verdict (R9-interview-stable / 0.82 / fit 65.8) is
# preserved.

def _dept_economics() -> Dict[str, Dict[str, Any]]:
    """Per-department replacement economics for the active industry."""
    return {k: asdict(v) for k, v in get_industry().dept_economics.items()}


def _interventions() -> Dict[str, Dict[str, Any]]:
    """Retention-lever catalog for the active industry."""
    return {k: asdict(v) for k, v in get_industry().interventions.items()}


def _cohort_thresholds() -> Dict[str, Any]:
    """Cohort segmentation thresholds for the active industry."""
    return asdict(get_industry().cohort_thresholds)


def _dept_adjacency() -> Dict[str, Dict[str, float]]:
    """Cross-department transferable-skill adjacency matrix."""
    return dict(get_industry().dept_adjacency)


def _feature_labels() -> Dict[str, str]:
    """Human-readable feature labels for driver attribution."""
    return dict(get_industry().feature_labels)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# (Former hardcoded _dept_economics() / _interventions() / _cohort_thresholds() /
# _dept_adjacency() / _feature_labels() removed 2026-04-20. Those constants now
# live in `industries/energy.py` and are read via the helpers above. To add
# a new vertical, copy `industries/_template.py` and fill in the fields.)
# -----------------------------------------------------------------------------


# =============================================================================
# Existing 4 tools — kept for backward compatibility
# =============================================================================

@tool("Attrition Risk Predictor")
def predict_attrition_risk(department: str) -> str:
    """
    Predict attrition risk for a department using the Bi-LSTM model.
    Analyzes 12-month temporal patterns to forecast turnover probability
    and projected headcount at 3, 6, and 12 months.
    Input: Department name (e.g., 'Operations', 'Engineering').
    Output: JSON with risk level, probability, and headcount projections.
    """
    if _forecast_engine is None or _monthly_df is None:
        return json.dumps({"error": "Forecasting model not initialized"})
    result = _forecast_engine.predict_department(_monthly_df, department)
    return json.dumps(result, indent=2)


@tool("Full Workforce Forecast")
def predict_all_departments() -> str:
    """
    Run attrition predictions for ALL departments simultaneously.
    Returns a ranked list from highest to lowest risk.
    Input: No input required.
    Output: JSON array of department predictions sorted by risk.
    """
    if _forecast_engine is None or _monthly_df is None:
        return json.dumps({"error": "Forecasting model not initialized"})
    results = _forecast_engine.predict_all_departments(_monthly_df)
    return json.dumps(results, indent=2)


@tool("HRIS Database Query")
def query_hris_data(department: str) -> str:
    """
    Query the HRIS database for current workforce metrics for a department.
    Returns headcount, average tenure, satisfaction, compensation ratio,
    and recent departure counts.
    Input: Department name.
    Output: JSON with current department workforce metrics.
    """
    if _monthly_df is None:
        return json.dumps({"error": "Workforce data not loaded"})
    dept_data = _monthly_df[_monthly_df["department"] == department]
    if dept_data.empty:
        return json.dumps({"error": f"Department '{department}' not found"})
    latest = dept_data.sort_values("month").iloc[-1]
    last_3 = dept_data.sort_values("month").tail(3)
    return json.dumps({
        "department": department,
        "current_headcount": int(latest["headcount"]),
        "avg_tenure_years": float(latest["avg_tenure"]),
        "avg_satisfaction": float(latest["avg_satisfaction"]),
        "avg_comp_ratio": float(latest["avg_comp_ratio"]),
        "avg_engagement": float(latest["avg_engagement"]),
        "departures_last_3_months": int(last_3["departures"].sum()),
        "headcount_trend_3m": int(last_3["headcount"].iloc[-1]
                                  - last_3["headcount"].iloc[0]),
    }, indent=2)


@tool("BLS Labor Market Data")
def query_bls_data(sector: str = "energy") -> str:
    """
    Query Bureau of Labor Statistics data for energy sector employment trends.
    Returns the latest energy employment index and recent trend.
    Input: Sector name (default: 'energy').
    Output: JSON with BLS employment index data and trend analysis.

    DEVIATION NOTE: Using static simulated BLS data instead of live API.
    Production version would use BLS QCEW API.
    """
    if _monthly_df is None:
        return json.dumps({"error": "Data not loaded"})
    bls_data = _monthly_df.groupby("month")["bls_energy_index"].first().reset_index()
    latest = bls_data.iloc[-1]
    prev_q = bls_data.iloc[-3] if len(bls_data) >= 3 else bls_data.iloc[0]
    trend = latest["bls_energy_index"] - prev_q["bls_energy_index"]
    direction = "growing" if trend > 0 else "declining" if trend < 0 else "stable"
    return json.dumps({
        "sector": sector,
        "current_index": round(float(latest["bls_energy_index"]), 2),
        "previous_quarter_index": round(float(prev_q["bls_energy_index"]), 2),
        "quarterly_change": round(float(trend), 2),
        "trend_direction": direction,
        "interpretation": (
            f"Energy sector employment is {direction}. "
            f"Index moved from {prev_q['bls_energy_index']:.1f} to "
            f"{latest['bls_energy_index']:.1f} over the last quarter. "
            f"{'This may increase competition for talent.' if trend > 0 else 'Hiring pressure may ease slightly.'}"
        ),
        "data_source": "Simulated BLS QCEW (production would use live API)",
    }, indent=2)


# =============================================================================
# Helper: latest per-employee snapshot for a department
# =============================================================================

def _latest_employee_snapshot(department: str) -> Optional[pd.DataFrame]:
    """
    Return per-employee records for the most recent month in the dataset,
    filtered to `department`. Uses `_individual_df` (per-person monthly
    records) and picks each employee's latest row. This is the canonical
    source for cohort segmentation and driver attribution.
    """
    if _individual_df is None:
        return None
    dept_ind = _individual_df[_individual_df["department"] == department]
    if dept_ind.empty:
        return None
    # Grab each employee's latest record — employees leave at different months
    latest_rows = dept_ind.sort_values("month").groupby("employee_id").tail(1)
    # Filter out rows from employees who already departed — keep only those
    # still active as of the dataset's max month (within 1 month tolerance
    # to handle end-of-data employees).
    max_month = int(_individual_df["month"].max())
    still_active = latest_rows[latest_rows["month"] >= max_month - 1]
    return still_active if not still_active.empty else latest_rows


# =============================================================================
# Signal 1-2 — Driver attribution via Bi-LSTM gradients
# =============================================================================

def _driver_attribution(department: str) -> dict:
    """
    Attribute the Bi-LSTM's attrition probability to each of its 8 input
    features using an input-gradient approximation of SHAP:

        importance_i = |x_i * d(prob) / d(x_i)| averaged across the
                       12-month lookback window.

    This is a fast, well-known saliency method (Sundararajan et al.
    "SmoothGrad" / Simonyan "Gradient × Input"). It's directionally
    correct for telling the agent *which* features are driving risk,
    which is what matters for retention planning.

    Returns
    -------
    dict with:
      per_feature_importance: {feature_name: relative_%_contribution}
      top_drivers: ordered list of (feature_name, %_contribution)
      raw_prob: float (model's attrition probability for this dept)
    """
    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine / data not ready"}

    import torch  # local import — only needed when gradients are computed

    features, _, _ = _forecast_engine.prepare_department_data(
        _individual_df if _individual_df is not None else pd.DataFrame(),
        _monthly_df,
        department,
    )
    if features is None:
        return {"error": f"Insufficient data for {department}"}

    from config.settings import LSTM_SEQ_LENGTH
    window = features[-LSTM_SEQ_LENGTH:]

    # Build a gradient-enabled input tensor
    x = torch.tensor(window, dtype=torch.float32,
                     device=_forecast_engine.device).unsqueeze(0)
    x.requires_grad_(True)

    _forecast_engine.model.eval()
    # Zero any lingering grads on the model
    for p in _forecast_engine.model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    # Model now emits raw logits — apply sigmoid so we're attributing the
    # *probability*, not the logit. Gradient direction is preserved; only
    # the magnitude normalization across features changes.
    logit, _ = _forecast_engine.model(x)
    prob_scalar = torch.sigmoid(logit).squeeze()
    prob_scalar.backward()

    grad = x.grad.detach().cpu().numpy()[0]   # (seq_len, n_features)
    inp = x.detach().cpu().numpy()[0]

    # Gradient × input, absolute, averaged across the time axis
    saliency = np.abs(grad * inp).mean(axis=0)
    total = float(saliency.sum())

    feats = _forecast_engine.feature_columns
    per_feature = {}
    for i, name in enumerate(feats):
        share = float(saliency[i] / total) if total > 0 else 0.0
        per_feature[name] = round(share, 4)

    ordered = sorted(per_feature.items(), key=lambda kv: -kv[1])
    top_drivers = [
        {
            "feature": name,
            "label": _feature_labels().get(name, name),
            "pct_of_risk": round(share * 100, 1),
        }
        for name, share in ordered
    ]

    return {
        "raw_attrition_probability": round(float(prob_scalar.item()), 4),
        "per_feature_importance": per_feature,
        "top_drivers": top_drivers,
    }


# =============================================================================
# Signal 3-6 — Cohort segmentation
# =============================================================================

def _cohort_segmentation(department: str) -> dict:
    """
    Break the active workforce of `department` into decision-relevant
    cohorts. Each cohort is a count + % of dept, with the employee-IDs
    preserved in a list so downstream tools (intervention cost, mobility
    bench) can act on the specific people.
    """
    snap = _latest_employee_snapshot(department)
    if snap is None or snap.empty:
        return {"error": f"No employee data for {department}"}

    T = _cohort_thresholds()
    n = len(snap)

    retire_critical = snap[snap["age"] >= T["retirement_critical_age"]]
    retire_emerging = snap[
        (snap["age"] >= T["retirement_emerging_age"])
        & (snap["age"] < T["retirement_critical_age"])
    ]
    new_hire_churn = snap[snap["tenure_years"] < T["new_hire_tenure_years"]]
    low_engagement = snap[
        (snap["engagement"] < T["low_engagement_score"])
        | (snap["satisfaction"] < T["low_satisfaction_score"])
    ]
    comp_gap = snap[snap["comp_ratio"] < T["comp_gap_ratio"]]
    tenured_ip = snap[snap["tenure_years"] >= T["tenured_ip_years"]]
    high_perf = snap[snap["performance"] >= T["high_performer_score"]]

    def _pack(df, label):
        ids = df["employee_id"].tolist() if "employee_id" in df.columns else []
        return {
            "label": label,
            "count": int(len(df)),
            "pct_of_department": round(len(df) / n * 100, 1) if n else 0.0,
            "employee_ids": ids[:50],  # cap for JSON size
        }

    return {
        "department": department,
        "department_size": n,
        "retirement_cliff_critical": _pack(retire_critical, "Age ≥ 58"),
        "retirement_cliff_emerging": _pack(retire_emerging, "Age 55-57"),
        "new_hire_churn_window":     _pack(new_hire_churn, "Tenure < 2 years"),
        "low_engagement":            _pack(low_engagement,
                                            "Engagement or satisfaction < 2.5"),
        "comp_gap":                  _pack(comp_gap, "Comp ratio < 0.90"),
        "tenured_ip":                _pack(tenured_ip, "Tenure ≥ 8 years"),
        "high_performers":           _pack(high_perf, "Performance ≥ 4.0"),
        "high_performer_at_risk": int(len(
            high_perf[high_perf["employee_id"].isin(
                set(low_engagement["employee_id"]).union(
                    set(comp_gap["employee_id"]))
            )]
        )),
    }


# =============================================================================
# Signal 7-9 — Temporal intelligence
# =============================================================================

def _trend_analysis(department: str) -> dict:
    """
    Direction + velocity of attrition over the last 12 months, split into
    recent-6 vs prior-6 windows. Classifies as accelerating / stable /
    decelerating so the agent can tell whether a high-risk reading is
    getting worse or already past its peak.
    """
    if _monthly_df is None:
        return {"error": "Monthly data not loaded"}
    dd = _monthly_df[_monthly_df["department"] == department].sort_values("month")
    if len(dd) < 12:
        return {"status": "insufficient_history",
                "months_available": int(len(dd))}

    tail = dd.tail(12)
    recent = tail.tail(6)
    prior = tail.head(6)
    rate = lambda r: float(r["departures"].sum()) / float(
        max(1, r["headcount"].mean()))
    r_now = rate(recent)
    r_prior = rate(prior)
    delta = r_now - r_prior

    if delta > 0.005:
        direction = "accelerating"
    elif delta < -0.005:
        direction = "decelerating"
    else:
        direction = "stable"

    return {
        "department": department,
        "trend_direction": direction,
        "recent_6mo_attrition_rate": round(r_now, 4),
        "prior_6mo_attrition_rate": round(r_prior, 4),
        "delta": round(delta, 4),
        "interpretation": (
            f"{department} attrition is {direction} "
            f"({r_prior:.1%} → {r_now:.1%})."
        ),
    }


def _seasonality(department: str) -> dict:
    """
    Identify the peak attrition month-of-year by aggregating departures
    across all years in the dataset. Picks up the January-after-bonus
    pattern and any summer/project-cycle effects.
    """
    if _monthly_df is None:
        return {"error": "Monthly data not loaded"}
    dd = _monthly_df[_monthly_df["department"] == department]
    if dd.empty:
        return {"error": f"No data for {department}"}

    dd = dd.copy()
    dd["month_of_year"] = dd["month"].astype(int) % 12
    moy = dd.groupby("month_of_year")["departures"].mean().round(2)
    peak_m = int(moy.idxmax())
    peak_val = float(moy.max())
    trough_m = int(moy.idxmin())
    amplitude = peak_val - float(moy.min())

    m_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Aug","Sep","Oct","Nov","Dec"]

    return {
        "department": department,
        "peak_month_index": peak_m,
        "peak_month_name": m_names[peak_m] if 0 <= peak_m < 12 else "?",
        "peak_avg_departures": round(peak_val, 2),
        "trough_month_index": trough_m,
        "amplitude": round(amplitude, 2),
        "has_seasonal_pattern": bool(amplitude >= 1.0),
    }


def _lead_indicators(department: str) -> dict:
    """
    Three-month momentum in satisfaction and engagement. Declining
    leading indicators = early warning that attrition rises in 3-6 months.
    """
    if _monthly_df is None:
        return {"error": "Monthly data not loaded"}
    dd = _monthly_df[_monthly_df["department"] == department].sort_values("month")
    if len(dd) < 6:
        return {"status": "insufficient_history"}

    last3 = dd.tail(3)
    prev3 = dd.tail(6).head(3)
    sat_delta = float(last3["avg_satisfaction"].mean()
                      - prev3["avg_satisfaction"].mean())
    eng_delta = float(last3["avg_engagement"].mean()
                      - prev3["avg_engagement"].mean())

    flags = []
    if sat_delta <= -0.15:
        flags.append("satisfaction_declining")
    if eng_delta <= -0.15:
        flags.append("engagement_declining")
    if not flags and (sat_delta >= 0.10 or eng_delta >= 0.10):
        flags.append("morale_improving")

    return {
        "department": department,
        "satisfaction_3mo_momentum": round(sat_delta, 3),
        "engagement_3mo_momentum": round(eng_delta, 3),
        "flags": flags,
        "early_warning": bool(
            "satisfaction_declining" in flags
            or "engagement_declining" in flags),
    }


# =============================================================================
# Signal 10 — Intervention simulator (counterfactual forward pass)
# =============================================================================

def _simulate_intervention_effect(department: str,
                                  intervention: str,
                                  magnitude: float) -> dict:
    """
    Run the Bi-LSTM forward TWICE — once on the observed 12-month window
    and once on a counterfactual window where the intervention has lifted
    its target feature by `magnitude`. Returns the delta in attrition
    probability, plus a rough cost estimate for the intervention.

    Parameters
    ----------
    department  : name
    intervention: key in `_interventions()` (e.g., 'comp_adjustment')
    magnitude   : how much to move the feature, in standardized units.
                  E.g., comp_adjustment magnitude=0.10 => +0.10 comp_ratio.
    """
    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine not initialized"}
    if intervention not in _interventions():
        return {"error": f"Unknown intervention '{intervention}'. "
                          f"Available: {list(_interventions().keys())}"}

    import torch

    cfg = _interventions()[intervention]
    feat_name = cfg["feature_affected"]
    if feat_name is None:
        # e.g., knowledge_capture doesn't move probability
        return {
            "department": department,
            "intervention": intervention,
            "description": cfg["description"],
            "feature_moved": None,
            "baseline_prob": None,
            "counterfactual_prob": None,
            "delta_prob": 0.0,
            "causal_status": cfg.get("causal_status", "NONE"),
            "causal_rationale": cfg.get("causal_rationale", ""),
            "note": "This intervention targets knowledge loss, not attrition rate.",
        }
    if feat_name not in _forecast_engine.feature_columns:
        return {"error": f"Feature '{feat_name}' not in Bi-LSTM inputs."}

    features, _, _ = _forecast_engine.prepare_department_data(
        _individual_df if _individual_df is not None else pd.DataFrame(),
        _monthly_df, department,
    )
    if features is None:
        return {"error": f"Insufficient data for {department}"}

    from config.settings import LSTM_SEQ_LENGTH
    window = features[-LSTM_SEQ_LENGTH:].copy()
    feat_idx = _forecast_engine.feature_columns.index(feat_name)

    # The features are already standard-scaled by ForecastingEngine.
    # The scaler's scale_ attribute tells us how many scaled-units equal
    # `magnitude` raw units. If unavailable (old weights), fall back to a
    # direct bump on the scaled value — directionally still correct.
    scale_ = getattr(_forecast_engine.scaler, "scale_", None)
    if scale_ is not None and feat_idx < len(scale_):
        scaled_bump = float(magnitude) / float(scale_[feat_idx])
    else:
        scaled_bump = float(magnitude)

    cf_window = window.copy()
    cf_window[:, feat_idx] = cf_window[:, feat_idx] + scaled_bump

    device = _forecast_engine.device
    _forecast_engine.model.eval()
    with torch.no_grad():
        x0 = torch.tensor(window, dtype=torch.float32,
                          device=device).unsqueeze(0)
        x1 = torch.tensor(cf_window, dtype=torch.float32,
                          device=device).unsqueeze(0)
        # Model returns raw logits; sigmoid them for probability comparison
        l0, _ = _forecast_engine.model(x0)
        l1, _ = _forecast_engine.model(x1)
        p0 = torch.sigmoid(l0)
        p1 = torch.sigmoid(l1)

    p0 = float(p0.item())
    p1 = float(p1.item())

    # Crude cost: cost_per_head_per_magnitude_unit * magnitude * dept size
    n = int(_monthly_df[_monthly_df["department"] == department]
            ["headcount"].iloc[-1])
    # cost scales linearly in |magnitude|
    est_cost = (cfg["cost_per_head_per_unit"]
                * abs(float(magnitude)) / 0.10
                * n)

    return {
        "department": department,
        "intervention": intervention,
        "description": cfg["description"],
        "feature_moved": feat_name,
        "magnitude_raw": magnitude,
        "baseline_prob": round(p0, 4),
        "counterfactual_prob": round(p1, 4),
        "delta_prob": round(p1 - p0, 4),
        "relative_change_pct": (round((p1 - p0) / p0 * 100, 1)
                                 if p0 > 0 else 0.0),
        "estimated_program_cost_usd": int(est_cost),
        "lead_time_days": cfg["lead_time_days"],
        "causal_status": cfg.get("causal_status", "MIXED"),
        "causal_rationale": cfg.get("causal_rationale", ""),
    }


# =============================================================================
# Signal 11-13 — Cost and impact
# =============================================================================

def _replacement_cost(department: str, predicted_departures: int) -> dict:
    """
    Dollarize predicted attrition at the department level:
      replacement_cost = base_salary * multiplier * n_departures
    """
    econ = _dept_economics().get(department)
    if not econ:
        return {"error": f"No replacement economics for {department}"}
    cost = econ["base_salary"] * econ["replacement_multiplier"] * predicted_departures
    return {
        "department": department,
        "predicted_departures": int(predicted_departures),
        "base_salary": int(econ["base_salary"]),
        "replacement_multiplier": econ["replacement_multiplier"],
        "estimated_replacement_cost_usd": int(cost),
        "safety_critical": econ["safety_critical"],
        "hiring_difficulty": econ["hiring_difficulty"],
    }


def _vacancy_days_at_risk(department: str, predicted_departures: int) -> dict:
    """Total expected days of open-seat-time before backfills ramp up."""
    econ = _dept_economics().get(department)
    if not econ:
        return {"error": f"No fill-time data for {department}"}
    total_days = econ["fill_days"] * predicted_departures
    return {
        "department": department,
        "predicted_departures": int(predicted_departures),
        "avg_fill_days_per_role": econ["fill_days"],
        "total_vacancy_days_at_risk": int(total_days),
        "interpretation": (
            f"If all {predicted_departures} predicted departures occur, "
            f"{department} faces ≈{total_days} vacancy-days across "
            f"{predicted_departures} open roles before backfills ramp up."
        ),
    }


def _knowledge_loss_severity(department: str) -> dict:
    """
    Institutional-knowledge risk = expected departures weighted by tenure.
    Quick proxy: look at the tenured cohort (≥8y) that is ALSO in the
    retirement-cliff emerging or critical cohort. The more senior people
    leaving, the higher the severity.
    """
    snap = _latest_employee_snapshot(department)
    if snap is None or snap.empty:
        return {"error": f"No employee data for {department}"}
    T = _cohort_thresholds()
    at_risk = snap[
        (snap["tenure_years"] >= T["tenured_ip_years"])
        & (snap["age"] >= T["retirement_emerging_age"])
    ]
    tenure_years_lost = float(at_risk["tenure_years"].sum())
    n_at_risk = int(len(at_risk))

    # Severity bucket
    if tenure_years_lost >= 80:
        severity = "CRITICAL"
    elif tenure_years_lost >= 40:
        severity = "HIGH"
    elif tenure_years_lost >= 15:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "department": department,
        "tenured_ip_at_retirement_risk": n_at_risk,
        "tenure_years_at_risk": round(tenure_years_lost, 1),
        "severity": severity,
        "interpretation": (
            f"{n_at_risk} tenured employee(s) — "
            f"{tenure_years_lost:.0f} combined years of institutional "
            f"knowledge — are in the retirement-cliff window."
        ),
    }


# =============================================================================
# Signal 14 — Internal mobility bench
# =============================================================================

def _internal_mobility_bench(for_department: str, top_n: int = 5) -> dict:
    """
    For each adjacent department, find employees that could plausibly
    fill a vacancy in `for_department`. Ranks by adjacency score × a
    "portability" score (tenure >= 3, performance >= 3.5, satisfaction
    >= 3.0 all lift the score).
    """
    if _individual_df is None:
        return {"error": "Individual data not loaded"}

    adj = _dept_adjacency().get(for_department, {})
    if not adj:
        return {"department": for_department, "candidates": [],
                "note": "No adjacent departments defined."}

    max_m = int(_individual_df["month"].max())
    latest = _individual_df[_individual_df["month"] >= max_m - 1]

    out = []
    for src_dept, adj_score in adj.items():
        pool = latest[latest["department"] == src_dept]
        if pool.empty:
            continue
        for _, row in pool.iterrows():
            portability = (
                (1.0 if row["tenure_years"] >= 3 else 0.6)
                * (1.0 if row["performance"] >= 3.5 else 0.7)
                * (1.0 if row["satisfaction"] >= 3.0 else 0.8)
                * (1.0 if row["engagement"] >= 3.0 else 0.8)
            )
            score = float(adj_score) * float(portability)
            out.append({
                "employee_id": int(row["employee_id"]),
                "from_department": src_dept,
                "to_department": for_department,
                "adjacency_score": round(float(adj_score), 2),
                "portability_score": round(portability, 2),
                "mobility_score": round(score, 3),
                "tenure_years": round(float(row["tenure_years"]), 1),
                "performance": round(float(row["performance"]), 2),
                "satisfaction": round(float(row["satisfaction"]), 2),
            })
    out.sort(key=lambda x: -x["mobility_score"])
    return {
        "for_department": for_department,
        "total_candidates_considered": len(out),
        "top_candidates": out[:top_n],
    }


# =============================================================================
# Signal 15 — Market competition (BLS × hiring difficulty)
# =============================================================================

def _market_competition_signal(department: str) -> dict:
    """
    Blend external labor-market signal (BLS index trend) with the
    department's hiring difficulty to produce a headwind/tailwind
    classification that an HR team can act on.
    """
    if _monthly_df is None:
        return {"error": "Monthly data not loaded"}
    bls_series = _monthly_df.groupby("month")["bls_energy_index"].first()
    if len(bls_series) < 4:
        return {"status": "insufficient_bls_history"}
    latest = float(bls_series.iloc[-1])
    prior = float(bls_series.iloc[-4])
    bls_delta = latest - prior

    econ = _dept_economics().get(department, {})
    difficulty = econ.get("hiring_difficulty", "moderate")

    # Headwind if labor market is hot AND dept is already hard to fill.
    if bls_delta > 1.0 and difficulty in ("hard",):
        classification = "strong_headwind"
        narrative = ("External market is hiring aggressively and "
                     f"{department} roles are already hard to fill — "
                     "expect longer time-to-hire and higher comp pressure.")
    elif bls_delta > 0.5:
        classification = "headwind"
        narrative = ("External market is growing; expect modest comp and "
                     "time-to-hire pressure.")
    elif bls_delta < -1.0:
        classification = "tailwind"
        narrative = ("External market is softening; hiring pressure "
                     "should ease and internal retention gets easier.")
    else:
        classification = "neutral"
        narrative = "External market is roughly balanced."

    return {
        "department": department,
        "bls_quarterly_delta": round(bls_delta, 2),
        "hiring_difficulty": difficulty,
        "classification": classification,
        "narrative": narrative,
    }


# =============================================================================
# Signal 16 — Retention plan
# =============================================================================

def _retention_plan(department: str, top_drivers: list,
                    cohorts: dict) -> list:
    """
    Map the top 1-3 risk drivers to concrete retention interventions
    from `_interventions()`. For each action, estimate the program cost
    based on the targeted cohort size.
    """
    plan = []
    driver_to_intervention = {
        "comp_ratio":   ["comp_adjustment", "retention_bonus"],
        "satisfaction": ["satisfaction_program", "flex_work"],
        "engagement":   ["engagement_initiative", "satisfaction_program"],
        "tenure_years": ["knowledge_capture"],  # new-hire churn OR retirement
        "age":          ["knowledge_capture", "retention_bonus"],
    }
    cohort_for_driver = {
        "comp_ratio":   cohorts.get("comp_gap", {}),
        "satisfaction": cohorts.get("low_engagement", {}),
        "engagement":   cohorts.get("low_engagement", {}),
        "tenure_years": cohorts.get("new_hire_churn_window", {}),
        "age":          cohorts.get("retirement_cliff_emerging", {}),
    }

    seen_interventions = set()
    for d in top_drivers[:3]:
        feat = d.get("feature")
        options = driver_to_intervention.get(feat)
        if not options:
            continue
        cohort = cohort_for_driver.get(feat, {}) or {}
        n = int(cohort.get("count", 0))
        if n == 0:
            # Fall back to 20% of the department for a scoped intervention
            n = max(5, int(cohorts.get("department_size", 20) * 0.2))

        for iv_key in options:
            if iv_key in seen_interventions:
                continue
            seen_interventions.add(iv_key)
            iv = _interventions()[iv_key]
            # Scope cost at a standard magnitude of 0.10 (unit of effect)
            est_cost = iv["cost_per_head_per_unit"] * n
            plan.append({
                "target_driver": feat,
                "target_driver_label": _feature_labels().get(feat, feat),
                "intervention": iv_key,
                "description": iv["description"],
                "cohort_size": n,
                "cohort_label": cohort.get("label", "(scoped subset)"),
                "est_program_cost_usd": int(est_cost),
                "lead_time_days": iv["lead_time_days"],
                "feature_moved": iv["feature_affected"],
                "causal_status": iv.get("causal_status", "MIXED"),
                "causal_rationale": iv.get("causal_rationale", ""),
            })
            break  # one intervention per driver — keeps plan tight

    return plan


# =============================================================================
# Signal 17 — Confidence
# =============================================================================

def _compute_workforce_confidence(department: str, months_available: int,
                                   dept_size: int,
                                   top_driver_share: float) -> dict:
    """
    Confidence in the analysis, factoring in sample size, history depth,
    and how concentrated the driver-attribution signal is.
    """
    reasons = []
    score = 1.0

    if dept_size < 15:
        score -= 0.25
        reasons.append(f"Small department (n={dept_size}) — "
                       "cohort counts are noisy.")
    elif dept_size < 30:
        score -= 0.10
        reasons.append(f"Moderate department size (n={dept_size}).")

    if months_available < 24:
        score -= 0.20
        reasons.append(f"Only {months_available} months of history — "
                       "seasonality and trend estimates are unstable.")
    elif months_available < 36:
        score -= 0.05

    if top_driver_share >= 0.40:
        reasons.append(
            f"Top driver explains {top_driver_share:.0%} of the Bi-LSTM's "
            "attrition score — strong attribution signal."
        )
    elif top_driver_share <= 0.20:
        score -= 0.05
        reasons.append(
            f"Top driver explains only {top_driver_share:.0%} of risk — "
            "multiple factors are contributing roughly equally."
        )

    score = max(0.0, min(1.0, score))
    if score >= 0.80:
        level = "high"
    elif score >= 0.50:
        level = "medium"
    else:
        level = "low"
    return {"level": level, "score": round(score, 2), "reasons": reasons}


# =============================================================================
# Signal 18 — Executive briefing (3-paragraph narrative)
# =============================================================================

_VERDICT_OPENERS = {
    "CRITICAL": "Immediate action required — {department} is at critical attrition risk.",
    "HIGH":     "Urgent — {department} is trending high risk and needs a retention plan now.",
    "MEDIUM":   "Watch item — {department} shows meaningful attrition signal.",
    "LOW":      "Healthy — {department}'s attrition signal is within normal range.",
}


def _generate_workforce_briefing(result: dict) -> dict:
    """
    Boardroom-ready 3-paragraph narrative that stitches every signal
    together. Mirrors the shape of talent_tools' executive_briefing so
    the dashboard can render both tabs with the same UI component.
    """
    dept = result.get("department", "?")
    risk_level = result.get("risk_level", "MEDIUM")
    prob = result.get("attrition_probability", 0.0)
    drivers = result.get("drivers", {}).get("top_drivers", []) or []
    cohorts = result.get("cohorts", {}) or {}
    trend = result.get("trend", {}) or {}
    leads = result.get("lead_indicators", {}) or {}
    cost = result.get("replacement_cost", {}) or {}
    knowledge = result.get("knowledge_loss", {}) or {}
    bench = result.get("internal_mobility", {}) or {}
    market = result.get("market_competition", {}) or {}
    plan = result.get("retention_plan", []) or []
    confidence = result.get("confidence", {}) or {}

    # ---- Paragraph 1: verdict + headline ----
    opener_template = _VERDICT_OPENERS.get(risk_level, _VERDICT_OPENERS["MEDIUM"])
    p1 = [opener_template.format(department=dept)]
    # Cite uncertainty interval when available — boardroom rigour.
    uncertainty = result.get("uncertainty", {}) or {}
    if (uncertainty.get("mean_probability") is not None
            and uncertainty.get("std") is not None):
        p5 = uncertainty.get("percentile_5", 0)
        p95 = uncertainty.get("percentile_95", 0)
        p1.append(
            f"Bi-LSTM attrition probability: {prob:.0%} "
            f"±{uncertainty['std']*100:.1f}pp "
            f"(90% CI: [{p5:.0%}, {p95:.0%}], "
            f"{uncertainty.get('uncertainty_level', 'medium')} confidence). "
            f"Trend is {trend.get('trend_direction', 'stable')}."
        )
    else:
        p1.append(
            f"Bi-LSTM attrition probability: {prob:.0%}. "
            f"Trend is {trend.get('trend_direction', 'stable')}."
        )
    if cost.get("estimated_replacement_cost_usd"):
        p1.append(
            f"Modelled replacement exposure: "
            f"${cost['estimated_replacement_cost_usd']:,} "
            f"({cost['predicted_departures']} predicted departures)."
        )
    if market.get("classification") in ("strong_headwind", "headwind"):
        p1.append(market.get("narrative", ""))

    # ---- Paragraph 2: drivers + cohorts ----
    p2_parts = []
    if drivers:
        top3 = drivers[:3]
        driver_line = ", ".join(
            f"{d['label']} ({d['pct_of_risk']:.0f}%)" for d in top3
        )
        p2_parts.append(f"Top risk drivers: {driver_line}.")
    cohort_items = []
    for key, label_emoji in [
        ("retirement_cliff_critical", "retirement-cliff (critical)"),
        ("new_hire_churn_window", "new-hire churn"),
        ("comp_gap", "comp-gap"),
        ("low_engagement", "low-engagement"),
    ]:
        c = cohorts.get(key, {})
        if c.get("count", 0) >= 3:
            cohort_items.append(
                f"{c['count']} in {label_emoji} ({c['pct_of_department']}%)"
            )
    if cohort_items:
        p2_parts.append("Cohorts at risk: " + ", ".join(cohort_items) + ".")
    if leads.get("early_warning"):
        p2_parts.append(
            f"Lead indicator: "
            f"satisfaction Δ {leads.get('satisfaction_3mo_momentum', 0):+.2f} / "
            f"engagement Δ {leads.get('engagement_3mo_momentum', 0):+.2f} — "
            "morale trend is negative ahead of the attrition signal."
        )
    if knowledge.get("severity") in ("HIGH", "CRITICAL"):
        p2_parts.append(
            f"Institutional-knowledge risk is {knowledge['severity']}: "
            f"{knowledge.get('tenure_years_at_risk', 0):.0f} years of tenure "
            "are exposed to retirement."
        )

    # ---- Paragraph 3: plan + mobility + confidence ----
    p3_parts = []
    if plan:
        total_cost = sum(p.get("est_program_cost_usd", 0) for p in plan)
        p3_parts.append(
            f"Recommended retention plan: "
            + "; ".join(
                f"{p['intervention'].replace('_', ' ')} → "
                f"{p['target_driver_label']}"
                for p in plan[:3]
            )
            + f" — estimated program cost ${total_cost:,}."
        )
    if bench.get("top_candidates"):
        n = len(bench["top_candidates"])
        srcs = sorted({c["from_department"] for c in bench["top_candidates"]})
        p3_parts.append(
            f"Internal mobility bench: {n} portable candidate(s) "
            f"sourceable from {', '.join(srcs)}."
        )
    roi_note = ""
    if plan and cost.get("estimated_replacement_cost_usd"):
        plan_cost = sum(p.get("est_program_cost_usd", 0) for p in plan)
        if plan_cost > 0:
            ratio = cost["estimated_replacement_cost_usd"] / plan_cost
            if ratio >= 2:
                roi_note = (
                    f"Program cost is {ratio:.1f}× below the modelled "
                    "replacement exposure — favourable ROI."
                )
    if roi_note:
        p3_parts.append(roi_note)
    if confidence.get("level") and confidence["level"] != "high":
        p3_parts.append(
            f"Analysis confidence: {confidence['level']} — "
            "validate cohort counts with HRIS before execution."
        )

    return {
        "verdict_paragraph": " ".join(p1),
        "drivers_paragraph": " ".join(p2_parts) or (
            "No individual driver dominates; risk is distributed across "
            "multiple factors."),
        "action_paragraph": " ".join(p3_parts) or (
            "No targeted interventions recommended at this risk level — "
            "continue monitoring."),
    }


# =============================================================================
# MAIN ANALYZER TOOL — the "analyze_skill_gap" counterpart for Agent 2
# =============================================================================

@tool("Workforce Risk Analyzer")
def analyze_workforce_risk(department: str) -> str:
    """
    Enterprise-grade workforce-risk analysis for a department.

    Produces an 18-signal JSON that combines the Bi-LSTM's attrition
    probability with:
      - gradient-based driver attribution (which features drive the risk)
      - cohort segmentation (retirement cliff, new-hire churn, comp gap,
        low engagement, tenured institutional-knowledge)
      - temporal intelligence (trend direction, seasonality, 3-mo momentum)
      - intervention simulator (counterfactual Bi-LSTM forward pass)
      - replacement cost + vacancy-days + knowledge-loss severity
      - internal mobility bench (cross-department backfill candidates)
      - market competition classifier (BLS × hiring difficulty)
      - targeted retention plan with program-cost estimates
      - confidence signal
      - executive briefing (3-paragraph narrative, boardroom-ready)

    Input: Department name (e.g., 'Operations', 'Engineering', 'HSE').
    Output: JSON with ~18 named signal blocks plus a natural-language
    summary the LLM agent can paraphrase directly.
    """
    if _forecast_engine is None or _monthly_df is None:
        return json.dumps({"error": "Forecast engine / data not initialized"})
    if department not in _monthly_df["department"].unique():
        return json.dumps({"error": f"Department '{department}' not found"})

    # ---- Core Bi-LSTM prediction ----
    prediction = _forecast_engine.predict_department(_monthly_df, department)
    if "error" in prediction:
        return json.dumps(prediction)

    prob = prediction["attrition_probability"]
    risk_level = prediction["risk_level"]
    current_hc = prediction["current_headcount"]
    predicted_departures = max(1, int(round(current_hc * prob * 12
                                            / 12)))  # 12-month projection
    # Cap at department size
    predicted_departures = min(predicted_departures, current_hc)

    # ---- 1-2: driver attribution ----
    drivers = _driver_attribution(department)

    # ---- 3-6: cohort segmentation ----
    cohorts = _cohort_segmentation(department)

    # ---- 7-9: temporal intelligence ----
    trend = _trend_analysis(department)
    seas = _seasonality(department)
    leads = _lead_indicators(department)

    # ---- 11-13: cost and impact ----
    cost = _replacement_cost(department, predicted_departures)
    vacancy = _vacancy_days_at_risk(department, predicted_departures)
    knowledge = _knowledge_loss_severity(department)

    # ---- 14: internal mobility ----
    bench = _internal_mobility_bench(department)

    # ---- 15: market competition ----
    market = _market_competition_signal(department)

    # ---- 16: retention plan ----
    plan = _retention_plan(
        department,
        drivers.get("top_drivers", []) if isinstance(drivers, dict) else [],
        cohorts if isinstance(cohorts, dict) else {},
    )

    # ---- 17: confidence ----
    months_available = int(_monthly_df[_monthly_df["department"] == department]
                           .shape[0])
    dept_size = cohorts.get("department_size", 0) if isinstance(cohorts, dict) else 0
    top_share = (drivers.get("top_drivers", [{}])[0].get("pct_of_risk", 0) / 100
                 if isinstance(drivers, dict) and drivers.get("top_drivers")
                 else 0.0)
    confidence = _compute_workforce_confidence(
        department, months_available, dept_size, top_share,
    )

    # ---- 19: epistemic uncertainty via MC Dropout (frontier capability) ----
    # 30 samples is a good trade-off — tighter CIs with negligible extra cost
    # on the small Bi-LSTM, keeps main analyzer latency under 2s.
    uncertainty = _mc_dropout_uncertainty(department, n_samples=30)

    # ---- Assemble partial result for briefing generation ----
    partial = {
        "department": department,
        "attrition_probability": prob,
        "risk_level": risk_level,
        "current_headcount": current_hc,
        "predicted_departures_12mo": predicted_departures,
        "projected_headcount_3m": prediction["projected_headcount_3m"],
        "projected_headcount_6m": prediction["projected_headcount_6m"],
        "projected_headcount_12m": prediction["projected_headcount_12m"],
        "monthly_headcount_delta": prediction["monthly_delta"],
        "drivers": drivers,
        "cohorts": cohorts,
        "trend": trend,
        "seasonality": seas,
        "lead_indicators": leads,
        "replacement_cost": cost,
        "vacancy_exposure": vacancy,
        "knowledge_loss": knowledge,
        "internal_mobility": bench,
        "market_competition": market,
        "retention_plan": plan,
        "confidence": confidence,
        "uncertainty": uncertainty,
    }

    # ---- 18: executive briefing (depends on partial above) ----
    briefing = _generate_workforce_briefing(partial)
    partial["executive_briefing"] = briefing

    # ---- Natural-language one-line summary ----
    summary_parts = [
        f"{department}: {risk_level} risk "
        f"({prob:.0%} attrition probability, "
        f"≈{predicted_departures} predicted departures over 12mo)."
    ]
    if drivers.get("top_drivers"):
        d0 = drivers["top_drivers"][0]
        summary_parts.append(
            f"Primary driver: {d0['label']} "
            f"({d0['pct_of_risk']:.0f}% of risk)."
        )
    if cost.get("estimated_replacement_cost_usd"):
        summary_parts.append(
            f"Modelled replacement exposure: "
            f"${cost['estimated_replacement_cost_usd']:,}."
        )
    if plan:
        summary_parts.append(
            f"Plan: {plan[0]['intervention'].replace('_', ' ')} "
            f"for {plan[0]['cohort_size']} people "
            f"(${plan[0]['est_program_cost_usd']:,})."
        )
    if confidence["level"] != "high":
        summary_parts.append(f"Confidence: {confidence['level']}.")
    partial["summary"] = " ".join(summary_parts)

    return json.dumps(partial, indent=2, default=str)


# =============================================================================
# Supporting tools — callable individually by the agent when it doesn't need
# the full 18-signal analyzer.
# =============================================================================

@tool("Risk Driver Ranker")
def rank_risk_drivers(department: str) -> str:
    """
    Attribute the Bi-LSTM's attrition score for `department` to its 8
    input features using gradient × input saliency. Returns ranked list.
    Input: department name.
    Output: JSON with per_feature_importance + top_drivers list.
    """
    return json.dumps(_driver_attribution(department), indent=2, default=str)


@tool("Retention Cohort Identifier")
def identify_retention_cohorts(department: str) -> str:
    """
    Segment `department`'s active workforce into decision-relevant
    cohorts: retirement cliff, new-hire churn, low engagement, comp gap,
    tenured IP, high performers at risk.
    Input: department name.
    Output: JSON with each cohort's count, %-of-dept, and employee IDs.
    """
    return json.dumps(_cohort_segmentation(department), indent=2, default=str)


@tool("Replacement Cost Estimator")
def estimate_replacement_cost(department: str) -> str:
    """
    Estimate the 12-month replacement cost, vacancy-day exposure, and
    knowledge-loss severity for `department`. Uses Bi-LSTM prediction as
    the departure count estimator, then applies per-role replacement
    economics.
    Input: department name.
    Output: JSON with replacement cost $, vacancy days, knowledge severity.
    """
    if _forecast_engine is None or _monthly_df is None:
        return json.dumps({"error": "Forecast engine not initialized"})
    pred = _forecast_engine.predict_department(_monthly_df, department)
    if "error" in pred:
        return json.dumps(pred)
    cur_hc = pred["current_headcount"]
    n_dep = min(cur_hc, max(1, int(round(cur_hc * pred["attrition_probability"]))))
    return json.dumps({
        "department": department,
        "bi_lstm_attrition_probability": pred["attrition_probability"],
        "predicted_departures_12mo": n_dep,
        "replacement_cost": _replacement_cost(department, n_dep),
        "vacancy_exposure": _vacancy_days_at_risk(department, n_dep),
        "knowledge_loss": _knowledge_loss_severity(department),
    }, indent=2, default=str)


@tool("Intervention Simulator")
def simulate_intervention(department: str, intervention: str,
                          magnitude: float = 0.10) -> str:
    """
    Run a counterfactual forward pass through the Bi-LSTM to estimate how
    much a specific retention intervention would lower attrition risk.

    Input:
      department   — e.g., 'Engineering'
      intervention — one of: comp_adjustment, satisfaction_program,
                     engagement_initiative, retention_bonus, flex_work,
                     knowledge_capture
      magnitude    — size of the lift, default 0.10 (10% on the target
                     feature; e.g., comp_ratio +0.10)
    Output: JSON with baseline vs counterfactual probability + cost est.
    """
    return json.dumps(
        _simulate_intervention_effect(department, intervention, float(magnitude)),
        indent=2, default=str,
    )


@tool("Internal Mobility Candidate Finder")
def find_internal_mobility_candidates(for_department: str,
                                       top_n: int = 5) -> str:
    """
    Find employees in adjacent departments who could fill a vacancy in
    `for_department`. Ranks by adjacency × portability (tenure,
    performance, satisfaction, engagement).

    Input:
      for_department — the receiving department (e.g., 'Operations')
      top_n          — number of candidates to return (default 5)
    Output: JSON with ranked candidate list including from_department,
    mobility_score, and individual signals.
    """
    return json.dumps(
        _internal_mobility_bench(for_department, top_n=int(top_n)),
        indent=2, default=str,
    )


@tool("Workforce Executive Briefing")
def generate_workforce_briefing(department: str) -> str:
    """
    Produce a boardroom-ready 3-paragraph workforce briefing for
    `department`: verdict, drivers + cohorts, action plan. Runs the full
    18-signal analyzer and returns just the narrative section — useful
    when the agent wants the story, not the raw JSON.

    Input: department name.
    Output: JSON with verdict_paragraph / drivers_paragraph / action_paragraph.
    """
    raw = analyze_workforce_risk.run(department=department)
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "analyzer did not return JSON"})
    if "error" in result:
        return json.dumps(result)
    return json.dumps({
        "department": department,
        "executive_briefing": result.get("executive_briefing", {}),
        "summary": result.get("summary", ""),
    }, indent=2)


# =============================================================================
# ============================================================================
# FRONTIER CAPABILITIES — intelligence-layer upgrades beyond the core 18
# signals. These turn Agent 2 into a decision-grade product that quantifies
# uncertainty, acts at the individual level, projects the future, and
# optimizes portfolios under budget constraints.
#
#   19. Epistemic uncertainty (Monte Carlo Dropout) — "probability ± pp"
#   20. Per-employee individual risk scoring — "which 7 people first?"
#   21. Risk trajectory forecast — "if we do nothing, where are we in 12mo?"
#   22. Budget-constrained intervention portfolio optimizer — "$500K → max
#       attrition reduction across all depts"
# ============================================================================
# =============================================================================


# -----------------------------------------------------------------------------
# Signal 19 — Epistemic uncertainty via Monte Carlo Dropout
# -----------------------------------------------------------------------------

def _mc_dropout_uncertainty(department: str, n_samples: int = 50) -> dict:
    """
    Monte Carlo Dropout — epistemic uncertainty quantification.

    Enables dropout at inference time (by toggling the model to train mode
    without gradient tracking) and runs `n_samples` forward passes. The
    spread of the resulting probability distribution is a variational
    approximation of the model's epistemic uncertainty: narrow spread = the
    model is confident, wide spread = the model doesn't know.

    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (ICML 2016).

    Returns
    -------
    dict with mean / std / 5th-95th percentile / range / uncertainty_level
    / natural-language interpretation. Suitable for boardroom output:
    "57% ± 8pp (90% CI: [42%, 69%])".
    """
    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine / data not ready"}

    import torch

    features, _, _ = _forecast_engine.prepare_department_data(
        _individual_df if _individual_df is not None else pd.DataFrame(),
        _monthly_df, department,
    )
    if features is None:
        return {"error": f"Insufficient data for {department}"}

    from config.settings import LSTM_SEQ_LENGTH
    window = features[-LSTM_SEQ_LENGTH:]
    x = torch.tensor(window, dtype=torch.float32,
                     device=_forecast_engine.device).unsqueeze(0)

    # Enable dropout — model.train() turns on Dropout/LSTM-dropout layers,
    # but we wrap inference in torch.no_grad() so no gradient state leaks.
    _forecast_engine.model.train()

    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logit, _reg = _forecast_engine.model(x)
            probs.append(float(torch.sigmoid(logit).item()))

    # Reset to eval mode so other tools aren't affected.
    _forecast_engine.model.eval()

    arr = np.array(probs)
    mean = float(arr.mean())
    std = float(arr.std())
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))

    if std < 0.04:
        level = "high"
        level_narrative = "very confident"
    elif std < 0.10:
        level = "medium"
        level_narrative = "moderately confident"
    else:
        level = "low"
        level_narrative = "uncertain"

    return {
        "department": department,
        "n_samples": n_samples,
        "mean_probability": round(mean, 4),
        "std": round(std, 4),
        "percentile_5": round(p5, 4),
        "percentile_95": round(p95, 4),
        "confidence_interval_90pct": [round(p5, 4), round(p95, 4)],
        "range_width": round(float(arr.max() - arr.min()), 4),
        "uncertainty_level": level,
        "interpretation": (
            f"Model is {level_narrative}: predicts {mean:.1%} attrition "
            f"risk with ±{std*100:.1f}pp standard deviation. "
            f"90% CI: [{p5:.1%}, {p95:.1%}]."
        ),
    }


@tool("Risk Uncertainty Quantifier")
def quantify_risk_uncertainty(department: str, n_samples: int = 50) -> str:
    """
    Monte Carlo Dropout uncertainty quantification (Gal & Ghahramani 2016).
    Runs the Bi-LSTM forward pass `n_samples` times with dropout ACTIVE to
    estimate epistemic uncertainty — what the model doesn't know.

    Use this whenever you need to report a probability with a credible
    interval, not just a point estimate. Executive briefings should cite
    the 90% CI alongside the mean for any high-stakes decision.

    Input:
      department: dept name (e.g., 'Engineering').
      n_samples:  MC samples (default 50, 100+ for publication-grade).
    Output: JSON with mean_probability, std, 5th/95th percentiles, 90% CI,
    uncertainty_level (high/medium/low), and a boardroom-ready interpretation.
    """
    return json.dumps(
        _mc_dropout_uncertainty(department, int(n_samples)),
        indent=2, default=str,
    )


# -----------------------------------------------------------------------------
# Signal 20 — Per-employee individual risk scoring
# -----------------------------------------------------------------------------

def _score_individual_employees(department: str, top_n: int = 10) -> dict:
    """
    Score each active employee in `department` by an interpretable
    individual risk model built from the SAME thresholds used in cohort
    segmentation (_cohort_thresholds()). Each signal contributes a weight;
    high-performer status amplifies urgency (losing them is strictly
    worse). Returns the top-N highest-risk employees with itemized reasons.

    This is the bridge from department-level Bi-LSTM scores to the
    question HR actually needs answered: "which specific people do we
    talk to first on Monday?"
    """
    snap = _latest_employee_snapshot(department)
    if snap is None or snap.empty:
        return {"error": f"No employee data for {department}"}
    T = _cohort_thresholds()

    dept_sat_mean = float(snap["satisfaction"].mean())
    dept_eng_mean = float(snap["engagement"].mean())

    def score_one(row):
        s = 0.0
        reasons = []
        # Retirement pressure
        if row["age"] >= T["retirement_critical_age"]:
            s += 0.30
            reasons.append(f"age {int(row['age'])} — retirement-critical")
        elif row["age"] >= T["retirement_emerging_age"]:
            s += 0.15
            reasons.append(f"age {int(row['age'])} — retirement-emerging")
        # New-hire churn window
        if row["tenure_years"] < T["new_hire_tenure_years"]:
            s += 0.20
            reasons.append(f"tenure {row['tenure_years']:.1f}y — <2y churn window")
        # Comp gap
        if row["comp_ratio"] < T["comp_gap_ratio"]:
            s += 0.20
            reasons.append(f"comp_ratio {row['comp_ratio']:.2f} — below 0.90 threshold")
        # Low absolute engagement / satisfaction
        if row["satisfaction"] < T["low_satisfaction_score"]:
            s += 0.15
            reasons.append(f"satisfaction {row['satisfaction']:.1f} — below 2.5")
        if row["engagement"] < T["low_engagement_score"]:
            s += 0.15
            reasons.append(f"engagement {row['engagement']:.1f} — below 2.5")
        # Deviation from department mean (relative risk)
        if row["satisfaction"] < dept_sat_mean - 0.5:
            s += 0.10
            reasons.append(
                f"satisfaction {row['satisfaction']:.1f} vs dept mean "
                f"{dept_sat_mean:.1f} — {dept_sat_mean - row['satisfaction']:.1f}pt gap"
            )
        if row["engagement"] < dept_eng_mean - 0.5:
            s += 0.08
            reasons.append(
                f"engagement {row['engagement']:.1f} vs dept mean "
                f"{dept_eng_mean:.1f}"
            )
        # Amplify for high performers — losing them is strictly worse.
        is_high_perf = row["performance"] >= T["high_performer_score"]
        if is_high_perf and s > 0:
            s *= 1.2
            reasons.append("HIGH PERFORMER — urgency multiplied 1.2×")
        return min(s, 1.0), reasons, is_high_perf

    scored = []
    for _, row in snap.iterrows():
        s, rs, is_hp = score_one(row)
        scored.append({
            "employee_id": int(row["employee_id"]),
            "age": int(row["age"]),
            "tenure_years": round(float(row["tenure_years"]), 1),
            "comp_ratio": round(float(row["comp_ratio"]), 2),
            "satisfaction": round(float(row["satisfaction"]), 2),
            "engagement": round(float(row["engagement"]), 2),
            "performance": round(float(row["performance"]), 2),
            "is_high_performer": bool(is_hp),
            "individual_risk_score": round(s, 3),
            "risk_band": ("CRITICAL" if s >= 0.70 else
                           "HIGH" if s >= 0.50 else
                           "MEDIUM" if s >= 0.30 else "LOW"),
            "risk_reasons": rs,
        })
    scored.sort(key=lambda x: -x["individual_risk_score"])

    high_risk_count = sum(1 for x in scored if x["individual_risk_score"] >= 0.50)
    hp_at_risk = sum(1 for x in scored
                     if x["is_high_performer"] and x["individual_risk_score"] >= 0.40)

    return {
        "department": department,
        "department_size": len(scored),
        "top_at_risk_employees": scored[:top_n],
        "avg_risk_score": round(float(np.mean(
            [x["individual_risk_score"] for x in scored])), 3),
        "high_risk_count": int(high_risk_count),
        "high_performers_at_risk": int(hp_at_risk),
        "interpretation": (
            f"{department}: {high_risk_count}/{len(scored)} employees at "
            f"individual risk ≥0.50. "
            + (f"{hp_at_risk} high-performer(s) at risk — "
               "prioritize these 1:1 conversations first."
               if hp_at_risk else "No high-performers in the high-risk band.")
        ),
    }


@tool("Individual Employee Risk Scorer")
def score_individual_employees(department: str, top_n: int = 10) -> str:
    """
    Score each active employee in `department` by individual risk factors
    (retirement cliff, new-hire churn, comp gap, low engagement/satisfaction,
    deviations from department means). Amplifies urgency 1.2× for high
    performers — losing them is strictly worse.

    Use this to answer "which specific 7 people should I talk to first on
    Monday?" Every score is itemized with human-readable reasons so HR can
    defend the prioritization.

    Input:
      department: dept name.
      top_n:      number of highest-risk employees to return (default 10).
    Output: JSON with ranked employee list (employee_id, age, tenure, comp,
    satisfaction, engagement, performance, is_high_performer, risk_score,
    risk_band, itemized reasons) plus aggregate counts.
    """
    return json.dumps(
        _score_individual_employees(department, top_n=int(top_n)),
        indent=2, default=str,
    )


# -----------------------------------------------------------------------------
# Signal 21 — 12-month risk trajectory forecast
# -----------------------------------------------------------------------------

def _project_risk_trajectory(department: str,
                              horizon_months: int = 12) -> dict:
    """
    Project the Bi-LSTM's attrition probability for `department` forward
    by `horizon_months`, assuming no intervention.

    Method
    ------
    1. Compute 3-month per-feature momentum:
           momentum = mean(last 3 months) - mean(previous 3 months)
    2. For each horizon step h = 1..H:
         - Synthesize h new "future" monthly snapshots by linearly
           extrapolating each feature from the last observed row.
         - Slide the 12-month window forward h steps (drop h oldest,
           append h extrapolated).
         - Run Bi-LSTM on the projected window; sigmoid the logit.
    3. Return trajectory, drift (final-initial), peak month, direction.

    This gives the CHRO the "if we do nothing, where are we in 12 months?"
    curve. Pair with Intervention Simulator to compare trajectories under
    different action plans.
    """
    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine / data not ready"}

    import torch
    from config.settings import LSTM_SEQ_LENGTH

    features, _, _ = _forecast_engine.prepare_department_data(
        _individual_df if _individual_df is not None else pd.DataFrame(),
        _monthly_df, department,
    )
    if features is None:
        return {"error": f"Insufficient data for {department}"}
    if len(features) < LSTM_SEQ_LENGTH + 3:
        return {"error": "Insufficient history for momentum projection "
                          "(need at least 15 months)"}

    base_window = features[-LSTM_SEQ_LENGTH:].copy()
    # Per-feature monthly momentum (standardized units — the features are
    # already scaled by the engine's StandardScaler).
    recent3 = features[-3:].mean(axis=0)
    prior3 = features[-6:-3].mean(axis=0)
    momentum = recent3 - prior3

    device = _forecast_engine.device
    _forecast_engine.model.eval()
    trajectory = []

    with torch.no_grad():
        # t = 0 — current
        x0 = torch.tensor(base_window, dtype=torch.float32,
                          device=device).unsqueeze(0)
        p0 = float(torch.sigmoid(_forecast_engine.model(x0)[0]).item())
        trajectory.append({"months_ahead": 0,
                           "probability": round(p0, 4)})

        last_row = base_window[-1]
        for h in range(1, horizon_months + 1):
            # Build h extrapolated months: last_row + momentum * [1..h]
            new_months = np.stack([
                last_row + momentum * (i + 1) for i in range(h)
            ])
            if h >= LSTM_SEQ_LENGTH:
                proj_window = new_months[-LSTM_SEQ_LENGTH:]
            else:
                proj_window = np.vstack([base_window[h:], new_months])
            xh = torch.tensor(proj_window, dtype=torch.float32,
                              device=device).unsqueeze(0)
            ph = float(torch.sigmoid(_forecast_engine.model(xh)[0]).item())
            trajectory.append({"months_ahead": h,
                               "probability": round(ph, 4)})

    probs = [t["probability"] for t in trajectory]
    drift = probs[-1] - probs[0]
    peak_month = int(np.argmax(probs))
    peak_prob = float(np.max(probs))

    if drift > 0.08:
        direction = "worsening"
        direction_narrative = f"worsens from {probs[0]:.0%} to {probs[-1]:.0%}"
    elif drift < -0.08:
        direction = "improving"
        direction_narrative = f"improves from {probs[0]:.0%} to {probs[-1]:.0%}"
    else:
        direction = "stable"
        direction_narrative = f"holds near {probs[0]:.0%}"

    return {
        "department": department,
        "horizon_months": horizon_months,
        "trajectory": trajectory,
        "current_probability": round(float(probs[0]), 4),
        "final_probability": round(float(probs[-1]), 4),
        "drift": round(float(drift), 4),
        "peak_month_ahead": peak_month,
        "peak_probability": round(peak_prob, 4),
        "trajectory_direction": direction,
        "interpretation": (
            f"If no action is taken, {department}'s attrition risk "
            f"{direction_narrative} over {horizon_months} months. "
            f"Peak: {peak_prob:.0%} at month {peak_month}."
        ),
    }


@tool("Risk Trajectory Forecaster")
def forecast_risk_trajectory(department: str,
                              horizon_months: int = 12) -> str:
    """
    Project the attrition probability trajectory for `department` over
    the next `horizon_months`, assuming no intervention. Uses 3-month
    per-feature momentum to extrapolate the Bi-LSTM's input window forward,
    then runs the model at each horizon step.

    Use this to answer "if we do nothing, where are we in 12 months?" —
    pairs with Intervention Simulator to compare "no-action" vs "action-
    plan" trajectories.

    Input:
      department:     dept name.
      horizon_months: how far forward to project (default 12).
    Output: JSON with monthly trajectory array, drift, peak month,
    direction (worsening/stable/improving), and natural-language summary.
    """
    return json.dumps(
        _project_risk_trajectory(department, horizon_months=int(horizon_months)),
        indent=2, default=str,
    )


# -----------------------------------------------------------------------------
# Signal 22 — Budget-constrained intervention portfolio optimizer
# -----------------------------------------------------------------------------

def _optimize_intervention_portfolio(budget_usd: int,
                                      focus_depts: Optional[list] = None
                                      ) -> dict:
    """
    Solve: max Σ (attrition reduction in pp) s.t. Σ cost ≤ budget, one
    intervention per (dept, intervention) pair.

    Strategy: enumerate all (dept × intervention × magnitude) candidates,
    compute ROI = reduction_in_pp / cost_in_$M, sort descending, and
    greedily select until budget exhausted. Greedy is optimal for this
    class of knapsack when items are divisible-like (same shape function)
    and near-optimal otherwise. Good enough for 144 candidates.

    Returns the selected portfolio with per-item ROI, cumulative cost,
    total attrition reduction, and unused budget.
    """
    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine / data not ready"}

    all_depts = list(_monthly_df["department"].unique())
    depts = focus_depts if focus_depts else all_depts
    # Only interventions that actually move probability
    interventions = [k for k, v in _interventions().items()
                     if v.get("feature_affected") is not None]
    magnitudes = [0.05, 0.10, 0.20]

    candidates = []
    for d in depts:
        for iv in interventions:
            for mag in magnitudes:
                sim = _simulate_intervention_effect(d, iv, mag)
                if "error" in sim:
                    continue
                delta = sim.get("delta_prob")
                cost = sim.get("estimated_program_cost_usd")
                if delta is None or cost is None or cost <= 0:
                    continue
                if delta >= 0:
                    # Intervention didn't reduce (or even raised) risk.
                    continue
                reduction_pp = -delta * 100
                roi = reduction_pp / (cost / 1_000_000)  # pp per $M
                candidates.append({
                    "department": d,
                    "intervention": iv,
                    "intervention_label": iv.replace("_", " ").title(),
                    "magnitude": mag,
                    "baseline_prob": sim.get("baseline_prob"),
                    "counterfactual_prob": sim.get("counterfactual_prob"),
                    "delta_prob": delta,
                    "attrition_reduction_pp": round(reduction_pp, 2),
                    "cost_usd": int(cost),
                    "roi_pp_per_million": round(roi, 2),
                    "lead_time_days": sim.get("lead_time_days"),
                })

    # Sort by ROI (best bang-for-buck first)
    candidates.sort(key=lambda c: -c["roi_pp_per_million"])

    # Greedy selection — one action per (dept, intervention) pair.
    selected = []
    spent = 0
    total_delta = 0.0
    used = set()
    for c in candidates:
        key = (c["department"], c["intervention"])
        if key in used:
            continue
        if spent + c["cost_usd"] > budget_usd:
            continue
        selected.append(c)
        spent += c["cost_usd"]
        total_delta += c["delta_prob"]
        used.add(key)

    # Per-dept aggregation
    by_dept = {}
    for s in selected:
        d = s["department"]
        by_dept.setdefault(d, {
            "department": d, "n_actions": 0,
            "cost_usd": 0, "reduction_pp": 0.0,
        })
        by_dept[d]["n_actions"] += 1
        by_dept[d]["cost_usd"] += s["cost_usd"]
        by_dept[d]["reduction_pp"] += s["attrition_reduction_pp"]

    dept_rollup = sorted(by_dept.values(), key=lambda r: -r["reduction_pp"])

    return {
        "budget_usd": int(budget_usd),
        "spent_usd": int(spent),
        "unused_budget_usd": int(budget_usd - spent),
        "n_interventions_selected": len(selected),
        "total_attrition_reduction_pp": round(-total_delta * 100, 2),
        "avg_roi_pp_per_million": (
            round(sum(s["roi_pp_per_million"] for s in selected)
                  / max(1, len(selected)), 2)
        ),
        "portfolio": selected,
        "per_department_rollup": dept_rollup,
        "candidates_evaluated": len(candidates),
        "interpretation": (
            f"With a ${budget_usd:,} budget, the optimal portfolio "
            f"selects {len(selected)} intervention(s) across "
            f"{len(by_dept)} department(s) for a combined "
            f"{-total_delta*100:.1f}pp attrition reduction. "
            f"Unused budget: ${budget_usd - spent:,}."
            + (f" Top action: {selected[0]['intervention_label']} in "
               f"{selected[0]['department']} "
               f"({selected[0]['attrition_reduction_pp']:.1f}pp at "
               f"${selected[0]['cost_usd']:,})."
               if selected else " No profitable interventions found.")
        ),
    }


# -----------------------------------------------------------------------------
# Signal 23 — Conformal Prediction (distribution-free coverage guarantee)
#
# MC Dropout gives a Bayesian-ish 90% credible interval — but its coverage
# is only approximate and depends on the dropout approximation being a good
# posterior. Conformal prediction (Vovk; Angelopoulos & Bates 2021) provides
# a DISTRIBUTION-FREE interval with a provably exact marginal coverage
# guarantee, regardless of the base model, data distribution, or class
# balance. This is the successor frontier method — 2024+ SOTA.
#
# Implementation: Split Conformal Prediction.
#   1. Reserve a calibration set (held-out monthly sequences).
#   2. For each calibration sample, compute the non-conformity score
#      s_i = |p_hat_i - y_i|  (absolute residual; valid for binary labels).
#   3. Compute threshold q_hat = quantile(S, ceil((n+1)(1-alpha))/n).
#   4. For a new test input, output interval [p_hat - q_hat, p_hat + q_hat]
#      clipped to [0,1]. This interval is guaranteed to contain the true
#      label with probability >= 1-alpha on average over the calibration+
#      test exchangeability assumption.
# -----------------------------------------------------------------------------

_CONFORMAL_CALIBRATION_CACHE: Dict[str, Any] = {}


def _build_conformal_calibration(alpha: float = 0.10) -> Dict[str, Any]:
    """
    Build (or retrieve from cache) the calibration set for Split Conformal
    Prediction. Uses the LAST N months per department as held-out
    calibration. Caches on alpha so re-calling the same tool is instant.

    Math (Angelopoulos & Bates 2021, "A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification"):

        Given:   n calibration pairs (X_i, Y_i)
        Compute: S_i = |p_hat(X_i) - Y_i|   for i = 1..n
        Threshold: q_hat = quantile(S, ceil((n+1)(1-alpha)) / n)

    Coverage guarantee: for any new exchangeable (X, Y),
        P(Y in [p_hat(X) - q_hat, p_hat(X) + q_hat]) >= 1 - alpha

    This holds REGARDLESS of the model's calibration quality. MC Dropout
    cannot make that claim.
    """
    cache_key = f"alpha_{alpha}"
    if cache_key in _CONFORMAL_CALIBRATION_CACHE:
        return _CONFORMAL_CALIBRATION_CACHE[cache_key]

    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine / data not ready"}

    import torch
    from config.settings import LSTM_SEQ_LENGTH

    # Build calibration pairs from the last window of each dept.
    # Each calibration sample is (12-month input window, binary attrition
    # label at the last month of the window).
    scores: List[float] = []
    per_dept_counts: Dict[str, int] = {}
    for dept in _monthly_df["department"].unique():
        features, attrition, _ = _forecast_engine.prepare_department_data(
            _individual_df if _individual_df is not None else pd.DataFrame(),
            _monthly_df, dept,
        )
        if features is None or len(features) < LSTM_SEQ_LENGTH + 3:
            continue
        # Use the last 6 months of each dept as calibration windows
        n_cal = min(6, len(features) - LSTM_SEQ_LENGTH)
        device = _forecast_engine.device
        _forecast_engine.model.eval()
        for end_idx in range(len(features) - n_cal, len(features)):
            if end_idx < LSTM_SEQ_LENGTH:
                continue
            window = features[end_idx - LSTM_SEQ_LENGTH:end_idx]
            y_true = float(attrition[end_idx - 1])
            x = torch.tensor(window, dtype=torch.float32,
                             device=device).unsqueeze(0)
            with torch.no_grad():
                logit, _reg = _forecast_engine.model(x)
                p_hat = float(torch.sigmoid(logit).item())
            # Non-conformity score: absolute residual.
            scores.append(abs(p_hat - y_true))
            per_dept_counts[dept] = per_dept_counts.get(dept, 0) + 1

    if not scores:
        return {"error": "Insufficient data to build calibration set"}

    n = len(scores)
    import math
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)   # clamp
    q_hat = float(np.quantile(np.array(scores), q_level, method="higher"))

    result = {
        "alpha": alpha,
        "coverage_target": 1 - alpha,
        "n_calibration": n,
        "q_hat": q_hat,
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "per_dept_counts": per_dept_counts,
    }
    _CONFORMAL_CALIBRATION_CACHE[cache_key] = result
    return result


def _conformal_prediction_interval(department: str,
                                    alpha: float = 0.10) -> Dict[str, Any]:
    """
    Produce a Split Conformal Prediction interval [p_low, p_high] for
    `department`'s attrition probability, with a distribution-free
    coverage guarantee of at least 1-alpha.

    Compare this with MC Dropout (signal 19): dropout's CI is approximate
    and depends on the dropout-as-Bayesian assumption. Conformal's
    guarantee holds for ANY base model. This is the frontier rigour.

    Returns
    -------
    point_estimate    — the Bi-LSTM's (deterministic) predicted probability
    p_low, p_high     — the conformal interval, clipped to [0,1]
    interval_width    — p_high - p_low
    q_hat             — the calibration quantile
    coverage_target   — 1 - alpha (e.g., 0.90 for alpha=0.10)
    n_calibration     — # of calibration samples used
    method            — "Split Conformal Prediction (Vovk; Angelopoulos 2021)"
    """
    cal = _build_conformal_calibration(alpha=alpha)
    if "error" in cal:
        return cal

    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine / data not ready"}

    import torch
    from config.settings import LSTM_SEQ_LENGTH

    features, _, _ = _forecast_engine.prepare_department_data(
        _individual_df if _individual_df is not None else pd.DataFrame(),
        _monthly_df, department,
    )
    if features is None:
        return {"error": f"Insufficient data for {department}"}

    window = features[-LSTM_SEQ_LENGTH:]
    x = torch.tensor(window, dtype=torch.float32,
                     device=_forecast_engine.device).unsqueeze(0)
    _forecast_engine.model.eval()
    with torch.no_grad():
        logit, _reg = _forecast_engine.model(x)
        p_hat = float(torch.sigmoid(logit).item())

    q_hat = cal["q_hat"]
    p_low = max(0.0, p_hat - q_hat)
    p_high = min(1.0, p_hat + q_hat)

    return {
        "department": department,
        "method": "Split Conformal Prediction (Vovk; Angelopoulos 2021)",
        "point_estimate": round(p_hat, 4),
        "p_low": round(p_low, 4),
        "p_high": round(p_high, 4),
        "interval_width": round(p_high - p_low, 4),
        "q_hat": round(q_hat, 4),
        "alpha": alpha,
        "coverage_target": cal["coverage_target"],
        "n_calibration": cal["n_calibration"],
        "interpretation": (
            f"{department} attrition probability: {p_hat:.1%} with "
            f"{int((1-alpha)*100)}% conformal interval "
            f"[{p_low:.1%}, {p_high:.1%}]. This interval has a "
            f"distribution-free coverage guarantee — it contains the true "
            f"label with probability >= {1-alpha:.0%}, regardless of the "
            f"Bi-LSTM's calibration quality."
        ),
    }


@tool("Conformal Prediction Interval")
def conformal_prediction_interval(department: str,
                                   alpha: float = 0.10) -> str:
    """
    Compute a Split Conformal Prediction interval for the attrition
    probability of `department`. Provides a DISTRIBUTION-FREE coverage
    guarantee — unlike MC Dropout's Bayesian-ish credible interval.

    Reference: Angelopoulos & Bates 2021, "A Gentle Introduction to
    Conformal Prediction and Distribution-Free Uncertainty Quantification"
    (https://arxiv.org/abs/2107.07511). This is the 2024+ frontier
    replacement for MC Dropout when coverage guarantees matter.

    Input:
      department: dept name.
      alpha:      desired miscoverage rate (0.10 = 90% coverage, default).
    Output: JSON with point_estimate, p_low, p_high, interval_width,
    q_hat, n_calibration, coverage_target, and natural-language summary.
    """
    return json.dumps(
        _conformal_prediction_interval(department, float(alpha)),
        indent=2, default=str,
    )


# -----------------------------------------------------------------------------
# Signal 24 — Scenario simulator (workforce shock modeling)
#
# Answers the CHRO's "what-if" questions:
#   - "What if we lose 20% of Engineering in Q3?"
#   - "What if the external market heats up (BLS +2 points)?"
#   - "What if we successfully run the full retention plan?"
#
# For each scenario, run the Bi-LSTM forward with the shocked features
# and report the probability delta. Composable — compare multiple
# scenarios side by side.
# -----------------------------------------------------------------------------

_SCENARIO_CATALOG: Dict[str, Dict[str, Any]] = {
    "mass_departure_20pct": {
        "description": "Sudden 20% headcount loss in the target department "
                       "(e.g., competitor recruits a cohort away).",
        "feature_deltas": {"headcount": -0.20},  # relative: -20%
        "relative": True,
    },
    "market_shock_hot": {
        "description": "External labor market heats sharply (BLS index "
                       "jumps +2 points) — e.g., new data center cluster.",
        "feature_deltas": {"bls_energy_index": 2.0},
        "relative": False,
    },
    "market_shock_cold": {
        "description": "External labor market softens (BLS -2 points) — "
                       "e.g., recessionary conditions.",
        "feature_deltas": {"bls_energy_index": -2.0},
        "relative": False,
    },
    "full_retention_plan_success": {
        "description": "Retention program hits all targets: comp_ratio +0.10, "
                       "satisfaction +0.5, engagement +0.3.",
        "feature_deltas": {
            "comp_ratio": 0.10, "satisfaction": 0.5, "engagement": 0.3,
        },
        "relative": False,
    },
    "morale_crash": {
        "description": "Morale collapse — satisfaction -0.8, engagement -0.6 "
                       "(e.g., botched reorg).",
        "feature_deltas": {"satisfaction": -0.8, "engagement": -0.6},
        "relative": False,
    },
    "retirement_wave": {
        "description": "Retirement wave — average age +3 years over 12 months "
                       "(no rehiring, senior cohort concentrates).",
        "feature_deltas": {"age": 3.0},
        "relative": False,
    },
}


def _simulate_scenario(department: str, scenario: str) -> Dict[str, Any]:
    """
    Run a named scenario through the Bi-LSTM. Baseline uses the current
    window; counterfactual applies the feature_deltas.
    """
    if _forecast_engine is None or _monthly_df is None:
        return {"error": "Forecast engine / data not ready"}
    if scenario not in _SCENARIO_CATALOG:
        return {"error": f"Unknown scenario '{scenario}'. "
                          f"Available: {list(_SCENARIO_CATALOG.keys())}"}

    import torch
    from config.settings import LSTM_SEQ_LENGTH

    cfg = _SCENARIO_CATALOG[scenario]
    features, _, _ = _forecast_engine.prepare_department_data(
        _individual_df if _individual_df is not None else pd.DataFrame(),
        _monthly_df, department,
    )
    if features is None:
        return {"error": f"Insufficient data for {department}"}

    window = features[-LSTM_SEQ_LENGTH:].copy()
    cf_window = window.copy()

    # Apply deltas in scaled space (features are already standard-scaled).
    scale_ = getattr(_forecast_engine.scaler, "scale_", None)
    feats_moved = []
    for feat_name, delta in cfg["feature_deltas"].items():
        if feat_name not in _forecast_engine.feature_columns:
            continue
        idx = _forecast_engine.feature_columns.index(feat_name)
        if cfg.get("relative", False):
            # Relative shock: multiply by (1 + delta) on the raw axis.
            # Approximate in scaled space via unscale → multiply → rescale.
            if scale_ is not None and idx < len(scale_) and hasattr(
                _forecast_engine.scaler, "mean_"
            ):
                mean_ = _forecast_engine.scaler.mean_[idx]
                raw_window = cf_window[:, idx] * scale_[idx] + mean_
                raw_window = raw_window * (1 + delta)
                cf_window[:, idx] = (raw_window - mean_) / scale_[idx]
            else:
                cf_window[:, idx] *= (1 + delta)
        else:
            # Absolute delta in raw units → scaled units.
            if scale_ is not None and idx < len(scale_):
                cf_window[:, idx] += delta / scale_[idx]
            else:
                cf_window[:, idx] += delta
        feats_moved.append(feat_name)

    device = _forecast_engine.device
    _forecast_engine.model.eval()
    with torch.no_grad():
        x0 = torch.tensor(window, dtype=torch.float32,
                          device=device).unsqueeze(0)
        x1 = torch.tensor(cf_window, dtype=torch.float32,
                          device=device).unsqueeze(0)
        p0 = float(torch.sigmoid(_forecast_engine.model(x0)[0]).item())
        p1 = float(torch.sigmoid(_forecast_engine.model(x1)[0]).item())

    return {
        "department": department,
        "scenario": scenario,
        "description": cfg["description"],
        "features_moved": feats_moved,
        "feature_deltas": cfg["feature_deltas"],
        "relative_mode": cfg.get("relative", False),
        "baseline_probability": round(p0, 4),
        "scenario_probability": round(p1, 4),
        "delta": round(p1 - p0, 4),
        "relative_change_pct": (round((p1 - p0) / p0 * 100, 1)
                                 if p0 > 0 else 0.0),
        "interpretation": (
            f"{department} under scenario '{scenario}': attrition would "
            f"move from {p0:.1%} (baseline) to {p1:.1%} "
            f"({(p1-p0)*100:+.1f}pp). "
            + cfg["description"]
        ),
    }


@tool("Workforce Scenario Simulator")
def simulate_scenario(department: str, scenario: str) -> str:
    """
    Simulate a named workforce scenario against the Bi-LSTM and report the
    delta vs baseline. Use for "what-if" planning: mass departures,
    market shocks, retention-plan success, morale crashes, retirement waves.

    Available scenarios:
      - mass_departure_20pct
      - market_shock_hot / market_shock_cold
      - full_retention_plan_success
      - morale_crash
      - retirement_wave

    Input:
      department: dept name (e.g., 'Engineering').
      scenario:   one of the named scenarios above.
    Output: JSON with baseline_probability, scenario_probability, delta,
    features_moved, description, and natural-language interpretation.
    """
    return json.dumps(
        _simulate_scenario(department, scenario),
        indent=2, default=str,
    )


@tool("Budget-Constrained Intervention Optimizer")
def optimize_intervention_portfolio(budget_usd: int = 500_000,
                                     focus_depts: str = "") -> str:
    """
    Given a fixed budget, solve for the portfolio of (department ×
    intervention × magnitude) actions that maximizes total attrition
    reduction across the company.

    Enumerates up to 8 depts × 6 interventions × 3 magnitudes = 144
    candidates, scores each by ROI (percentage-point attrition reduction
    per million dollars), and greedily fills the budget with best-ROI
    actions — one per (dept, intervention) pair.

    Use this to answer "we have $500K for retention this quarter — where
    do we spend it to maximize impact?" The output explains cost, ROI,
    and lead time for every selected action plus a per-department rollup.

    Input:
      budget_usd:  total budget in $ (default 500,000).
      focus_depts: comma-separated dept names to focus on, or empty for all.
    Output: JSON with selected portfolio, per-action ROI, per-dept rollup,
    total reduction in percentage points, and unused budget.
    """
    focus = None
    if focus_depts and focus_depts.strip():
        focus = [d.strip() for d in focus_depts.split(",") if d.strip()]
    return json.dumps(
        _optimize_intervention_portfolio(int(budget_usd), focus),
        indent=2, default=str,
    )
