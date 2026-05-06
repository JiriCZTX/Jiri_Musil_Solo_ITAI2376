"""
Per-class Gate 2 thresholds for v14 / GLiNER 10-type ship/no-ship decisions.

Per PATH_B5_PLUS_FINAL.md §11. DEV-SET ONLY. Blind set is NEVER touched
by Gate 2.

Schema version: v7 (10-type)
Locked: 2026-04-25
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ClassGate:
    """Per-class ship rule. All conditions must be true to ship for that class."""
    entity_type: str
    f1_improvement_pp: float            # min absolute pp gain over routing+gazetteer baseline
    precision_floor: Optional[float]    # min precision; None if suspended
    extra_rule: str = ""                # human-readable extra condition
    suspended: bool = False             # if True: candidate-only forever until extra_rule met

    def passes(
        self,
        candidate_f1: float,
        baseline_f1: float,
        candidate_precision: float,
        gold_support: int,
        max_owned_regression_pp: float,
    ) -> tuple[bool, str]:
        """Return (passes, reason). reason describes pass or specific failure."""
        if self.suspended:
            if "30 gold spans" in self.extra_rule and gold_support < 30:
                return False, f"{self.entity_type}: suspended — needs >=30 gold spans (have {gold_support})"
            # suspension extra-rule not yet met → still suspended
            return False, f"{self.entity_type}: suspended — {self.extra_rule}"
        if max_owned_regression_pp > 2.0:
            return False, f"{self.entity_type}: regression {max_owned_regression_pp:.2f}pp on owned class > 2pp ceiling"
        improvement = (candidate_f1 - baseline_f1) * 100
        if improvement < self.f1_improvement_pp:
            return False, (
                f"{self.entity_type}: F1 gain {improvement:.2f}pp < required +{self.f1_improvement_pp}pp "
                f"(candidate {candidate_f1:.4f} vs baseline {baseline_f1:.4f})"
            )
        if self.precision_floor is not None and candidate_precision < self.precision_floor:
            return False, (
                f"{self.entity_type}: precision {candidate_precision:.4f} < floor {self.precision_floor}"
            )
        return True, (
            f"{self.entity_type}: PASS (+{improvement:.2f}pp F1, precision {candidate_precision:.4f})"
        )


# ---------------------------------------------------------------------------
# Locked per-class Gate 2 table
# ---------------------------------------------------------------------------
# Source: PATH_B5_PLUS_FINAL.md §11. Per-class only — macro Gate 2 forbidden.
# Owned classes (legacy 5 + TOOL) are not in this table because v14/GLiNER
# never own them.
GATE2: Dict[str, ClassGate] = {
    "INDUSTRY": ClassGate(
        entity_type="INDUSTRY",
        f1_improvement_pp=3.0,
        precision_floor=0.50,
    ),
    "LOCATION": ClassGate(
        entity_type="LOCATION",
        f1_improvement_pp=3.0,
        precision_floor=0.60,
        extra_rule="zero personal-address leakage; PII guard verified",
    ),
    "SOFT_SKILL": ClassGate(
        entity_type="SOFT_SKILL",
        f1_improvement_pp=3.0,
        precision_floor=0.50,
        extra_rule="precision-priority; recall secondary",
    ),
    "PROJECT": ClassGate(
        entity_type="PROJECT",
        f1_improvement_pp=3.0,
        precision_floor=None,
        extra_rule=">=30 gold spans on dev/blind required before any F1 claim; otherwise candidate-only",
        suspended=True,
    ),
}

# Owned-class regression ceiling (no candidate may degrade these by > 2pp)
OWNED_CLASS_REGRESSION_CEILING_PP: float = 2.0

# Classes that the legacy stack owns; never delegated to v14/GLiNER
OWNED_BY_V6: tuple = ("SKILL", "CERT", "DEGREE", "EMPLOYER", "YEARS_EXP")
OWNED_BY_V11: tuple = ("TOOL",)
OWNED_LOCKED: tuple = OWNED_BY_V6 + OWNED_BY_V11

# Gate-2-eligible classes (per-class decision via GATE2)
GATE2_ELIGIBLE: tuple = ("INDUSTRY", "LOCATION", "SOFT_SKILL", "PROJECT")

# Gate-eval source: dev set only. Forbidden to call this with blind metrics.
GATE2_EVAL_SET: str = "eval_dev_v1"


def evaluate_gate2(
    candidate_name: str,
    per_class_metrics: Dict[str, Dict[str, float]],
    baseline_per_class_metrics: Dict[str, Dict[str, float]],
    eval_set_name: str,
) -> Dict[str, Dict[str, object]]:
    """
    Apply Gate 2 per-class for one candidate ({GLiNER, v14, gazetteer-only}).

    per_class_metrics: {class: {"f1": float, "precision": float, "support": int}}
    baseline_per_class_metrics: same shape, from routing+gazetteer baseline
    eval_set_name: must equal GATE2_EVAL_SET; raises if not.

    Returns: {class: {"passes": bool, "reason": str, "candidate_f1": float, ...}}
    """
    assert eval_set_name == GATE2_EVAL_SET, (
        f"Gate 2 must run on '{GATE2_EVAL_SET}' only; got '{eval_set_name}'. "
        "Blind set is forbidden for Gate 2 selection."
    )
    results: Dict[str, Dict[str, object]] = {}
    # owned-class regression check (worst-case across v6/v11 owned classes)
    max_owned_reg = 0.0
    for c in OWNED_LOCKED:
        if c in per_class_metrics and c in baseline_per_class_metrics:
            reg = (baseline_per_class_metrics[c]["f1"] - per_class_metrics[c]["f1"]) * 100
            if reg > max_owned_reg:
                max_owned_reg = reg

    for c in GATE2_ELIGIBLE:
        gate = GATE2[c]
        cm = per_class_metrics.get(c, {"f1": 0.0, "precision": 0.0, "support": 0})
        bm = baseline_per_class_metrics.get(c, {"f1": 0.0, "precision": 0.0, "support": 0})
        passes, reason = gate.passes(
            candidate_f1=cm["f1"],
            baseline_f1=bm["f1"],
            candidate_precision=cm["precision"],
            gold_support=cm.get("support", 0),
            max_owned_regression_pp=max_owned_reg,
        )
        results[c] = {
            "candidate": candidate_name,
            "passes": passes,
            "reason": reason,
            "candidate_f1": cm["f1"],
            "baseline_f1": bm["f1"],
            "candidate_precision": cm["precision"],
            "gold_support": cm.get("support", 0),
            "max_owned_regression_pp": max_owned_reg,
        }
    return results
