"""
PROPRIETARY WORKFORCE INTELLIGENCE BRAIN — LLM-free agent orchestrator.

This module is the alternative to agents.py (which uses CrewAI + Claude
Sonnet). It replaces the LLM reasoning layer with a deterministic
rule-based orchestrator that composes the 21 tools already built for
Agent 1 (talent_tools.py — 19 signals) and Agent 2 (forecast_tools.py —
22 signals) into five end-to-end workflows:

  1. joint_hire_analysis(resume, job, department)
       — Agent 1 analyzes the candidate, Agent 2 analyzes the receiving
         team's workforce risk, then the brain synthesizes a unified
         hire / retention / backfill memo. Mirrors CrewAI's joint
         synthesis mode — without the LLM.

  2. workforce_risk_scan(departments=None, top_n=3)
       — Cross-department risk scan producing a prioritized action
         list for the CHRO.

  3. quarterly_retention_plan(budget_usd, focus_depts=None)
       — Budget-constrained portfolio optimization wrapped in a
         board-ready memo. Calls the knapsack optimizer under the hood.

  4. rank_candidates_for_req(job, candidates)
       — Applicant shortlisting with narrative summary.

  5. match_candidate_across_reqs(resume, jobs)
       — Internal mobility triage: which of N open roles fits this
         candidate best?

Why proprietary:
  - Zero external API dependency (no Claude, no GPT, no local LLM).
  - Deterministic: same inputs → identical output, forever.
  - Auditable: every verdict traces to a named rule in the decision
    matrix. Satisfies EU AI Act explainability requirements.
  - Fast: <500 ms per memo vs 15–30 s for CrewAI+Claude.
  - Cheap: $0/memo vs $0.03–0.10 for LLM-backed alternatives.
  - Safe: narratives are templated from verified tool JSON — no
    hallucination surface.

Course connection: Module 10 — Agentic AI. This implementation shows
that "agentic" does not require an LLM brain; a principled orchestrator
with deep tools is sufficient and often superior for domains where
traceability matters more than open-ended generation.
"""

from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


def _new_memo_id() -> str:
    """UUIDv4-derived short memo identifier for cross-session feedback links."""
    return f"m_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Risk-classification policies (workforce-side analogue of the hire-side
# multi-brain pattern). The Bi-LSTM emits an attrition probability in [0, 1];
# the *risk_level* (CRITICAL / HIGH / MEDIUM / LOW) that downstream rules
# act on is a policy choice on top of that probability. Three policies:
#
#   - "sophisticated"  Production absolute thresholds (matches
#                       ForecastingEngine._prob_to_risk in bilstm_model.py).
#                       Best precision; surfaces only the clearly-elevated.
#   - "conservative"   Lower bars — surfaces more departments for review.
#                       Useful for risk-averse organizations and audit modes.
#   - "heuristic"      Rank-based, data-relative. Top quartile by probability
#                       → CRITICAL; next quartile → HIGH; next → MEDIUM; rest
#                       → LOW. Independent of absolute scale; answers
#                       "who's worst compared to peers?"
#
# Disagreement between the three is itself a first-class diagnostic — the
# borderline departments where threshold choice is doing the work get
# flagged for human review, mirroring the consensus pattern on the
# hire side.
# =============================================================================

_RISK_POLICY_THRESHOLDS: Dict[str, Tuple[float, float, float]] = {
    # (CRITICAL_min, HIGH_min, MEDIUM_min). Below MEDIUM_min → LOW.
    "sophisticated": (0.70, 0.50, 0.30),
    "conservative":  (0.55, 0.35, 0.20),
}

_RISK_POLICY_LABELS: Dict[str, str] = {
    "sophisticated": "Sophisticated",
    "conservative":  "Conservative",
    "heuristic":     "Heuristic baseline",
}

_RISK_POLICY_BLURBS: Dict[str, str] = {
    "sophisticated": "Production absolute thresholds (≥0.70 CRITICAL, ≥0.50 HIGH, ≥0.30 MEDIUM).",
    "conservative":  "Risk-averse absolute thresholds (≥0.55 CRITICAL, ≥0.35 HIGH, ≥0.20 MEDIUM).",
    "heuristic":     "Rank-based quartiles across the scanned set — top 25% CRITICAL, etc.",
}


def _classify_under_absolute(prob: float, policy_key: str) -> str:
    """Map a probability to a risk level under one of the absolute policies."""
    crit, high, med = _RISK_POLICY_THRESHOLDS[policy_key]
    if prob >= crit:
        return "CRITICAL"
    if prob >= high:
        return "HIGH"
    if prob >= med:
        return "MEDIUM"
    return "LOW"


def _classify_under_heuristic(prob: float,
                                all_probs: List[float]) -> str:
    """Rank-based classification — CRITICAL/HIGH/MEDIUM/LOW by quartile.

    With N departments, the top ceil(N/4) are CRITICAL, next ceil(N/4) HIGH,
    etc. Stable for ties: a probability sitting on a quartile boundary
    inherits the worse classification, which mirrors how a risk-averse
    reviewer would read the chart.
    """
    if not all_probs:
        return "LOW"
    sorted_probs = sorted(all_probs, reverse=True)
    n = len(sorted_probs)
    # Quartile cutoffs by index — ceil-based so the top bucket is non-empty
    # for any N >= 1. With N=8 → 2/2/2/2; with N=5 → 2/2/1/0; with N=1 → 1/0/0/0.
    import math as _math
    q = _math.ceil(n / 4)
    crit_cut = sorted_probs[min(q - 1, n - 1)]
    high_cut = sorted_probs[min(2 * q - 1, n - 1)]
    med_cut = sorted_probs[min(3 * q - 1, n - 1)]
    if prob >= crit_cut:
        return "CRITICAL"
    if prob >= high_cut:
        return "HIGH"
    if prob >= med_cut:
        return "MEDIUM"
    return "LOW"


def _multi_policy_risk_classifications(
    departments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For every department in the scan, return its per-policy
    risk-level classification plus a disagreement diagnostic.

    Input ``departments`` is the ``ranked`` list emitted by
    ``predict_all_departments`` (each item has ``department`` and
    ``attrition_probability``). The output preserves order and adds a
    ``policy_classifications`` block per department.
    """
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    all_probs = [r.get("attrition_probability", 0.0) or 0.0
                 for r in departments]
    out: List[Dict[str, Any]] = []
    for r in departments:
        prob = r.get("attrition_probability", 0.0) or 0.0
        soph = _classify_under_absolute(prob, "sophisticated")
        cons = _classify_under_absolute(prob, "conservative")
        heur = _classify_under_heuristic(prob, all_probs)
        levels = [soph, cons, heur]
        unique_levels = set(levels)
        max_lvl = max(levels, key=lambda L: severity_rank[L])
        min_lvl = min(levels, key=lambda L: severity_rank[L])
        out.append({
            "department": r.get("department"),
            "attrition_probability": prob,
            "policy_classifications": {
                "sophisticated": soph,
                "conservative":  cons,
                "heuristic":     heur,
            },
            "agreement": len(unique_levels) == 1,
            "max_severity": max_lvl,
            "min_severity": min_lvl,
            # spread = how many band-steps separate max from min (0 = full
            # agreement; 1 = adjacent bands; 2+ = jump across categories).
            "severity_spread": severity_rank[max_lvl] - severity_rank[min_lvl],
        })
    return out


# =============================================================================
# Decision matrices — the "policy" the brain applies. Each cell is a named
# rule, auditable and testable in isolation. Swapping this dict is how you
# adjust the brain's hiring philosophy without retraining anything.
# =============================================================================

# Fit tier × receiving-team risk level → hire verdict + rule name + rationale.
# The "rule" string is what gets logged in the execution trace so every
# decision is traceable to a named policy.
_HIRE_DECISION_MATRIX: Dict[Tuple[str, str], Dict[str, str]] = {
    # STRONG_HIRE candidates — strongest fit
    ("STRONG_HIRE", "LOW"): {
        "verdict": "FAST_TRACK",
        "rule": "R1-strong-stable",
        "rationale": (
            "Strong candidate joining a stable team — fast-track to final "
            "round. No elevated retention risk on the receiving side."
        ),
    },
    ("STRONG_HIRE", "MEDIUM"): {
        "verdict": "HIRE_WITH_ONBOARDING_FOCUS",
        "rule": "R2-strong-watch",
        "rationale": (
            "Strong candidate, receiving team shows watch-level attrition "
            "signal. Hire, but assign a deliberate onboarding buddy and "
            "schedule 30/60/90-day satisfaction check-ins."
        ),
    },
    ("STRONG_HIRE", "HIGH"): {
        "verdict": "HIRE_WITH_RETENTION_PLAN",
        "rule": "R3-strong-high-risk",
        "rationale": (
            "Strong candidate, but receiving team is high-risk. Pair the "
            "hire with an immediate retention plan for the existing team "
            "so the new hire doesn't inherit a destabilizing environment."
        ),
    },
    ("STRONG_HIRE", "CRITICAL"): {
        "verdict": "HIRE_WITH_AGGRESSIVE_RETENTION",
        "rule": "R4-strong-critical",
        "rationale": (
            "Strong candidate into a critical-risk department. Hire ONLY "
            "if the retention program for the existing team launches "
            "first — otherwise the new hire becomes a parachute into a "
            "fire. Aggressive retention plan is non-negotiable."
        ),
    },
    # HIRE tier — solid
    ("HIRE", "LOW"): {
        "verdict": "HIRE",
        "rule": "R5-hire-stable",
        "rationale": "Solid candidate, stable team — advance to panel interview.",
    },
    ("HIRE", "MEDIUM"): {
        "verdict": "HIRE",
        "rule": "R6-hire-watch",
        "rationale": (
            "Solid candidate, watch-level team risk — advance. Note the "
            "receiving team's top driver in the offer stage so the hiring "
            "manager communicates it transparently at offer."
        ),
    },
    ("HIRE", "HIGH"): {
        "verdict": "HIRE_WITH_RETENTION_PLAN",
        "rule": "R7-hire-high-risk",
        "rationale": (
            "Solid candidate into a high-risk team. Advance, and pair "
            "the offer with a retention plan aimed at the team's top "
            "risk driver."
        ),
    },
    ("HIRE", "CRITICAL"): {
        "verdict": "CONDITIONAL_HIRE",
        "rule": "R8-hire-critical",
        "rationale": (
            "Solid fit but the receiving team is at critical attrition "
            "risk. Hire is conditional on the retention program "
            "actually launching within 60 days. Otherwise defer."
        ),
    },
    # INTERVIEW tier — borderline
    ("INTERVIEW", "LOW"): {
        "verdict": "ADVANCE_TO_INTERVIEW",
        "rule": "R9-interview-stable",
        "rationale": (
            "Viable candidate, stable team — advance. Use the interview "
            "to verify the gaps named by the Skill-Gap Analyzer."
        ),
    },
    ("INTERVIEW", "MEDIUM"): {
        "verdict": "ADVANCE_WITH_CAVEATS",
        "rule": "R10-interview-watch",
        "rationale": (
            "Viable candidate, watch-level team risk. Advance with explicit "
            "interview focus on the critical gaps + pressure-test fit "
            "for the team's current morale state."
        ),
    },
    ("INTERVIEW", "HIGH"): {
        "verdict": "DEFER",
        "rule": "R11-interview-high-risk",
        "rationale": (
            "Borderline fit into a high-risk team is the worst combination: "
            "a weak match with inevitable retention churn will underperform "
            "and likely leave within 12 months. Defer the role; fix the "
            "team first."
        ),
    },
    ("INTERVIEW", "CRITICAL"): {
        "verdict": "DEFER",
        "rule": "R12-interview-critical",
        "rationale": (
            "Borderline candidate + critical-risk team — do not advance. "
            "Focus resources on the retention program first; revisit "
            "sourcing after the team stabilizes."
        ),
    },
    # CONDITIONAL tier — closeable gap with a training plan
    ("CONDITIONAL", "LOW"): {
        "verdict": "ADVANCE_WITH_SKILL_PLAN",
        "rule": "R13-conditional-stable",
        "rationale": (
            "Fit gap is closeable. Advance contingent on a named "
            "skill-closure plan (cert path, shadow assignment, or "
            "equivalent-experience waiver)."
        ),
    },
    ("CONDITIONAL", "MEDIUM"): {
        "verdict": "ADVANCE_WITH_CAVEATS",
        "rule": "R14-conditional-watch",
        "rationale": (
            "Closeable fit gap on a watch-level team — advance cautiously. "
            "The team can't absorb a multi-month ramp on top of its own "
            "retention risk."
        ),
    },
    ("CONDITIONAL", "HIGH"): {
        "verdict": "DECLINE",
        "rule": "R15-conditional-high-risk",
        "rationale": (
            "Closeable gap on a high-risk team is not worth the compound "
            "exposure. Decline and expand sourcing."
        ),
    },
    ("CONDITIONAL", "CRITICAL"): {
        "verdict": "DECLINE",
        "rule": "R16-conditional-critical",
        "rationale": (
            "Decline. Critical-risk team cannot absorb a ramp-up candidate."
        ),
    },
    # DO_NOT_ADVANCE tier — hard decline regardless of team state
    ("DO_NOT_ADVANCE", "LOW"): {
        "verdict": "DECLINE",
        "rule": "R17-decline-stable",
        "rationale": "Candidate does not meet the bar. Decline and continue sourcing.",
    },
    ("DO_NOT_ADVANCE", "MEDIUM"): {
        "verdict": "DECLINE",
        "rule": "R18-decline-watch",
        "rationale": "Candidate does not meet the bar. Decline.",
    },
    ("DO_NOT_ADVANCE", "HIGH"): {
        "verdict": "DECLINE",
        "rule": "R19-decline-high-risk",
        "rationale": "Candidate does not meet the bar. Decline.",
    },
    ("DO_NOT_ADVANCE", "CRITICAL"): {
        "verdict": "DECLINE",
        "rule": "R20-decline-critical",
        "rationale": "Candidate does not meet the bar. Decline.",
    },
}


# =============================================================================
# CONSERVATIVE decision matrix — second brain for the multi-brain consensus.
# =============================================================================
#
# Same (fit_tier, risk_level) keys as the sophisticated matrix above, but
# with a systematically stricter policy: favours DEFER over HIRE in
# borderline cells, refuses to hire into CRITICAL-risk teams regardless
# of candidate strength, and treats ADVANCE_WITH_SKILL_PLAN as too
# aggressive for conditional-fit candidates.
#
# Philosophy: the main brain represents a sophisticated HR policy that
# balances hire velocity against team risk. The conservative brain
# represents a risk-minimizing HR policy (e.g., a CFO or audit perspective)
# that refuses to layer new hire-ramp risk on top of team-retention risk.
# Comparing the two reveals the brain's "policy-dependent" cells — cells
# where reasonable HR leaders might disagree.
_HIRE_DECISION_MATRIX_CONSERVATIVE: Dict[Tuple[str, str], Dict[str, str]] = {
    # STRONG_HIRE tier — even strong candidates face stricter gating.
    ("STRONG_HIRE", "LOW"): {
        "verdict": "FAST_TRACK", "rule": "R1c-strong-stable",
        "rationale": "Strong candidate + stable team — fast-track agreed.",
    },
    ("STRONG_HIRE", "MEDIUM"): {
        "verdict": "HIRE", "rule": "R2c-strong-watch",
        "rationale": "Strong candidate — hire at standard pace. Conservative"
                     " policy skips the ONBOARDING_FOCUS uplift since the"
                     " team is only watch-level.",
    },
    ("STRONG_HIRE", "HIGH"): {
        "verdict": "CONDITIONAL_HIRE", "rule": "R3c-strong-high-risk",
        "rationale": "Strong candidate, but conservative policy requires"
                     " the retention plan to LAUNCH before the offer —"
                     " conditional hire.",
    },
    ("STRONG_HIRE", "CRITICAL"): {
        "verdict": "DEFER", "rule": "R4c-strong-critical",
        "rationale": "Conservative policy: freeze external hires into a"
                     " CRITICAL-risk team regardless of candidate strength."
                     " Stabilize first.",
    },
    # HIRE tier — systematically demoted one rung.
    ("HIRE", "LOW"): {
        "verdict": "HIRE", "rule": "R5c-hire-stable",
        "rationale": "Solid candidate, stable team — hire.",
    },
    ("HIRE", "MEDIUM"): {
        "verdict": "ADVANCE_WITH_CAVEATS", "rule": "R6c-hire-watch",
        "rationale": "Conservative policy: watch-level team risk warrants"
                     " an interview pressure-test before offer.",
    },
    ("HIRE", "HIGH"): {
        "verdict": "CONDITIONAL_HIRE", "rule": "R7c-hire-high-risk",
        "rationale": "Conditional on retention program launching first.",
    },
    ("HIRE", "CRITICAL"): {
        "verdict": "DEFER", "rule": "R8c-hire-critical",
        "rationale": "Conservative: no external hires into CRITICAL-risk"
                     " teams. Defer until risk level is at most HIGH.",
    },
    # INTERVIEW tier — conservative pushes up defer/decline.
    ("INTERVIEW", "LOW"): {
        "verdict": "ADVANCE_WITH_CAVEATS", "rule": "R9c-interview-stable",
        "rationale": "Borderline fit — interview with explicit focus on"
                     " the critical gaps the analyzer flagged.",
    },
    ("INTERVIEW", "MEDIUM"): {
        "verdict": "DEFER", "rule": "R10c-interview-watch",
        "rationale": "Borderline fit + watch-level team — defer; fix team"
                     " first.",
    },
    ("INTERVIEW", "HIGH"): {
        "verdict": "DECLINE", "rule": "R11c-interview-high-risk",
        "rationale": "Borderline fit + high-risk team — decline.",
    },
    ("INTERVIEW", "CRITICAL"): {
        "verdict": "DECLINE", "rule": "R12c-interview-critical",
        "rationale": "Decline. Fix the team before sourcing.",
    },
    # CONDITIONAL tier — conservative will not bet on training plans.
    ("CONDITIONAL", "LOW"): {
        "verdict": "DEFER", "rule": "R13c-conditional-stable",
        "rationale": "Conservative: won't commit to a skill-closure plan;"
                     " expand sourcing for a stronger baseline fit.",
    },
    ("CONDITIONAL", "MEDIUM"): {
        "verdict": "DECLINE", "rule": "R14c-conditional-watch",
        "rationale": "Decline.",
    },
    ("CONDITIONAL", "HIGH"): {
        "verdict": "DECLINE", "rule": "R15c-conditional-high-risk",
        "rationale": "Decline.",
    },
    ("CONDITIONAL", "CRITICAL"): {
        "verdict": "DECLINE", "rule": "R16c-conditional-critical",
        "rationale": "Decline.",
    },
    # DO_NOT_ADVANCE — unchanged; a hard floor is a hard floor.
    ("DO_NOT_ADVANCE", "LOW"): {
        "verdict": "DECLINE", "rule": "R17c-decline-stable",
        "rationale": "Below bar — decline.",
    },
    ("DO_NOT_ADVANCE", "MEDIUM"): {
        "verdict": "DECLINE", "rule": "R18c-decline-watch",
        "rationale": "Below bar — decline.",
    },
    ("DO_NOT_ADVANCE", "HIGH"): {
        "verdict": "DECLINE", "rule": "R19c-decline-high-risk",
        "rationale": "Below bar — decline.",
    },
    ("DO_NOT_ADVANCE", "CRITICAL"): {
        "verdict": "DECLINE", "rule": "R20c-decline-critical",
        "rationale": "Below bar — decline.",
    },
}


# Verdict "families" for the consensus classifier. Two brains that return
# different verdicts within the same family count as agreeing — it's only
# cross-family disagreement (e.g., HIRE vs DEFER) that we count as a real
# split.
_VERDICT_FAMILY: Dict[str, str] = {
    "FAST_TRACK": "HIRE",
    "HIRE": "HIRE",
    "HIRE_WITH_ONBOARDING_FOCUS": "HIRE",
    "HIRE_WITH_RETENTION_PLAN": "HIRE",
    "HIRE_WITH_AGGRESSIVE_RETENTION": "HIRE",
    "CONDITIONAL_HIRE": "CONDITIONAL",
    "ADVANCE_TO_INTERVIEW": "ADVANCE",
    "ADVANCE_WITH_CAVEATS": "ADVANCE",
    "ADVANCE_WITH_SKILL_PLAN": "ADVANCE",
    "DEFER": "DEFER",
    "DECLINE": "DECLINE",
    "INCONCLUSIVE": "ESCALATE",
}


# =============================================================================
# HEURISTIC BASELINE — third brain for the multi-brain consensus.
# =============================================================================
#
# A deliberately naive 2-signal decision procedure. Uses only the composite
# fit score and the department-level risk level — no seniority alignment,
# no years-of-experience gap, no critical-cert gating. This is the "what
# would a spreadsheet tell us?" baseline. It agrees with the sophisticated
# brain in the easy cases (fit clearly high on a clearly stable team; fit
# clearly low) and diverges on the subtle ones — which is exactly the
# signal consensus analysis exists to surface.
#
# Returns the same shape as a _HIRE_DECISION_MATRIX cell so the rest of
# the consensus logic can treat all three brains uniformly.
def _heuristic_hire_verdict(fit_score: Optional[float],
                             risk_level: str) -> Dict[str, str]:
    """Minimum-viable 2-signal heuristic — the "dumb spreadsheet" brain."""
    if fit_score is None:
        return {
            "verdict": "INCONCLUSIVE",
            "rule": "H0-no-signal",
            "rationale": "Heuristic baseline: no fit score available.",
        }
    risk = (risk_level or "MEDIUM").upper()
    if fit_score >= 75 and risk in ("LOW", "MEDIUM"):
        return {
            "verdict": "HIRE",
            "rule": "H1-strong-stable",
            "rationale": (
                f"Heuristic: fit {fit_score:.0f} ≥ 75 with {risk} team "
                "risk — hire."
            ),
        }
    if fit_score >= 75:
        return {
            "verdict": "HIRE_WITH_RETENTION_PLAN",
            "rule": "H2-strong-risky",
            "rationale": (
                f"Heuristic: fit {fit_score:.0f} ≥ 75 but team risk "
                f"{risk} — hire conditional on retention program."
            ),
        }
    if fit_score >= 60 and risk == "LOW":
        return {
            "verdict": "ADVANCE_TO_INTERVIEW",
            "rule": "H3-mid-stable",
            "rationale": (
                f"Heuristic: fit {fit_score:.0f} (mid) with stable team"
                " — advance to interview."
            ),
        }
    if fit_score >= 60:
        return {
            "verdict": "DEFER",
            "rule": "H4-mid-risky",
            "rationale": (
                f"Heuristic: fit {fit_score:.0f} (mid) with {risk} team"
                " risk — defer."
            ),
        }
    return {
        "verdict": "DECLINE",
        "rule": "H5-below-bar",
        "rationale": (
            f"Heuristic: fit {fit_score:.0f} < 60 — below hire bar."
        ),
    }


# How urgent is the retention program on the receiving team for each verdict?
_RETENTION_URGENCY_BY_VERDICT: Dict[str, str] = {
    "FAST_TRACK": "standard",
    "HIRE": "standard",
    "HIRE_WITH_ONBOARDING_FOCUS": "elevated",
    "HIRE_WITH_RETENTION_PLAN": "high",
    "HIRE_WITH_AGGRESSIVE_RETENTION": "critical",
    "CONDITIONAL_HIRE": "critical",
    "ADVANCE_TO_INTERVIEW": "standard",
    "ADVANCE_WITH_CAVEATS": "elevated",
    "ADVANCE_WITH_SKILL_PLAN": "standard",
    "DEFER": "critical",
    "DECLINE": "high",
}


# Thresholds for brain-level confidence blending. A memo's confidence is
# a min across talent analysis, workforce analysis, uncertainty quantifier,
# and trajectory stability — the weakest link dominates.
_CONFIDENCE_WEIGHTS = {
    "talent": 0.25,
    "workforce": 0.25,
    "uncertainty": 0.10,   # MC Dropout spread — Bayesian approx (demoted)
    "conformal": 0.25,     # Split Conformal interval width — the distribution-
                             # free, guaranteed-coverage signal. Promoted to
                             # dominate the calibration side because it is
                             # mathematically stronger than MC-Dropout.
    "trajectory": 0.15,    # stable/improving = higher, worsening = lower
}


# =============================================================================
# Data classes for structured memo output
# =============================================================================

@dataclass
class ExecutionStep:
    """One tool call in the brain's execution trace."""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    had_error: bool = False
    error_message: Optional[str] = None


@dataclass
class MemoSection:
    """One section of a memo — header + narrative + key metrics."""
    title: str
    narrative: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMemo:
    """A structured, machine- and human-readable memo."""
    memo_type: str               # "hire_decision" / "risk_scan" / etc.
    headline: str                # one-sentence TL;DR
    sections: List[MemoSection] = field(default_factory=list)
    verdict: Optional[str] = None
    rule_applied: Optional[str] = None
    confidence: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[ExecutionStep] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    brain_version: str = "1.0-proprietary"
    tool_layer_version: str = "2026-04-18"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memo_type": self.memo_type,
            "headline": self.headline,
            "verdict": self.verdict,
            "rule_applied": self.rule_applied,
            "sections": [asdict(s) for s in self.sections],
            "confidence": self.confidence,
            "metrics": self.metrics,
            "execution_trace": [asdict(e) for e in self.execution_trace],
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "brain_version": self.brain_version,
            "tool_layer_version": self.tool_layer_version,
        }

    def to_markdown(self) -> str:
        """Render as a markdown memo a human can read."""
        md = [f"# {self.headline}", ""]
        if self.verdict:
            md.append(f"**Verdict:** `{self.verdict}` "
                      f"(rule: `{self.rule_applied}`)")
            md.append("")
        for s in self.sections:
            md.append(f"## {s.title}")
            md.append("")
            md.append(s.narrative)
            md.append("")
            if s.metrics:
                md.append("**Key numbers:**")
                for k, v in s.metrics.items():
                    md.append(f"- {k}: {v}")
                md.append("")
        if self.confidence:
            md.append(f"---")
            md.append(f"*Brain confidence: "
                      f"{self.confidence.get('level', '?')} "
                      f"({self.confidence.get('score', 0):.2f}). "
                      f"Executed in {self.total_elapsed_ms:.0f} ms via "
                      f"{len(self.execution_trace)} tool calls. "
                      f"Brain {self.brain_version}.*")
        # Human-in-the-Lead footer — learning version + review priority.
        # Only surfaced when the memo carries learning metadata.
        lv = self.metrics.get("learning_version") if self.metrics else None
        if lv:
            rp = self.metrics.get("review_priority")
            h = self.metrics.get("feedback_state_hash")
            adj = self.metrics.get("adjustments_applied") or []
            parts = [f"Learning: **{lv}**"]
            if h:
                parts.append(f"state hash `{h}`")
            if rp:
                reasons = self.metrics.get("review_priority_reasons") or []
                parts.append(
                    f"review priority **{rp}**"
                    + (f" ({', '.join(reasons)})" if reasons else "")
                )
            if adj:
                parts.append(f"{len(adj)} adjustment(s) applied")
            md.append(f"*{' · '.join(parts)}.*")
        return "\n".join(md)


# =============================================================================
# The Brain
# =============================================================================

class WorkforceIntelligenceBrain:
    """
    Deterministic, LLM-free orchestrator for the Workforce Intelligence
    System. Composes 21 tools from talent_tools.py + forecast_tools.py
    into five end-to-end workflows with fully templated narrative output.

    Usage
    -----
    >>> from models.ner_model import EnsembleNEREngine
    >>> from models.sbert_matcher import SBERTMatcher
    >>> from models.bilstm_model import ForecastingEngine
    >>> from tools.talent_tools import set_ner_engine, set_sbert_matcher
    >>> from tools.forecast_tools import set_forecast_engine, set_workforce_data
    >>> from brain import WorkforceIntelligenceBrain
    >>> # ... register the DL models ...
    >>> brain = WorkforceIntelligenceBrain()
    >>> memo = brain.joint_hire_analysis(resume, job, "Engineering")
    >>> print(memo.to_markdown())
    """

    def __init__(self, verbose: bool = False,
                 enable_learning: Optional[bool] = None,
                 learning_engine: Any = None):
        """
        Parameters
        ----------
        verbose          : log tool calls to stdout as they execute.
        enable_learning  : toggle the Human-in-the-Lead feedback layer.
                           True (default, honours WORKFORCE_BRAIN_LEARNING
                           env var) enables adaptive mode — feedback store
                           is consulted on each decision.
                           False forces static mode (byte-identical to the
                           pre-feedback brain) for regulatory snapshots
                           and reproducibility tests.
        learning_engine  : optional pre-built LearningEngine for testing.
                           When None and learning is enabled, a default
                           engine is lazily instantiated on first use.
        """
        self.verbose = verbose
        self._trace: List[ExecutionStep] = []

        # Resolve the learning toggle. Priority: explicit arg > env var.
        if enable_learning is None:
            try:
                from feedback import is_learning_enabled_default
                enable_learning = is_learning_enabled_default()
            except ImportError:
                enable_learning = False
        self.learning_enabled = bool(enable_learning)

        # Engine: lazy-load on first decision when enabled and not injected.
        self._engine = learning_engine
        self._adjustments_cache: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------
    # Internal — tool invocation with timing + error trapping
    # -----------------------------------------------------------------

    def _call(self, tool_name: str, fn, **kwargs) -> Dict[str, Any]:
        """Call a @tool-decorated function, parse JSON, log to trace."""
        t0 = time.perf_counter()
        try:
            raw = fn.run(**kwargs)
            result = json.loads(raw)
            err = "error" in result
            elapsed = (time.perf_counter() - t0) * 1000
            self._trace.append(ExecutionStep(
                tool=tool_name, args=kwargs, elapsed_ms=elapsed,
                had_error=err,
                error_message=result.get("error") if err else None,
            ))
            if self.verbose:
                status = "ERROR" if err else "OK"
                print(f"  [{status:5s}] {tool_name}({kwargs}) — {elapsed:.1f}ms")
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            self._trace.append(ExecutionStep(
                tool=tool_name, args=kwargs, elapsed_ms=elapsed,
                had_error=True, error_message=str(e),
            ))
            if self.verbose:
                print(f"  [EXCEP] {tool_name} raised {type(e).__name__}: {e}")
            return {"error": str(e)}

    def _reset_trace(self) -> None:
        self._trace = []

    # -----------------------------------------------------------------
    # Human-in-the-Lead continual learning — lazy engine init +
    # adjustment lookup + memo metadata stamping. All no-ops when
    # learning_enabled is False (static mode).
    # -----------------------------------------------------------------

    def _get_adjustments(self) -> Optional[Dict[str, Any]]:
        """Lazily fetch adjustments from the LearningEngine.

        Returns None when learning is disabled (static mode) OR when the
        feedback store is empty. Either case → brain behaves identically
        to the pre-feedback version.
        """
        if not self.learning_enabled:
            return None
        if self._engine is None:
            try:
                from feedback import LearningEngine
                self._engine = LearningEngine()
            except ImportError:
                return None
        if self._adjustments_cache is None:
            self._adjustments_cache = self._engine.compute_adjustments()
        return self._adjustments_cache

    def _apply_feedback_adjustments(self, decision: Dict[str, Any]
                                      ) -> Dict[str, Any]:
        """If a rule override exists for (fit_tier, risk_level), swap the
        verdict/rule/rationale and record the adjustment. Returns the
        decision dict (possibly modified in-place with metadata)."""
        adjustments = self._get_adjustments()
        decision.setdefault("adjustments_applied", [])
        if not adjustments:
            return decision
        ft = decision.get("fit_tier")
        rl = decision.get("risk_level")
        if not ft or not rl:
            return decision
        override = adjustments.get("rule_overrides", {}).get(f"{ft}|{rl}")
        if not override:
            return decision

        original_verdict = decision["verdict"]
        original_rule = decision["rule"]
        decision["verdict"] = override["new_verdict"]
        decision["rule"] = f"{original_rule}+ADJ"
        decision["rationale"] = (
            f"{decision['rationale']} "
            f"[Adjusted via {override['source']} — original verdict was "
            f"'{original_verdict}', overridden to '{override['new_verdict']}' "
            f"based on {override.get('evidence_votes', 1)} human "
            f"correction(s).]"
        )
        decision["adjustments_applied"].append({
            "type": "rule_override",
            "original_verdict": original_verdict,
            "new_verdict": override["new_verdict"],
            "source": override["source"],
            "evidence_votes": override.get("evidence_votes", 1),
            "event_ids": override.get("event_ids", []),
        })
        return decision

    def _review_priority_info(self, *,
                                confidence_level: Optional[str] = None,
                                conformal_width: Optional[float] = None,
                                consensus_class: Optional[str] = None
                                ) -> Optional[Dict[str, Any]]:
        """Compute active-learning review priority for a memo. When
        learning is disabled returns None (priority stamp omitted)."""
        if not self.learning_enabled:
            return None
        if self._engine is None:
            try:
                from feedback import LearningEngine
                self._engine = LearningEngine()
            except ImportError:
                return None
        return self._engine.get_review_priority(
            confidence_level=confidence_level,
            conformal_width=conformal_width,
            consensus_class=consensus_class,
        )

    def _stamp_memo_metadata(self, memo: "AgentMemo",
                              priority_info: Optional[Dict[str, Any]] = None,
                              adjustments_applied: Optional[List] = None
                              ) -> "AgentMemo":
        """Attach memo_id, learning_version, feedback_state_hash,
        review_priority, and adjustments_applied to the memo metrics
        dict. In static mode learning_version is 'static' and the hash
        is null."""
        memo.metrics = memo.metrics or {}
        memo.metrics["memo_id"] = _new_memo_id()
        if self.learning_enabled:
            adj = self._get_adjustments()
            if adj:
                memo.metrics["learning_version"] = adj.get("learning_version")
                memo.metrics["feedback_state_hash"] = adj.get("state_hash")
                memo.metrics["n_feedback_events"] = adj.get("n_events_total", 0)
            else:
                memo.metrics["learning_version"] = "v0-empty"
                memo.metrics["feedback_state_hash"] = None
                memo.metrics["n_feedback_events"] = 0
        else:
            memo.metrics["learning_version"] = "static"
            memo.metrics["feedback_state_hash"] = None
            memo.metrics["n_feedback_events"] = 0
        memo.metrics["adjustments_applied"] = list(adjustments_applied or [])
        if priority_info:
            memo.metrics["review_priority"] = priority_info["priority"]
            memo.metrics["review_priority_reasons"] = priority_info["reasons"]
        else:
            memo.metrics["review_priority"] = None
            memo.metrics["review_priority_reasons"] = []
        return memo

    # =================================================================
    # WORKFLOW 1 — Joint hire analysis
    # =================================================================

    def joint_hire_analysis(self, resume_text: str, job_text: str,
                             department: str) -> AgentMemo:
        """
        Full joint analysis: Agent 1 scores the candidate, Agent 2
        analyzes the receiving team, then the brain synthesizes a
        unified hire / retention / backfill memo.

        Replicates CrewAI's `create_combined_task` joint synthesis mode
        — entirely without an LLM.
        """
        self._reset_trace()
        t_start = time.perf_counter()

        from tools.talent_tools import analyze_skill_gap
        from tools.forecast_tools import (
            analyze_workforce_risk, quantify_risk_uncertainty,
            conformal_prediction_interval,
            forecast_risk_trajectory, find_internal_mobility_candidates,
        )

        # --- Agent 1 side: skill-gap analysis ---
        talent = self._call(
            "analyze_skill_gap", analyze_skill_gap,
            candidate_text=resume_text, job_text=job_text,
        )
        if "error" in talent:
            return self._error_memo("hire_decision",
                                     f"Talent analysis failed: {talent['error']}")

        # --- Agent 2 side: workforce risk + uncertainty + trajectory + bench ---
        workforce = self._call(
            "analyze_workforce_risk", analyze_workforce_risk,
            department=department,
        )
        if "error" in workforce:
            return self._error_memo("hire_decision",
                                     f"Workforce analysis failed: {workforce['error']}")

        uncertainty = self._call(
            "quantify_risk_uncertainty", quantify_risk_uncertainty,
            department=department, n_samples=30,
        )
        # Split Conformal Prediction — distribution-free coverage guarantee.
        # Pairs with MC Dropout so the memo can surface both the Bayesian
        # credible interval and the conformal guaranteed-coverage interval.
        # The gap between them is itself a diagnostic of model calibration.
        conformal = self._call(
            "conformal_prediction_interval", conformal_prediction_interval,
            department=department, alpha=0.10,
        )
        trajectory = self._call(
            "forecast_risk_trajectory", forecast_risk_trajectory,
            department=department, horizon_months=12,
        )
        bench = self._call(
            "find_internal_mobility_candidates",
            find_internal_mobility_candidates,
            for_department=department, top_n=5,
        )

        # --- Deterministic synthesis ---
        decision = self._synthesize_hire_decision(
            talent, workforce, uncertainty, conformal, trajectory,
        )
        # Apply any human-approved feedback-driven adjustments BEFORE
        # building the retention plan (which depends on the verdict).
        # In static mode or with an empty store this is a no-op.
        decision = self._apply_feedback_adjustments(decision)
        retention = self._build_retention_plan(
            workforce, urgency=_RETENTION_URGENCY_BY_VERDICT.get(
                decision["verdict"], "standard",
            ),
        )
        backfill = self._build_backfill_pipeline(talent, workforce, bench)
        confidence = self._blend_confidence(
            talent, workforce, uncertainty, conformal, trajectory,
        )
        # Counterfactual flip explainer — only fires for DEFER/DECLINE.
        # Positive-verdict memos skip this (returns None) so the happy
        # path stays unchanged byte-for-byte.
        counterfactual = self._counterfactual_flip_analysis(
            talent, workforce, decision, department,
        )

        # --- Render structured memo ---
        memo = self._render_hire_memo(
            talent, workforce, decision, retention, backfill, confidence,
            uncertainty, conformal, trajectory, department,
            counterfactual=counterfactual,
        )
        memo.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        memo.execution_trace = list(self._trace)
        # Stamp learning metadata (memo_id, feedback_state_hash, priority).
        priority = self._review_priority_info(
            confidence_level=confidence.get("level"),
            conformal_width=(conformal or {}).get("interval_width"),
        )
        memo = self._stamp_memo_metadata(
            memo,
            priority_info=priority,
            adjustments_applied=decision.get("adjustments_applied", []),
        )
        return memo

    # =================================================================
    # WORKFLOW 2 — Company-wide workforce risk scan
    # =================================================================

    def workforce_risk_scan(self,
                             departments: Optional[List[str]] = None,
                             top_n_actions: int = 3) -> AgentMemo:
        """
        Cross-department risk scan. Ranks departments by risk, names
        the top driver per department, recommends the top-N action
        items, and produces a one-page executive summary.
        """
        self._reset_trace()
        t_start = time.perf_counter()

        from tools.forecast_tools import (
            predict_all_departments, analyze_workforce_risk,
            quantify_risk_uncertainty, conformal_prediction_interval,
        )

        ranked = self._call("predict_all_departments", predict_all_departments)
        if isinstance(ranked, dict) and "error" in ranked:
            return self._error_memo("risk_scan", ranked["error"])

        # Filter to requested depts if provided
        if departments:
            ranked = [r for r in ranked if r.get("department") in departments]

        # Deep-dive top 3 by risk
        top = ranked[:3]
        details = []
        for r in top:
            d = r["department"]
            analysis = self._call(
                "analyze_workforce_risk", analyze_workforce_risk,
                department=d,
            )
            uncertainty = self._call(
                "quantify_risk_uncertainty", quantify_risk_uncertainty,
                department=d, n_samples=20,
            )
            conformal = self._call(
                "conformal_prediction_interval",
                conformal_prediction_interval,
                department=d, alpha=0.10,
            )
            details.append({"dept": d, "ranked": r,
                            "analysis": analysis,
                            "uncertainty": uncertainty,
                            "conformal": conformal})

        action_items = self._derive_scan_actions(details, top_n=top_n_actions)

        memo = self._render_scan_memo(ranked, details, action_items)
        memo.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        memo.execution_trace = list(self._trace)
        memo = self._stamp_memo_metadata(memo)
        return memo

    # =================================================================
    # WORKFLOW 3 — Quarterly retention plan (budget-constrained)
    # =================================================================

    def quarterly_retention_plan(self, budget_usd: int = 500_000,
                                  focus_depts: Optional[List[str]] = None
                                  ) -> AgentMemo:
        """
        Budget-bounded portfolio optimization wrapped in a board-ready
        memo. Calls the knapsack optimizer and then synthesizes prose
        with rollup tables.
        """
        self._reset_trace()
        t_start = time.perf_counter()

        from tools.forecast_tools import optimize_intervention_portfolio

        focus_str = ",".join(focus_depts) if focus_depts else ""
        result = self._call(
            "optimize_intervention_portfolio",
            optimize_intervention_portfolio,
            budget_usd=int(budget_usd), focus_depts=focus_str,
        )
        if "error" in result:
            return self._error_memo("retention_plan", result["error"])

        memo = self._render_plan_memo(result, budget_usd)
        memo.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        memo.execution_trace = list(self._trace)
        memo = self._stamp_memo_metadata(memo)
        return memo

    # =================================================================
    # WORKFLOW 4 — Rank candidates for a req
    # =================================================================

    def rank_candidates_for_req(self, job_text: str,
                                 candidates: List[Dict[str, str]]
                                 ) -> AgentMemo:
        """
        Thin wrapper around Agent 1's `rank_candidates_for_job` tool,
        but with the brain producing structured memo output on top.
        """
        self._reset_trace()
        t_start = time.perf_counter()

        from tools.talent_tools import rank_candidates_for_job

        result = self._call(
            "rank_candidates_for_job", rank_candidates_for_job,
            job_text=job_text,
            candidates_json=json.dumps(candidates),
        )
        if "error" in result:
            return self._error_memo("shortlist", result["error"])

        memo = self._render_shortlist_memo(result)
        memo.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        memo.execution_trace = list(self._trace)
        memo = self._stamp_memo_metadata(memo)
        return memo

    # =================================================================
    # WORKFLOW 5 — Match one candidate across multiple reqs
    # =================================================================

    def match_candidate_across_reqs(self, resume_text: str,
                                      jobs: List[Dict[str, str]]
                                      ) -> AgentMemo:
        """Multi-job triage for internal-mobility decisions."""
        self._reset_trace()
        t_start = time.perf_counter()

        from tools.talent_tools import triage_candidate_across_jobs

        result = self._call(
            "triage_candidate_across_jobs", triage_candidate_across_jobs,
            candidate_text=resume_text,
            jobs_json=json.dumps(jobs),
        )
        if "error" in result:
            return self._error_memo("triage", result["error"])

        memo = self._render_triage_memo(result)
        memo.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        memo.execution_trace = list(self._trace)
        memo = self._stamp_memo_metadata(memo)
        return memo

    # =================================================================
    # WORKFLOW 6 — Multi-brain consensus (frontier safety pattern)
    # =================================================================

    def multi_brain_consensus(self, resume_text: str, job_text: str,
                               department: str) -> AgentMemo:
        """
        Run THREE independent decision procedures on the same hire
        and compare their verdicts. Produces a ConsensusMemo that flags
        strong-consensus (all 3 agree), majority-consensus (2/3), and
        no-consensus (all disagree) cases.

        The three brains:
          1. **Sophisticated** — the default `_HIRE_DECISION_MATRIX`
             (20 cells, balanced hire-velocity × team-risk policy).
          2. **Conservative** — `_HIRE_DECISION_MATRIX_CONSERVATIVE`
             (same cells, risk-minimizing policy; refuses hires into
             CRITICAL teams regardless of candidate strength).
          3. **Heuristic baseline** — `_heuristic_hire_verdict`
             (naive 2-signal: fit_score × risk_level only; no seniority,
             no experience gap, no cert gating).

        Pattern reference: ensemble safety / multiple-decision-procedures
        framing from the 2024+ AI safety literature (e.g., Constitutional
        AI's critic + revision loop; the NeurIPS 2024 "Multiple
        Specialized Experts" consensus work). Disagreement between
        independent procedures is itself a first-class diagnostic —
        "this case is in the policy-dependent zone, escalate to a human."

        Side benefit: provides auditability under the EU AI Act Art. 14
        requirement for human oversight. When brains disagree, the memo
        can be automatically routed to a senior HR reviewer.
        """
        self._reset_trace()
        t_start = time.perf_counter()

        # Re-use the joint-hire pipeline to produce the full candidate +
        # workforce analysis. This is our "sophisticated" run.
        sophisticated_memo = self.joint_hire_analysis(
            resume_text, job_text, department,
        )
        # Rebuild the intermediate decision inputs so the conservative and
        # heuristic brains can be applied to the SAME facts without re-
        # running the tool layer. Pull from the sophisticated memo's
        # metrics + rule_applied.
        fit_score = sophisticated_memo.sections[0].metrics.get("fit_score")
        fit_tier = sophisticated_memo.sections[0].metrics.get("fit_tier")
        risk_level = sophisticated_memo.sections[0].metrics.get("risk_level")
        sophisticated_verdict = sophisticated_memo.verdict
        sophisticated_rule = sophisticated_memo.rule_applied

        # Brain 2 — Conservative matrix lookup on the same fit × risk cell.
        conservative_cell = _HIRE_DECISION_MATRIX_CONSERVATIVE.get(
            (fit_tier, risk_level)
        ) or {
            "verdict": "INCONCLUSIVE",
            "rule": "R99c-no-matching-rule",
            "rationale": "No conservative rule for this (fit, risk) pair.",
        }

        # Brain 3 — Heuristic baseline.
        heuristic_cell = _heuristic_hire_verdict(fit_score, risk_level)

        # Gather the three verdicts for consensus classification.
        verdicts = [
            {
                "brain": "Sophisticated",
                "verdict": sophisticated_verdict,
                "rule": sophisticated_rule,
                "rationale": (sophisticated_memo.sections[0].narrative
                              if sophisticated_memo.sections else ""),
                "family": _VERDICT_FAMILY.get(sophisticated_verdict or "", "?"),
            },
            {
                "brain": "Conservative",
                "verdict": conservative_cell["verdict"],
                "rule": conservative_cell["rule"],
                "rationale": conservative_cell["rationale"],
                "family": _VERDICT_FAMILY.get(conservative_cell["verdict"], "?"),
            },
            {
                "brain": "Heuristic baseline",
                "verdict": heuristic_cell["verdict"],
                "rule": heuristic_cell["rule"],
                "rationale": heuristic_cell["rationale"],
                "family": _VERDICT_FAMILY.get(heuristic_cell["verdict"], "?"),
            },
        ]

        # Consensus classification — family-level not verdict-level so that
        # (HIRE, HIRE_WITH_RETENTION_PLAN, HIRE) counts as agreement.
        families = [v["family"] for v in verdicts]
        unique_families = set(families)
        if len(unique_families) == 1:
            consensus_class = "STRONG_CONSENSUS"
            consensus_action = (
                f"All three brains converge on the {families[0]} family. "
                "Execute the sophisticated brain's specific verdict with "
                "normal review; no escalation required."
            )
        elif len(unique_families) == 2:
            # 2-1 split. Pick the majority family.
            from collections import Counter
            counts = Counter(families)
            majority_family, _ = counts.most_common(1)[0]
            dissenters = [v for v in verdicts
                          if v["family"] != majority_family]
            consensus_class = "MAJORITY_CONSENSUS"
            consensus_action = (
                f"Majority ({3 - len(dissenters)}/3) land on "
                f"{majority_family}. "
                + ", ".join(f"{d['brain']} dissents to {d['family']}"
                             for d in dissenters)
                + ". Recommend: senior-reviewer sign-off before executing."
            )
        else:
            consensus_class = "NO_CONSENSUS"
            consensus_action = (
                "All three brains return different verdict families "
                f"({', '.join(sorted(unique_families))}). "
                "This case is in the policy-dependent zone — ESCALATE to "
                "a human hiring committee before any action."
            )

        memo = self._render_consensus_memo(
            verdicts=verdicts,
            consensus_class=consensus_class,
            consensus_action=consensus_action,
            sophisticated_memo=sophisticated_memo,
            department=department,
            fit_score=fit_score,
            fit_tier=fit_tier,
            risk_level=risk_level,
        )
        memo.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        # Inherit the sophisticated trace (dominant tool-call cost) so the
        # consensus memo reports realistic execution cost.
        memo.execution_trace = list(sophisticated_memo.execution_trace)
        # Review priority for consensus memos is driven by the consensus
        # class itself — split/no-consensus are the highest-value review
        # items. Confidence / conformal still contribute.
        priority = self._review_priority_info(
            confidence_level=(sophisticated_memo.confidence or {}).get("level"),
            conformal_width=(sophisticated_memo.sections[0].metrics
                              .get("conformal_width")
                              if sophisticated_memo.sections else None),
            consensus_class=consensus_class,
        )
        memo = self._stamp_memo_metadata(memo, priority_info=priority)
        return memo

    # =================================================================
    # INTERNAL — decision logic
    # =================================================================

    def _synthesize_hire_decision(self, talent: Dict, workforce: Dict,
                                    uncertainty: Dict, conformal: Dict,
                                    trajectory: Dict
                                    ) -> Dict[str, Any]:
        """
        Apply the fit × risk decision matrix. Every verdict traces to a
        named rule in `_HIRE_DECISION_MATRIX`.
        """
        fit = talent.get("fit_score", {}) or {}
        fit_tier = fit.get("recommendation_tier", "DO_NOT_ADVANCE")
        risk_level = workforce.get("risk_level", "MEDIUM")

        cell = _HIRE_DECISION_MATRIX.get((fit_tier, risk_level))
        if cell is None:
            cell = {
                "verdict": "INCONCLUSIVE",
                "rule": "R99-no-matching-rule",
                "rationale": (
                    f"No decision rule for fit_tier={fit_tier}, "
                    f"risk_level={risk_level}. Escalate to human review."
                ),
            }

        # Bolt on the quantitative rationale.
        years = talent.get("years_experience_gap", {}) or {}
        senior = talent.get("seniority_alignment", {}) or {}
        modifiers = []
        if years.get("meets_requirement") is False:
            modifiers.append(
                f"years-of-experience shortfall "
                f"({years.get('candidate_has')} vs {years.get('required')})"
            )
        if senior.get("alignment") == "candidate_over_leveled":
            modifiers.append("seniority above target — comp/scope risk")
        if (trajectory.get("trajectory_direction") == "worsening"
                and cell["verdict"].startswith(("HIRE", "FAST"))):
            modifiers.append(
                "trajectory is worsening — treat retention-plan launch "
                "as a prerequisite for the offer, not a follow-up"
            )
        uncert_level = uncertainty.get("uncertainty_level")
        if uncert_level == "low":
            modifiers.append(
                "model uncertainty is high — recommend a second-source "
                "verification (manager reference, live skills test)"
            )

        # MC/Conformal gap diagnostic — when MC Dropout is tight but Split
        # Conformal is wide, the model is internally consistent but poorly
        # calibrated against held-out reality. This is precisely the finding
        # documented in WRITEUP §16.2 and the reason the memo cites BOTH.
        conf_width = None
        if isinstance(conformal, dict) and "error" not in conformal:
            conf_width = conformal.get("interval_width")
        mc_std = uncertainty.get("std") if isinstance(uncertainty, dict) else None
        if (conf_width is not None and conf_width > 0.7
                and mc_std is not None and mc_std < 0.05):
            modifiers.append(
                "MC-Dropout vs Conformal interval gap — model is "
                "internally consistent but held-out calibration is weak; "
                "weight human judgment accordingly"
            )

        return {
            **cell,
            "fit_tier": fit_tier,
            "fit_score": fit.get("composite_fit_score"),
            "risk_level": risk_level,
            "attrition_probability": workforce.get("attrition_probability"),
            "modifiers": modifiers,
            "uncertainty_level": uncert_level,
            "conformal_width": conf_width,
            "trajectory_direction": trajectory.get("trajectory_direction"),
        }

    def _build_retention_plan(self, workforce: Dict, urgency: str) -> Dict:
        """Pull the retention plan already computed by the 18-signal
        analyzer and layer the urgency flag on top."""
        plan = workforce.get("retention_plan", []) or []
        cost_total = sum(p.get("est_program_cost_usd", 0) for p in plan)
        replacement = workforce.get("replacement_cost", {}) or {}
        repl_cost = replacement.get("estimated_replacement_cost_usd", 0)
        roi = (repl_cost / cost_total) if cost_total > 0 else None
        return {
            "urgency": urgency,
            "actions": plan,
            "total_program_cost_usd": int(cost_total),
            "replacement_exposure_usd": int(repl_cost),
            "roi_multiplier": round(roi, 1) if roi is not None else None,
        }

    def _build_backfill_pipeline(self, talent: Dict, workforce: Dict,
                                   bench: Dict) -> Dict:
        """Combine Agent 1's named gaps with Agent 2's internal mobility
        bench to produce a concrete backfill plan."""
        top_gaps = talent.get("top_gaps", []) or []
        candidates = (bench.get("top_candidates") or []) if bench else []

        # Group internal candidates by originating department
        by_source_dept: Dict[str, List[Dict]] = {}
        for c in candidates:
            by_source_dept.setdefault(c.get("from_department", "?"),
                                       []).append(c)

        # Map gaps to sourcing channels (domain heuristics)
        external_channels = []
        for g in top_gaps[:5]:
            req = g.get("requirement", "").lower()
            if any(k in req for k in ("api 57", "api 51", "api 1169",
                                        "api 653", "api 571", "api-")):
                external_channels.append({
                    "for_gap": g["requirement"],
                    "channel": "API Events network + API ICP-certified pool",
                })
            elif any(k in req for k in ("nace", "cip", "coating")):
                external_channels.append({
                    "for_gap": g["requirement"],
                    "channel": "NACE/AMPP conferences + AMPP CIP-certified pool",
                })
            elif any(k in req for k in ("nebosh", "csp", "osha")):
                external_channels.append({
                    "for_gap": g["requirement"],
                    "channel": "BCSP / NEBOSH accredited-provider network",
                })
            elif any(k in req for k in ("gwo", "wind")):
                external_channels.append({
                    "for_gap": g["requirement"],
                    "channel": "GWO-certified offshore wind network",
                })
            elif any(k in req for k in ("scada", "plc", "dcs",
                                          "instrumentation")):
                external_channels.append({
                    "for_gap": g["requirement"],
                    "channel": "Rockwell / Siemens / Emerson automation alumni",
                })
            elif any(k in req for k in ("subsea", "riser", "offshore",
                                          "fpso")):
                external_channels.append({
                    "for_gap": g["requirement"],
                    "channel": "Offshore / subsea specialist recruiter network",
                })
            else:
                external_channels.append({
                    "for_gap": g["requirement"],
                    "channel": "Standard direct-sourcing + LinkedIn targeted outreach",
                })

        return {
            "internal_candidates": candidates[:5],
            "internal_sources": list(by_source_dept.keys()),
            "external_channels": external_channels,
            "total_internal_bench": bench.get("total_candidates_considered", 0),
        }

    # =================================================================
    # Counterfactual verdict explainer — "what would flip this hire?"
    # =================================================================
    #
    # When the sophisticated-brain verdict is DEFER or DECLINE, the CHRO's
    # immediate follow-up is "what WOULD have to change for this to be
    # hirable?" This method answers that question deterministically by
    # enumerating the minimal single-lever changes that move the
    # (fit_tier, risk_level) cell into a positive-verdict cell.
    #
    # Two kinds of flip levers are evaluated:
    #   1. Candidate-side — a named skill/cert gap is closed (re-uses
    #      `coaching_paths` already computed by analyze_skill_gap).
    #   2. Team-side — a retention intervention drops attrition risk
    #      enough to cross a band threshold (re-uses simulate_intervention
    #      counterfactually on each lever).
    #
    # This is an explainer, not a recommendation to execute. It gives the
    # hiring committee a concrete boundary — "this req is not DEFER
    # forever; close X or run Y and it flips." Complements the multi-
    # brain consensus story: consensus tells you "is this cell stable?";
    # counterfactual tells you "what would change the cell?"

    def _counterfactual_flip_analysis(self, talent: Dict, workforce: Dict,
                                         decision: Dict, department: str
                                         ) -> Optional[Dict[str, Any]]:
        """Compute flip paths out of a DEFER/DECLINE verdict. Returns None
        for positive-verdict decisions (zero work on the happy path)."""
        if decision["verdict"] not in {"DEFER", "DECLINE"}:
            return None

        current_tier = decision["fit_tier"]
        current_risk = decision["risk_level"]

        def _prob_to_risk(p: float) -> str:
            # Matches ForecastingEngine._prob_to_risk (bilstm_model.py).
            if p >= 0.70:
                return "CRITICAL"
            if p >= 0.50:
                return "HIGH"
            if p >= 0.30:
                return "MEDIUM"
            return "LOW"

        def _cell_is_positive(ft: str, rl: str) -> bool:
            cell = _HIRE_DECISION_MATRIX.get((ft, rl))
            return bool(
                cell and cell["verdict"] not in {"DEFER", "DECLINE",
                                                   "INCONCLUSIVE"}
            )

        paths: List[Dict[str, Any]] = []

        # ---- Candidate-side flips — re-use coaching_paths ----
        coaching_paths = talent.get("coaching_paths", []) or []
        for cp in coaching_paths:
            projected_tier = cp.get("projected_tier")
            if not projected_tier or projected_tier == current_tier:
                continue
            projected_cell = _HIRE_DECISION_MATRIX.get(
                (projected_tier, current_risk)
            )
            if not projected_cell:
                continue
            if not _cell_is_positive(projected_tier, current_risk):
                continue
            gap_type = cp.get("gap_type", "?")
            criticality = cp.get("criticality", "STANDARD")
            paths.append({
                "approach": "candidate_side",
                "lever": cp.get("if_candidate_closes_gap"),
                "lever_type": gap_type,
                "criticality": criticality,
                "action": cp.get("action_to_close", ""),
                "fit_delta_points": cp.get("delta"),
                "eta_days": self._eta_days_from_action(
                    cp.get("action_to_close", "")
                ),
                "projected_fit_tier": projected_tier,
                "projected_risk_level": current_risk,
                "projected_verdict": projected_cell["verdict"],
                "projected_rule": projected_cell["rule"],
                "feasibility": (
                    "low" if gap_type == "degree" else
                    "high" if criticality == "CRITICAL" else
                    "medium"
                ),
            })

        # ---- Team-side flips — counterfactual forward pass per intervention ----
        from tools.forecast_tools import (
            simulate_intervention, _interventions,
        )
        for iv_name, cfg in _interventions().items():
            # Skip levers that don't move attrition probability (e.g.,
            # knowledge_capture affects cost of departure, not rate).
            if cfg.get("feature_affected") is None:
                continue
            # Canonical magnitudes: comp levers at +0.10 (one market step);
            # satisfaction/engagement/flex at +0.20 (a noticeable but
            # realistic program delta). Simulator is linear in magnitude.
            magnitude = (0.10 if iv_name in ("comp_adjustment",
                                              "retention_bonus") else 0.20)
            result = self._call(
                f"simulate_intervention[{iv_name}]", simulate_intervention,
                department=department, intervention=iv_name,
                magnitude=magnitude,
            )
            if "error" in result:
                continue
            cf_prob = result.get("counterfactual_prob")
            baseline_prob = result.get("baseline_prob")
            if cf_prob is None or baseline_prob is None:
                continue
            new_risk = _prob_to_risk(float(cf_prob))
            # The lever must cross a band threshold AND land us in a
            # positive cell — otherwise it's not a flip.
            if new_risk == current_risk:
                continue
            projected_cell = _HIRE_DECISION_MATRIX.get(
                (current_tier, new_risk)
            )
            if not projected_cell:
                continue
            if not _cell_is_positive(current_tier, new_risk):
                continue
            causal = cfg.get("causal_status", "MIXED")
            paths.append({
                "approach": "team_side",
                "lever": iv_name,
                "lever_label": iv_name.replace("_", " ").title(),
                "causal_status": causal,
                "magnitude": magnitude,
                "baseline_prob": round(float(baseline_prob), 3),
                "counterfactual_prob": round(float(cf_prob), 3),
                "delta_prob_pp": round(
                    (float(baseline_prob) - float(cf_prob)) * 100, 1
                ),
                "est_program_cost_usd": result.get("estimated_program_cost_usd"),
                "lead_time_days": result.get("lead_time_days"),
                "projected_fit_tier": current_tier,
                "projected_risk_level": new_risk,
                "projected_verdict": projected_cell["verdict"],
                "projected_rule": projected_cell["rule"],
                "feasibility": (
                    "low" if causal == "CORRELATIONAL" else
                    "high" if causal == "CAUSAL" else
                    "medium"
                ),
            })

        # Rank by feasibility (causal+fast first), then by size of move.
        feasibility_rank = {"high": 0, "medium": 1, "low": 2}

        def _sort_key(p: Dict[str, Any]) -> Tuple[int, float]:
            feas = feasibility_rank.get(p.get("feasibility", "medium"), 3)
            if p["approach"] == "team_side":
                magnitude = -(p.get("delta_prob_pp") or 0.0)
            else:
                magnitude = -(p.get("fit_delta_points") or 0.0)
            return (feas, magnitude)

        paths.sort(key=_sort_key)

        return {
            "trigger_verdict": decision["verdict"],
            "trigger_rule": decision["rule"],
            "current_cell": {
                "fit_tier": current_tier,
                "risk_level": current_risk,
            },
            "flip_feasible": bool(paths),
            "n_candidate_paths": sum(
                1 for p in paths if p["approach"] == "candidate_side"
            ),
            "n_team_paths": sum(
                1 for p in paths if p["approach"] == "team_side"
            ),
            "paths": paths[:5],
        }

    def _eta_days_from_action(self, action: str) -> Optional[int]:
        """Rough ETA in days parsed from a coaching action string.

        Patterns recognized: "3-6mo", "12mo", "5-day", "1-2 days",
        "2-8wk", "4yr". The day-pattern accepts both dash-joined
        ("5-day") and space-joined ("5 days") forms. Returns None when
        no unit is parseable — callers render an empty ETA rather than
        a misleading default.
        """
        import re
        if not action:
            return None
        s = action.lower()
        m = re.search(r"(\d+)\s*(?:-\s*(\d+))?\s*mo\b", s)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2) or lo)
            return round((lo + hi) / 2 * 30)
        m = re.search(r"(\d+)\s*(?:-\s*(\d+))?\s*wk\b", s)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2) or lo)
            return round((lo + hi) / 2 * 7)
        # Accept both "1-2 days" (space) and "5-day" (dash) forms.
        m = re.search(r"(\d+)\s*(?:-\s*(\d+))?[\s-]*days?\b", s)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2) or lo)
            return round((lo + hi) / 2)
        m = re.search(r"(\d+)\s*yrs?\b", s)
        if m:
            return int(m.group(1)) * 365
        return None

    def _blend_confidence(self, talent: Dict, workforce: Dict,
                           uncertainty: Dict, conformal: Dict,
                           trajectory: Dict
                           ) -> Dict[str, Any]:
        """Combine the five confidence sources into a single blended
        score. Weakest-link logic with weighted-mean fallback.

        Conformal width is the new calibration component: a narrow
        guaranteed-coverage interval earns "high"; a near-[0, 1]
        interval earns "low" — this catches the MC-Dropout-tight-but-
        held-out-weak pathology that pure Bayesian spread misses.
        """
        level_to_num = {"high": 1.0, "medium": 0.6, "low": 0.3}
        scores = []
        components = {}

        t_conf = (talent.get("confidence") or {}).get("level")
        if t_conf:
            components["talent"] = t_conf
            scores.append(level_to_num.get(t_conf, 0.5) * _CONFIDENCE_WEIGHTS["talent"])

        w_conf = (workforce.get("confidence") or {}).get("level")
        if w_conf:
            components["workforce"] = w_conf
            scores.append(level_to_num.get(w_conf, 0.5) * _CONFIDENCE_WEIGHTS["workforce"])

        u_level = uncertainty.get("uncertainty_level")
        if u_level:
            components["uncertainty"] = u_level
            scores.append(level_to_num.get(u_level, 0.5) * _CONFIDENCE_WEIGHTS["uncertainty"])

        # Split Conformal → calibration-confidence signal.
        #
        # Threshold tuning: width 0.15 = a 15pp-wide 90% CI, which is the
        # practical ceiling for a classifier to produce actionable
        # hire/retention decisions. Anything wider degrades to "medium";
        # anything above 0.40 (40pp CI) is explicitly "low" confidence
        # in the model's calibration.
        conf_width = None
        if isinstance(conformal, dict) and "error" not in conformal:
            conf_width = conformal.get("interval_width")
        if conf_width is not None:
            if conf_width < 0.15:
                c_level = "high"
            elif conf_width < 0.40:
                c_level = "medium"
            else:
                c_level = "low"
            components["conformal"] = c_level
            scores.append(level_to_num.get(c_level, 0.5) * _CONFIDENCE_WEIGHTS["conformal"])

        tdir = trajectory.get("trajectory_direction", "stable")
        tscore = {"stable": 1.0, "improving": 1.0, "worsening": 0.6}.get(tdir, 0.5)
        components["trajectory"] = tdir
        scores.append(tscore * _CONFIDENCE_WEIGHTS["trajectory"])

        total = sum(scores) / sum(_CONFIDENCE_WEIGHTS.values())
        if total >= 0.80:
            level = "high"
        elif total >= 0.55:
            level = "medium"
        else:
            level = "low"

        # Pathological-width cap — weakest-link principle on calibration.
        # When the Split Conformal guaranteed-coverage CI spans essentially
        # all of [0, 1] (width > 0.80), the model has no actionable
        # calibration on held-out data. No amount of internal consistency
        # from the other components can honestly upgrade this to "high".
        # Floor the blended level at "medium". Documented in
        # WRITEUP_LEARNINGS §16.2 as the core MC-vs-Conformal diagnostic.
        capped = False
        if (conf_width is not None
                and conf_width > 0.80
                and level == "high"):
            level = "medium"
            capped = True

        return {"level": level, "score": round(total, 2),
                "components": components,
                "conformal_width_capped": capped}

    # =================================================================
    # INTERNAL — scan-mode action derivation
    # =================================================================

    def _derive_scan_actions(self, details: List[Dict],
                              top_n: int = 3) -> List[Dict]:
        """From a list of per-department deep-dives, extract the top-N
        most urgent action items."""
        actions = []
        for d in details:
            dept = d["dept"]
            analysis = d["analysis"] or {}
            plan = analysis.get("retention_plan", []) or []
            cost = analysis.get("replacement_cost", {}) or {}
            repl = cost.get("estimated_replacement_cost_usd", 0)
            for p in plan:
                actions.append({
                    "department": dept,
                    "action": p.get("intervention", "").replace("_", " ").title(),
                    "driver": p.get("target_driver_label"),
                    "cohort_size": p.get("cohort_size"),
                    "program_cost_usd": p.get("est_program_cost_usd", 0),
                    "replacement_exposure_usd": repl,
                    "lead_time_days": p.get("lead_time_days"),
                    "risk_level": analysis.get("risk_level"),
                    "roi_multiplier": (
                        round(repl / p["est_program_cost_usd"], 1)
                        if p.get("est_program_cost_usd", 0) > 0 and repl > 0
                        else None
                    ),
                })
        actions.sort(
            key=lambda a: -(a.get("roi_multiplier") or 0),
        )
        return actions[:top_n]

    # =================================================================
    # INTERNAL — narrative generators (template-based, deterministic)
    # =================================================================

    def _render_hire_memo(self, talent: Dict, workforce: Dict,
                           decision: Dict, retention: Dict,
                           backfill: Dict, confidence: Dict,
                           uncertainty: Dict, conformal: Dict,
                           trajectory: Dict,
                           department: str,
                           counterfactual: Optional[Dict[str, Any]] = None
                           ) -> AgentMemo:
        """Generate the 3-section hire memo — deterministic prose."""
        fit = talent.get("fit_score", {}) or {}
        coverage = talent.get("coverage", {}) or {}
        top_gaps = talent.get("top_gaps", []) or []
        years = talent.get("years_experience_gap", {}) or {}
        drivers = (workforce.get("drivers") or {}).get("top_drivers", [])
        cohorts = workforce.get("cohorts", {}) or {}

        # ---- Section A: HIRE DECISION ----
        a_parts = [decision["rationale"]]
        a_parts.append(
            f"Candidate fit: **{fit.get('composite_fit_score', 0)}/100** "
            f"({fit.get('recommendation_tier', '?')}). "
            f"Critical-weighted skills coverage: "
            f"{coverage.get('critical_weighted_coverage', 0)}%."
        )
        # Named critical gaps
        crit_gaps = [g["requirement"] for g in top_gaps
                     if g.get("criticality") == "CRITICAL"]
        if crit_gaps:
            a_parts.append(
                f"Top critical gaps to verify: "
                + ", ".join(f"**{g}**" for g in crit_gaps[:3]) + "."
            )
        if years.get("meets_requirement") is False:
            a_parts.append(
                f"Experience shortfall: {years.get('candidate_has')} vs "
                f"{years.get('required')} required."
            )
        # Workforce side — prefer the MC-Dropout Bayesian mean when available
        # so the probability and its CI are internally consistent. Falls
        # back to the deterministic point estimate when uncertainty is
        # missing.
        point_prob = workforce.get("attrition_probability", 0)
        mc_mean = uncertainty.get("mean_probability")
        u_p5 = uncertainty.get("percentile_5")
        u_p95 = uncertainty.get("percentile_95")
        prob = mc_mean if mc_mean is not None else point_prob
        a_parts.append(
            f"Receiving team ({department}) attrition probability: "
            f"**{prob:.0%}**. Risk level: **{decision['risk_level']}**."
        )
        # Dual uncertainty block: MC-Dropout for model-internal consistency,
        # Split Conformal for distribution-free guaranteed coverage. Citing
        # both is the frontier rigour documented in WRITEUP §16.2.
        if u_p5 is not None and u_p95 is not None:
            a_parts.append(
                f"• **MC-Dropout** (Bayesian, "
                f"{uncertainty.get('n_samples', 30)} samples): "
                f"90% CI [{u_p5:.0%}, {u_p95:.0%}] — model-internal consistency."
            )
        conf_ok = (isinstance(conformal, dict)
                   and "error" not in conformal
                   and conformal.get("p_low") is not None)
        if conf_ok:
            c_p5 = conformal["p_low"]
            c_p95 = conformal["p_high"]
            c_n = conformal.get("n_calibration", 0)
            c_width = conformal.get("interval_width", 0)
            a_parts.append(
                f"• **Split Conformal** (Angelopoulos 2021, "
                f"{c_n}-sample calibration): 90% CI "
                f"[{c_p5:.0%}, {c_p95:.0%}] — distribution-free "
                f"coverage guarantee."
            )
            # Surface the MC vs Conformal gap when it is diagnostic.
            mc_std = uncertainty.get("std", 0)
            if c_width > 0.7 and mc_std is not None and mc_std < 0.05:
                a_parts.append(
                    "*MC/Conformal gap: model is internally consistent "
                    "but the Bi-LSTM's held-out accuracy cannot support "
                    "a narrow guaranteed interval. For audit / regulatory "
                    "reporting, cite the Conformal CI.*"
                )
        if drivers:
            d0 = drivers[0]
            a_parts.append(
                f"Primary driver of that risk: **{d0['label']}** "
                f"({d0['pct_of_risk']:.0f}% of the model's attrition score)."
            )
        if decision.get("modifiers"):
            a_parts.append(
                "Additional considerations: " + "; ".join(decision["modifiers"]) + "."
            )
        # Surface feedback-driven adjustments prominently. This is the
        # user-facing evidence of continual learning: when a prior human
        # correction has reshaped the policy, callers should see that
        # the memo they are about to act on was adjusted away from the
        # base-policy verdict.
        for adj in decision.get("adjustments_applied", []) or []:
            if adj.get("type") == "rule_override":
                a_parts.append(
                    f"**Human-in-the-Lead adjustment:** verdict was "
                    f"`{adj['original_verdict']}` under the base policy; "
                    f"adjusted to `{adj['new_verdict']}` based on "
                    f"{adj.get('evidence_votes', 1)} prior human "
                    f"correction(s) via "
                    f"`{adj.get('source', 'feedback')}`. The decision "
                    f"trace cites the supporting event IDs."
                )

        section_a = MemoSection(
            title=f"A. Hire Decision — {decision['verdict']}",
            narrative=" ".join(a_parts),
            metrics={
                "fit_score": fit.get("composite_fit_score"),
                "fit_tier": fit.get("recommendation_tier"),
                "attrition_prob": round(prob, 3),
                "risk_level": decision["risk_level"],
                "rule": decision["rule"],
                "mc_dropout_ci": (
                    [round(u_p5, 3), round(u_p95, 3)]
                    if u_p5 is not None and u_p95 is not None else None
                ),
                "conformal_ci": (
                    [round(conformal["p_low"], 3),
                     round(conformal["p_high"], 3)]
                    if conf_ok else None
                ),
                "conformal_width": (round(conformal["interval_width"], 3)
                                     if conf_ok else None),
            },
        )

        # ---- Section B: RETENTION PLAN ----
        b_parts = []
        urgency = retention["urgency"]
        urgency_line = {
            "standard": "Retention posture is business-as-usual.",
            "elevated": "Retention focus is ELEVATED — assign a deliberate buddy and schedule 30/60/90-day check-ins for the new hire.",
            "high":     "Retention is a HIGH priority — the retention program below must launch alongside the offer, not after it.",
            "critical": "Retention is CRITICAL — the retention program is a PRECONDITION to extending the offer.",
        }.get(urgency, "")
        b_parts.append(urgency_line)
        if retention["actions"]:
            # Icons communicate the causal defensibility of each lever at a
            # glance — executives can't read each rationale but can scan the
            # status column. Full legend appears as a footer.
            _CAUSAL_ICONS = {
                "CAUSAL": "✓ Causal",
                "MIXED": "~ Mixed",
                "CORRELATIONAL": "⚠ Correlational",
                "NONE": "— Cost-only",
            }
            seen_statuses = set()
            for p in retention["actions"][:3]:
                status = p.get("causal_status", "MIXED")
                seen_statuses.add(status)
                icon = _CAUSAL_ICONS.get(status, status)
                b_parts.append(
                    f"- **{p['intervention'].replace('_', ' ').title()}** targeting "
                    f"**{p['target_driver_label']}** — "
                    f"{p['cohort_size']} people × "
                    f"${p['est_program_cost_usd']:,} "
                    f"({p['lead_time_days']}d lead time). "
                    f"*[{icon}]*"
                )
            b_parts.append(
                f"Total program cost: **${retention['total_program_cost_usd']:,}**. "
                f"Modelled replacement exposure if we do nothing: "
                f"**${retention['replacement_exposure_usd']:,}**."
                + (f" ROI: **{retention['roi_multiplier']}× "
                   "(replacement avoided / program cost)**."
                   if retention.get("roi_multiplier") else "")
            )
            # Causal legend — only shown when something non-causal is in the
            # plan. Surfaces the §16.2 WRITEUP diagnostic where it matters.
            if seen_statuses - {"CAUSAL"}:
                legend_parts = [
                    "**Causal-status legend (per intervention above):**"
                ]
                if "CORRELATIONAL" in seen_statuses:
                    legend_parts.append(
                        "`⚠ Correlational` — effect estimate relies on a "
                        "learned correlation confounded by selection "
                        "effects (headhunting of high-performers). "
                        "Treat simulator magnitude as an upper bound; "
                        "rely on published HR-program effect sizes for "
                        "the lower bound."
                    )
                if "MIXED" in seen_statuses:
                    legend_parts.append(
                        "`~ Mixed` — direct causal pathway exists but "
                        "the Bi-LSTM's learned association is partly "
                        "confounded. Magnitude approximate; direction "
                        "reliable."
                    )
                if "NONE" in seen_statuses:
                    legend_parts.append(
                        "`— Cost-only` — intervention reduces cost of "
                        "departure, not probability of one (e.g., "
                        "knowledge capture). Not a rate-lowering lever."
                    )
                b_parts.append(" ".join(legend_parts))
        else:
            b_parts.append(
                "No targeted interventions recommended — monitor lead "
                "indicators and revisit at next quarterly review."
            )
        cohort_highlights = []
        for cohort_key, label in [
            ("retirement_cliff_critical", "retirement-cliff (critical)"),
            ("new_hire_churn_window", "new-hire churn"),
            ("comp_gap", "comp-gap"),
        ]:
            c = cohorts.get(cohort_key, {}) or {}
            if c.get("count", 0) >= 3:
                cohort_highlights.append(
                    f"{c['count']} in {label} ({c['pct_of_department']}%)"
                )
        if cohort_highlights:
            b_parts.append(
                "Cohort focus: " + ", ".join(cohort_highlights) + "."
            )
        section_b = MemoSection(
            title="B. Retention Plan",
            narrative=" ".join(b_parts),
            metrics={
                "urgency": urgency,
                "total_program_cost_usd": retention["total_program_cost_usd"],
                "replacement_exposure_usd": retention["replacement_exposure_usd"],
                "roi_multiplier": retention.get("roi_multiplier"),
                "n_actions": len(retention["actions"]),
                "causal_mix": {
                    s: sum(
                        1 for p in retention["actions"]
                        if p.get("causal_status") == s
                    )
                    for s in ("CAUSAL", "MIXED", "CORRELATIONAL", "NONE")
                    if any(
                        p.get("causal_status") == s
                        for p in retention["actions"]
                    )
                },
            },
        )

        # ---- Section C: BACKFILL PIPELINE ----
        c_parts = []
        if backfill["internal_candidates"]:
            ids = [f"#{c['employee_id']}" for c in backfill["internal_candidates"][:5]]
            srcs = backfill["internal_sources"]
            c_parts.append(
                f"**Internal mobility bench:** "
                f"{len(backfill['internal_candidates'])} portable "
                f"candidate(s) from {', '.join(srcs)} — "
                f"employee IDs {', '.join(ids)}."
            )
            best = backfill["internal_candidates"][0]
            c_parts.append(
                f"Top internal candidate: employee #{best['employee_id']} "
                f"from {best['from_department']} "
                f"(mobility score {best['mobility_score']:.2f} — "
                f"tenure {best['tenure_years']:.0f}y, "
                f"perf {best['performance']:.1f}, "
                f"sat {best['satisfaction']:.1f})."
            )
        else:
            c_parts.append(
                "**Internal mobility bench:** no adjacent-department "
                "candidates meet the portability bar — backfill must be "
                "external."
            )
        if backfill["external_channels"]:
            c_parts.append("**External sourcing channels mapped to named gaps:**")
            for ch in backfill["external_channels"]:
                c_parts.append(f"- {ch['for_gap']} → {ch['channel']}")
        section_c = MemoSection(
            title="C. Backfill Pipeline",
            narrative="\n\n".join(c_parts),
            metrics={
                "internal_bench_size": backfill["total_internal_bench"],
                "n_external_channels": len(backfill["external_channels"]),
            },
        )

        headline = (
            f"{decision['verdict']}: candidate for {department} "
            f"(fit {fit.get('composite_fit_score', 0)}/100, "
            f"team risk {decision['risk_level']})"
        )

        # Section D is only appended when the verdict is DEFER/DECLINE
        # and counterfactual analysis was produced. Positive-verdict
        # memos remain 3-section (A/B/C) — preserving byte-identity on
        # the happy path.
        sections = [section_a, section_b, section_c]
        if counterfactual:
            sections.append(self._render_counterfactual_section(counterfactual))

        return AgentMemo(
            memo_type="hire_decision",
            headline=headline,
            verdict=decision["verdict"],
            rule_applied=decision["rule"],
            sections=sections,
            confidence=confidence,
            metrics={
                "department": department,
                "fit_score": fit.get("composite_fit_score"),
                "risk_level": decision["risk_level"],
                "attrition_probability": round(prob, 3),
                "trajectory_direction": trajectory.get("trajectory_direction"),
                "counterfactual_flip_feasible": (
                    counterfactual.get("flip_feasible")
                    if counterfactual else None
                ),
                "counterfactual_n_paths": (
                    len(counterfactual.get("paths", []))
                    if counterfactual else None
                ),
            },
        )

    def _render_counterfactual_section(self, cf: Dict[str, Any]
                                         ) -> MemoSection:
        """Render Section D of a DEFER/DECLINE hire memo — flip explainer."""
        current_tier = cf["current_cell"]["fit_tier"]
        current_risk = cf["current_cell"]["risk_level"]
        trigger = cf["trigger_verdict"]

        d_parts = [
            f"Under the current policy this verdict is `{trigger}` for a "
            f"candidate at (fit_tier=**{current_tier}**, "
            f"risk=**{current_risk}**). The following counterfactual "
            f"levers would move the (fit × risk) cell into a hire-family "
            f"or advance-family verdict. Feasibility grades are: **high** "
            f"= causal and fast, **medium** = mixed-causal or ~3-6 month "
            f"horizon, **low** = correlational-only or degree-level gap. "
            f"This is an explainer, not a recommendation — any lever "
            f"chosen still requires hiring-manager + CHRO sign-off."
        ]

        cand = [p for p in cf["paths"] if p["approach"] == "candidate_side"][:2]
        team = [p for p in cf["paths"] if p["approach"] == "team_side"][:2]

        _CAUSAL_ICONS = {
            "CAUSAL": "✓",
            "MIXED": "~",
            "CORRELATIONAL": "⚠",
            "NONE": "—",
        }

        if cand:
            d_parts.append("**Candidate-side flips** (close a named skill gap):")
            for p in cand:
                eta = (f" (~{p['eta_days']}d)"
                       if p.get("eta_days") else "")
                delta_pts = p.get("fit_delta_points")
                delta_str = (f"+{delta_pts:.1f}pts"
                             if delta_pts is not None else "+? pts")
                d_parts.append(
                    f"- Close **{p['lever']}** ({p['criticality']} "
                    f"{p['lever_type']}): fit lifts {delta_str} "
                    f"to **{p['projected_fit_tier']}**; verdict becomes "
                    f"`{p['projected_verdict']}` "
                    f"({p['projected_rule']}){eta}. "
                    f"*Feasibility: {p['feasibility']}.* "
                    f"Action: {p['action']}"
                )

        if team:
            d_parts.append("**Team-side flips** (retention intervention):")
            for p in team:
                icon = _CAUSAL_ICONS.get(
                    p.get("causal_status", "MIXED"), "?"
                )
                cost = (f" (~${p['est_program_cost_usd']:,})"
                        if p.get("est_program_cost_usd") else "")
                d_parts.append(
                    f"- **{p['lever_label']}** [{icon} "
                    f"{p['causal_status']}] at magnitude "
                    f"{p['magnitude']:.2f}: risk drops "
                    f"{p['baseline_prob']:.0%}→"
                    f"{p['counterfactual_prob']:.0%} "
                    f"(−{p['delta_prob_pp']:.1f}pp) to "
                    f"**{p['projected_risk_level']}**; verdict becomes "
                    f"`{p['projected_verdict']}` "
                    f"({p['projected_rule']}){cost}, "
                    f"lead-time {p['lead_time_days']}d. "
                    f"*Feasibility: {p['feasibility']}.*"
                )

        if not cand and not team:
            d_parts.append(
                "No single-lever change (candidate skill closure OR team "
                "intervention at canonical magnitude) moves this cell "
                "out of DEFER/DECLINE territory. The gap is beyond the "
                "in-scope levers — recommend re-scoping the role, "
                "expanding sourcing, or deferring the req until the "
                "receiving team stabilizes independently."
            )

        return MemoSection(
            title="D. Counterfactual — What Would Flip This Verdict?",
            narrative="\n\n".join(d_parts),
            metrics={
                "trigger_verdict": trigger,
                "trigger_rule": cf.get("trigger_rule"),
                "current_fit_tier": current_tier,
                "current_risk_level": current_risk,
                "flip_feasible": cf["flip_feasible"],
                "n_paths": len(cf["paths"]),
                "n_candidate_paths": cf["n_candidate_paths"],
                "n_team_paths": cf["n_team_paths"],
                # Structured paths — consumed by the dashboard's chart+cards
                # renderer. Keep the same lists the markdown narrative uses
                # so the two views stay synchronized.
                "candidate_paths": list(cand),
                "team_paths": list(team),
            },
        )

    def _render_scan_memo(self, ranked: List[Dict], details: List[Dict],
                           actions: List[Dict]) -> AgentMemo:
        """Company-wide risk scan memo."""
        # Section 1: Risk ranking
        lines = ["| Rank | Department | Risk | Prob | 6-mo HC |", "|---|---|---|---|---|"]
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"| {i} | {r.get('department', '?')} | "
                f"**{r.get('risk_level', '?')}** | "
                f"{r.get('attrition_probability', 0):.0%} | "
                f"{r.get('current_headcount', 0)} → {r.get('projected_headcount_6m', 0)} |"
            )
        # Per-department classifications under all three risk policies.
        # Used by the dashboard's Threshold Sensitivity panel and surfaced
        # in section_1.metrics so it stays alongside the ranking it
        # describes.
        policy_classifications = _multi_policy_risk_classifications(ranked)
        n_disagree = sum(1 for d in policy_classifications if not d["agreement"])
        n_jump = sum(1 for d in policy_classifications if d["severity_spread"] >= 2)
        section_1 = MemoSection(
            title="Department Risk Ranking",
            narrative="\n".join(lines),
            metrics={"n_departments": len(ranked),
                     "n_critical": sum(1 for r in ranked
                                        if r.get("risk_level") == "CRITICAL"),
                     "n_high": sum(1 for r in ranked
                                     if r.get("risk_level") == "HIGH"),
                     "ranked": list(ranked),
                     "policy_classifications": policy_classifications,
                     "n_policy_disagreements": n_disagree,
                     "n_policy_jumps": n_jump,
                     "policy_thresholds": dict(_RISK_POLICY_THRESHOLDS),
                     "policy_blurbs": dict(_RISK_POLICY_BLURBS)},
        )

        # Section 2: Deep-dive on top 3
        parts = []
        for d in details:
            a = d["analysis"] or {}
            u = d["uncertainty"] or {}
            c = d.get("conformal") or {}
            drivers = (a.get("drivers") or {}).get("top_drivers", [])
            cohorts = a.get("cohorts", {}) or {}
            retire_crit = (cohorts.get("retirement_cliff_critical") or {}).get("count", 0)
            comp_gap = (cohorts.get("comp_gap") or {}).get("count", 0)
            cost = (a.get("replacement_cost") or {}).get("estimated_replacement_cost_usd", 0)
            top_drv = drivers[0] if drivers else {}
            # Dual-interval uncertainty line — MC-Dropout + Split Conformal.
            mc_line = (
                f"MC-Dropout 90% CI "
                f"[{u.get('percentile_5', 0):.0%}, "
                f"{u.get('percentile_95', 0):.0%}]"
            )
            conf_line = ""
            if isinstance(c, dict) and "error" not in c and c.get("p_low") is not None:
                conf_line = (
                    f" · Conformal 90% CI "
                    f"[{c['p_low']:.0%}, {c['p_high']:.0%}] "
                    f"(width {c.get('interval_width', 0):.2f})"
                )
            parts.append(
                f"### {d['dept']} — {a.get('risk_level', '?')} "
                f"({a.get('attrition_probability', 0):.0%} "
                f"± {u.get('std', 0)*100:.1f}pp)\n"
                f"Uncertainty: {mc_line}{conf_line}.\n"
                f"Primary driver: **{top_drv.get('label', '?')}** "
                f"({top_drv.get('pct_of_risk', 0):.0f}%). "
                f"Cohorts at risk: {retire_crit} retirement-cliff, "
                f"{comp_gap} comp-gap. "
                f"Modelled replacement exposure: **${cost:,}**."
            )
        # Compact structured deep-dive for the dashboard renderer (parallel
        # to the markdown narrative above; same facts, JSON-friendly shape).
        deep_dive_struct = []
        for d in details:
            a = d["analysis"] or {}
            u = d["uncertainty"] or {}
            c = d.get("conformal") or {}
            drivers = (a.get("drivers") or {}).get("top_drivers", []) or []
            cohorts = a.get("cohorts", {}) or {}
            top_drv = drivers[0] if drivers else {}
            deep_dive_struct.append({
                "department": d["dept"],
                "risk_level": a.get("risk_level"),
                "attrition_probability": a.get("attrition_probability"),
                "current_headcount": a.get("current_headcount"),
                "predicted_departures_12mo": a.get("predicted_departures_12mo"),
                "projected_headcount_6m": a.get("projected_headcount_6m"),
                "mc_p5": u.get("percentile_5"),
                "mc_p95": u.get("percentile_95"),
                "mc_std": u.get("std"),
                "conformal_p_low": (c.get("p_low") if isinstance(c, dict)
                                     and "error" not in c else None),
                "conformal_p_high": (c.get("p_high") if isinstance(c, dict)
                                      and "error" not in c else None),
                "conformal_width": (c.get("interval_width") if isinstance(c, dict)
                                     and "error" not in c else None),
                "primary_driver_label": top_drv.get("label"),
                "primary_driver_pct": top_drv.get("pct_of_risk"),
                "retirement_cliff_critical": (cohorts
                    .get("retirement_cliff_critical") or {}).get("count", 0),
                "comp_gap_count": (cohorts.get("comp_gap") or {}).get("count", 0),
                "knowledge_loss_severity": (a.get("knowledge_loss")
                                             or {}).get("severity"),
                "replacement_cost_usd": (a.get("replacement_cost") or {})
                    .get("estimated_replacement_cost_usd", 0),
            })
        section_2 = MemoSection(
            title="Top-3 Deep-Dive",
            narrative="\n\n".join(parts),
            metrics={"deep_dive": deep_dive_struct},
        )

        # Section 3: Action items
        if actions:
            action_lines = []
            for a in actions:
                roi_part = (f" (ROI {a['roi_multiplier']}×)"
                            if a.get("roi_multiplier") else "")
                action_lines.append(
                    f"- **{a['department']}**: {a['action']} for "
                    f"{a['cohort_size']} people targeting "
                    f"*{a['driver']}* — "
                    f"${a['program_cost_usd']:,}{roi_part}"
                )
            section_3 = MemoSection(
                title="Top Action Items (ranked by ROI)",
                narrative="\n".join(action_lines),
                metrics={"n_actions": len(actions),
                         "total_cost_usd": sum(a["program_cost_usd"]
                                                for a in actions),
                         "actions": list(actions)},
            )
        else:
            section_3 = MemoSection(
                title="Top Action Items",
                narrative="No urgent action items identified.",
            )

        headline = (
            f"Workforce Risk Scan — {section_1.metrics['n_critical']} "
            f"critical, {section_1.metrics['n_high']} high-risk departments"
        )
        return AgentMemo(
            memo_type="risk_scan",
            headline=headline,
            sections=[section_1, section_2, section_3],
            metrics={**section_1.metrics,
                     "n_actions_recommended": len(actions)},
        )

    def _render_plan_memo(self, opt: Dict, budget: int) -> AgentMemo:
        """Quarterly retention plan memo — from the portfolio optimizer."""
        section_1 = MemoSection(
            title="Plan Summary",
            narrative=opt.get("interpretation", ""),
            metrics={
                "budget_usd": opt["budget_usd"],
                "spent_usd": opt["spent_usd"],
                "unused_budget_usd": opt["unused_budget_usd"],
                "total_reduction_pp": opt["total_attrition_reduction_pp"],
                "avg_roi_pp_per_million": opt["avg_roi_pp_per_million"],
                "n_interventions_selected": opt["n_interventions_selected"],
            },
        )

        # Per-dept rollup
        per_dept = opt.get("per_department_rollup", []) or []
        lines = ["| Department | Actions | Cost | Reduction |",
                 "|---|---|---|---|"]
        for rd in per_dept:
            lines.append(
                f"| {rd['department']} | {rd['n_actions']} | "
                f"${rd['cost_usd']:,} | "
                f"**-{rd['reduction_pp']:.1f}pp** |"
            )
        section_2 = MemoSection(
            title="Per-Department Rollup",
            narrative="\n".join(lines),
            metrics={"per_department_rollup": list(per_dept)},
        )

        # Top 10 actions
        portfolio_top10 = list(opt.get("portfolio", []) or [])[:10]
        lines = ["| # | Dept | Action | Mag | Cost | Reduction | ROI (pp/$M) |",
                 "|---|---|---|---|---|---|---|"]
        for i, a in enumerate(portfolio_top10, 1):
            lines.append(
                f"| {i} | {a['department']} | {a['intervention_label']} | "
                f"{a['magnitude']:.2f} | ${a['cost_usd']:,} | "
                f"-{a['attrition_reduction_pp']:.1f}pp | "
                f"{a['roi_pp_per_million']:.0f} |"
            )
        section_3 = MemoSection(
            title="Top 10 Actions (by ROI)",
            narrative="\n".join(lines),
            metrics={"portfolio_top10": portfolio_top10},
        )

        headline = (
            f"Quarterly Retention Plan — ${opt['spent_usd']:,} allocated, "
            f"-{opt['total_attrition_reduction_pp']:.1f}pp total reduction "
            f"(avg ROI {opt['avg_roi_pp_per_million']:.0f} pp/$M)"
        )
        return AgentMemo(
            memo_type="retention_plan",
            headline=headline,
            sections=[section_1, section_2, section_3],
            metrics={**section_1.metrics,
                     "n_departments_served": len(opt.get("per_department_rollup", []))},
        )

    def _render_shortlist_memo(self, result: Dict) -> AgentMemo:
        """Candidate shortlist memo."""
        rankings = result.get("rankings", []) or []
        lines = ["| Rank | Candidate | Fit | Tier | Top strength | Top gap |",
                 "|---|---|---|---|---|---|"]
        for i, r in enumerate(rankings[:10], 1):
            tm = r.get("top_match") or {}
            tg = r.get("top_gap") or {}
            lines.append(
                f"| {i} | {r.get('name', '?')} | "
                f"{r.get('fit_score', 0)} | "
                f"{r.get('recommendation_tier', '?')} | "
                f"{tm.get('requirement', '—')} | "
                f"{tg.get('requirement', '—')} |"
            )
        section_1 = MemoSection(
            title="Candidate Shortlist",
            narrative="\n".join(lines),
            metrics={
                **(result.get("tier_counts", {}) or {}),
                "rankings": rankings[:10],
            },
        )
        section_2 = MemoSection(
            title="Recommendation",
            narrative=result.get("recommendation", ""),
        )
        headline = f"Shortlist — {len(rankings)} candidates evaluated"
        return AgentMemo(
            memo_type="shortlist",
            headline=headline,
            sections=[section_1, section_2],
            metrics={"n_candidates": len(rankings)},
        )

    def _render_triage_memo(self, result: Dict) -> AgentMemo:
        """Candidate-across-jobs triage memo."""
        rankings = result.get("rankings", []) or []
        lines = ["| Rank | Role | Fit | Tier | Top gap |",
                 "|---|---|---|---|---|"]
        for i, r in enumerate(rankings[:10], 1):
            tg = r.get("top_gap") or {}
            lines.append(
                f"| {i} | {r.get('title', '?')} | "
                f"{r.get('fit_score', 0)} | "
                f"{r.get('recommendation_tier', '?')} | "
                f"{tg.get('requirement', '—')} |"
            )
        section_1 = MemoSection(
            title="Role Fit Ranking",
            narrative="\n".join(lines),
            metrics={
                "n_jobs_evaluated": len(rankings),
                "rankings": rankings[:10],
            },
        )
        section_2 = MemoSection(
            title="Recommendation",
            narrative=result.get("recommendation", ""),
        )
        headline = f"Triage — {len(rankings)} roles evaluated"
        return AgentMemo(
            memo_type="triage",
            headline=headline,
            sections=[section_1, section_2],
            metrics={"n_jobs": len(rankings)},
        )

    def _render_consensus_memo(self,
                                 verdicts: List[Dict[str, Any]],
                                 consensus_class: str,
                                 consensus_action: str,
                                 sophisticated_memo: AgentMemo,
                                 department: str,
                                 fit_score: Any,
                                 fit_tier: Any,
                                 risk_level: Any) -> AgentMemo:
        """Render the multi-brain consensus memo."""
        # Section 1: Per-brain verdicts table
        lines = [
            "| # | Brain | Verdict | Rule | Family |",
            "|---|---|---|---|---|",
        ]
        for i, v in enumerate(verdicts, 1):
            lines.append(
                f"| {i} | **{v['brain']}** | `{v['verdict']}` | "
                f"`{v['rule']}` | {v['family']} |"
            )
        # Per-brain rationale block (one-liner each)
        rationale_lines = []
        for v in verdicts:
            rat = v["rationale"]
            # Keep rationale compact for the consensus memo; sophisticated
            # memo carries the full narrative already.
            if v["brain"] == "Sophisticated":
                n_secs = len(sophisticated_memo.sections)
                rat = (
                    f"See reference memo (Sophisticated): full "
                    f"{n_secs}-section analysis with dual-CI + causal "
                    f"annotations"
                    + (" + counterfactual flip explainer"
                       if n_secs >= 4 else "")
                    + " attached."
                )
            rationale_lines.append(
                f"- **{v['brain']}** ({v['rule']}): {rat}"
            )

        section_1 = MemoSection(
            title="Per-Brain Verdicts",
            narrative="\n".join(lines) + "\n\n" + "\n".join(rationale_lines),
            metrics={
                "n_brains": len(verdicts),
                "unique_families": len({v["family"] for v in verdicts}),
                "verdicts": [v["verdict"] for v in verdicts],
                "verdicts_full": list(verdicts),
            },
        )

        # Section 2: Consensus classification + recommended action
        badge_map = {
            "STRONG_CONSENSUS": "✓ All 3 agree",
            "MAJORITY_CONSENSUS": "⚠ 2/3 majority — escalate",
            "NO_CONSENSUS": "⛔ NO CONSENSUS — ESCALATE",
        }
        badge = badge_map.get(consensus_class, consensus_class)
        section_2 = MemoSection(
            title=f"Consensus — {badge}",
            narrative=consensus_action,
            metrics={
                "consensus_class": consensus_class,
                "n_agreeing_families": (
                    3 - (len({v["family"] for v in verdicts}) - 1)
                ),
                "escalation_required": consensus_class != "STRONG_CONSENSUS",
            },
        )

        # Section 3: Reference memo (sophisticated brain's full analysis,
        # flattened in for context)
        ref_lines = [
            "Full analysis from the Sophisticated brain (dual-CI, causal "
            "annotations, retention plan, backfill bench):",
            "",
            sophisticated_memo.to_markdown(),
        ]
        section_3 = MemoSection(
            title="Reference Memo — Sophisticated Brain",
            narrative="\n".join(ref_lines),
            metrics={
                "sophisticated_verdict": sophisticated_memo.verdict,
                "sophisticated_rule": sophisticated_memo.rule_applied,
                "sophisticated_confidence": (
                    sophisticated_memo.confidence.get("level")
                    if sophisticated_memo.confidence else None
                ),
            },
        )

        # Determine headline verdict + rule for the consensus memo
        # itself — use the majority family where possible, else flag
        # for escalation.
        if consensus_class == "STRONG_CONSENSUS":
            headline_verdict = sophisticated_memo.verdict
            headline_rule = f"CONSENSUS-{verdicts[0]['family']}"
            headline = (
                f"CONSENSUS ({verdicts[0]['family']}): "
                f"all 3 brains agree on {department} hire "
                f"(fit {fit_score}, risk {risk_level})"
            )
        elif consensus_class == "MAJORITY_CONSENSUS":
            from collections import Counter
            counts = Counter(v["family"] for v in verdicts)
            majority_family = counts.most_common(1)[0][0]
            headline_verdict = "ESCALATE_TO_SENIOR_REVIEW"
            headline_rule = f"MAJORITY-{majority_family}"
            headline = (
                f"SPLIT 2/1 (majority {majority_family}): "
                f"senior review recommended for {department} hire"
            )
        else:
            headline_verdict = "ESCALATE_TO_HIRING_COMMITTEE"
            headline_rule = "NO-CONSENSUS"
            headline = (
                f"NO CONSENSUS: all 3 brains disagree on {department} "
                f"hire — policy-dependent; escalate"
            )

        return AgentMemo(
            memo_type="consensus_decision",
            headline=headline,
            verdict=headline_verdict,
            rule_applied=headline_rule,
            sections=[section_1, section_2, section_3],
            confidence=sophisticated_memo.confidence,
            metrics={
                "department": department,
                "fit_score": fit_score,
                "fit_tier": fit_tier,
                "risk_level": risk_level,
                "consensus_class": consensus_class,
                "sophisticated_verdict": sophisticated_memo.verdict,
                "conservative_verdict": verdicts[1]["verdict"],
                "heuristic_verdict": verdicts[2]["verdict"],
            },
        )

    # =================================================================
    # INTERNAL — error envelope
    # =================================================================

    def _error_memo(self, memo_type: str, message: str) -> AgentMemo:
        return AgentMemo(
            memo_type=memo_type,
            headline=f"Error: {message}",
            sections=[MemoSection(
                title="Error", narrative=message,
            )],
            execution_trace=list(self._trace),
        )


# =============================================================================
# Module-level convenience — common usage patterns
# =============================================================================

def build_brain(verbose: bool = False,
                 enable_learning: Optional[bool] = None,
                 learning_engine: Any = None) -> WorkforceIntelligenceBrain:
    """Factory — makes the "new up" easy and discoverable.

    Parameters
    ----------
    verbose         : echo tool calls to stdout.
    enable_learning : override for the Human-in-the-Lead feedback layer.
                       None (default) honours the WORKFORCE_BRAIN_LEARNING
                       env var; False forces static mode.
    learning_engine : optional pre-built LearningEngine (for tests or
                       when the caller wants to inject a custom store path).
    """
    return WorkforceIntelligenceBrain(
        verbose=verbose,
        enable_learning=enable_learning,
        learning_engine=learning_engine,
    )


# Decision-matrix export — useful for auditability and documentation
def export_decision_matrix() -> List[Dict[str, str]]:
    """Return every (fit_tier, risk_level) rule as a flat list —
    suitable for CSV export, tests, and policy reviews."""
    out = []
    for (fit, risk), cell in _HIRE_DECISION_MATRIX.items():
        out.append({
            "fit_tier": fit,
            "risk_level": risk,
            "verdict": cell["verdict"],
            "rule": cell["rule"],
            "rationale": cell["rationale"],
        })
    return out
