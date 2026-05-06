"""
feedback.py — Human-in-the-Lead continual learning for the Workforce
Intelligence Brain.

The brain is deterministic by design (rule-based, template-driven, zero
LLM calls). This module adds a CONTROLLED, TRACEABLE, REVERSIBLE
adaptation layer on top: humans provide structured corrections, the
LearningEngine aggregates them, and when enough evidence accumulates the
brain starts applying the corrections to future decisions.

Core principles:
  1. ADDITIVE ONLY — the brain behaves identically to today when learning
     is disabled OR the feedback store is empty.
  2. AUDITABLE — every adjustment traces to the feedback events that
     triggered it. Every memo cites a feedback_state_hash.
  3. REVERSIBLE — the feedback log is append-only, but any adjustment can
     be unapplied by marking the relevant events applied_at=None.
  4. GOVERNED — high-impact event types (rule_override, new_intervention)
     require explicit approval, not just threshold voting.
  5. STATIC MODE — `enable_learning=False` turns the whole system off
     and restores byte-identical static behavior for reproducibility
     snapshots and regulatory audits.

This is an MVP. Production deployments need: multi-user auth, role-based
approval workflow, rollback UI beyond the single-click last-adjustment
undo, Bi-LSTM retraining trigger, drift-detection monitoring, and a real
database instead of JSONL. Each gap is explicitly commented.

Design reference:
  - "Constitutional AI" (Anthropic 2022) — explicit critique + revision loop
  - "RLHF" (Christiano et al. 2017) — reward-model style calibration
  - EU AI Act Art. 14 — human oversight of high-risk AI

Schema: see FeedbackEvent dataclass. Stored as one JSON object per line
in `data/feedback/feedback_log.jsonl`. Append-only; rewrites only occur
for the `mark_applied` operation which is explicit human override.
"""

from __future__ import annotations
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# POLICY CONSTANTS — tune per organization; every constant is surfaced here so
# a future engineer can move them to a config file without touching logic.
# =============================================================================

SCHEMA_VERSION = "1.0"
DEFAULT_STORE_PATH = Path("data/feedback/feedback_log.jsonl")
DEFAULT_MEMO_HISTORY_PATH = Path("data/feedback/memo_history.jsonl")

# How many independent same-direction corrections before an adjustment
# auto-applies. 3 is the smallest number that provides multi-user voting
# signal. Lower = faster learning, higher poisoning risk.
AUTO_APPLY_THRESHOLD = 3

# Minimum events before the confidence calibrator acts. Confidence is an
# aggregate signal so it needs more evidence than a single-rule override.
CONFIDENCE_RECALIBRATION_N = 10

# Minimum events before flipping an intervention's causal_status
CAUSAL_STATUS_FLIP_N = 5

# Event types that require explicit human approval even at threshold
APPROVAL_REQUIRED: Dict[str, bool] = {
    "verdict_correction": False,
    "rule_override": True,
    "confidence_calibration": False,
    "causal_update": False,
    "new_intervention": True,
    "general_comment": False,
}

EVENT_TYPES = tuple(APPROVAL_REQUIRED.keys())

USER_ROLES = ("chro", "hiring_manager", "hr_business_partner",
              "auditor", "demo_user", "other")


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass(frozen=True)
class FeedbackEvent:
    """Immutable record of a single human feedback action."""
    event_id: str
    timestamp: str
    user_id: str
    user_role: str
    event_type: str
    rationale: str
    confidence_in_correction: str = "medium"
    schema_version: str = SCHEMA_VERSION
    memo_id: Optional[str] = None
    department: Optional[str] = None
    brain_variant: Optional[str] = None
    original_value: Any = None
    corrected_value: Any = None
    applied_at: Optional[str] = None
    supersedes: Optional[str] = None
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeedbackEvent":
        # Gracefully ignore unknown fields (forward-compatible schema
        # evolution): drop any key the current dataclass doesn't know
        # about rather than raising.
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


def _now_iso() -> str:
    return (datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"))


def _new_event_id() -> str:
    return f"fb_{uuid.uuid4().hex[:10]}"


def _new_memo_id() -> str:
    return f"m_{uuid.uuid4().hex[:8]}"


# =============================================================================
# STORE — append-only JSONL
# =============================================================================

class FeedbackStore:
    """Append-only JSONL-backed feedback log.

    Schema versioning: each event carries `schema_version`. Readers filter
    unknown fields; writers always write the current SCHEMA_VERSION.
    Production would use a proper DB; JSONL is the MVP.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_STORE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    # ---- writes -----------------------------------------------------

    def append(self, event: FeedbackEvent) -> FeedbackEvent:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")
        return event

    def mark_applied(self, event_id: str, applied: bool = True) -> bool:
        """Toggle the applied_at timestamp of a stored event.

        Append-only purity compromise: this rewrites the JSONL file in
        place. Production would instead append a compensating event. For
        MVP this is acceptable — the semantics are clearer and the
        rollback story is a single-line flip.
        """
        events = self.load_all()
        new_events = []
        matched = False
        for e in events:
            if e.event_id == event_id:
                new_events.append(
                    replace(e, applied_at=_now_iso() if applied else None)
                )
                matched = True
            else:
                new_events.append(e)
        if matched:
            with self.path.open("w", encoding="utf-8") as f:
                for e in new_events:
                    f.write(json.dumps(e.to_dict(), default=str) + "\n")
        return matched

    def approve(self, event_id: str, approved_by: str) -> bool:
        """Mark a high-impact event (rule_override, new_intervention) as
        approved. The learning engine skips unapproved events of these
        types — approval is the governance gate."""
        events = self.load_all()
        new_events = []
        matched = False
        for e in events:
            if e.event_id == event_id:
                new_events.append(replace(
                    e, approved_at=_now_iso(), approved_by=approved_by,
                ))
                matched = True
            else:
                new_events.append(e)
        if matched:
            with self.path.open("w", encoding="utf-8") as f:
                for e in new_events:
                    f.write(json.dumps(e.to_dict(), default=str) + "\n")
        return matched

    # ---- reads ------------------------------------------------------

    def load_all(self) -> List[FeedbackEvent]:
        events: List[FeedbackEvent] = []
        if not self.path.exists():
            return events
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(FeedbackEvent.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return events

    def query(self, *,
              event_type: Optional[str] = None,
              memo_id: Optional[str] = None,
              department: Optional[str] = None,
              applied: Optional[bool] = None) -> List[FeedbackEvent]:
        result = self.load_all()
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        if memo_id:
            result = [e for e in result if e.memo_id == memo_id]
        if department:
            result = [e for e in result if e.department == department]
        if applied is True:
            result = [e for e in result if e.applied_at]
        elif applied is False:
            result = [e for e in result if not e.applied_at]
        return result

    def summarize(self) -> Dict[str, Any]:
        events = self.load_all()
        by_type = Counter(e.event_type for e in events)
        applied = sum(1 for e in events if e.applied_at)
        return {
            "total_events": len(events),
            "applied": applied,
            "pending": len(events) - applied,
            "by_type": dict(by_type),
            "latest_event_timestamp": (
                events[-1].timestamp if events else None
            ),
        }


# =============================================================================
# MEMO HISTORY — lightweight log of generated memos for the Review Queue
# =============================================================================

class MemoHistoryStore:
    """Small JSONL log of memo summaries (not full memos) for the dashboard
    Review Queue. Records only what's needed to show the memo in a list +
    let a reviewer open it. Full memo state is not persisted — Streamlit
    session state holds that within a session."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_MEMO_HISTORY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def log(self, memo_summary: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(memo_summary, default=str) + "\n")

    def load_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.path.exists():
            return rows
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-n:]


# =============================================================================
# LEARNING ENGINE — aggregate feedback into adjustments the brain can apply
# =============================================================================

class LearningEngine:
    """Computes applicable adjustments from the feedback store.

    Cache strategy: reads the store file mtime; if unchanged since last
    compute, returns cached adjustments. Otherwise recomputes. This keeps
    repeated brain calls within a session fast.
    """

    def __init__(self, store: Optional[FeedbackStore] = None):
        self.store = store or FeedbackStore()
        self._cache: Optional[Dict[str, Any]] = None
        self._cached_mtime: Optional[float] = None

    # ---- adjustment computation --------------------------------------

    def compute_adjustments(self, force: bool = False) -> Dict[str, Any]:
        if not force and not self._should_recompute():
            return self._cache
        events = self.store.load_all()

        rule_overrides = self._aggregate_rule_overrides(events)
        confidence_multipliers = self._aggregate_confidence(events)
        causal_updates = self._aggregate_causal(events)
        new_interventions = self._aggregate_new_interventions(events)

        applied_event_ids: set = set()
        for ov in rule_overrides.values():
            applied_event_ids.update(ov.get("event_ids", []))

        self._cache = {
            "rule_overrides": rule_overrides,
            "confidence_multipliers": confidence_multipliers,
            "causal_updates": causal_updates,
            "new_interventions": new_interventions,
            "state_hash": self._compute_state_hash(applied_event_ids),
            "learning_version": self._version_from_store(events),
            "n_events_total": len(events),
            "n_events_applied": sum(1 for e in events if e.applied_at),
            "applied_event_ids": sorted(applied_event_ids),
        }
        try:
            self._cached_mtime = self.store.path.stat().st_mtime
        except FileNotFoundError:
            self._cached_mtime = None
        return self._cache

    def _should_recompute(self) -> bool:
        if self._cache is None:
            return True
        try:
            mtime = self.store.path.stat().st_mtime
        except FileNotFoundError:
            return True
        return mtime != self._cached_mtime

    # ---- per-event-type aggregation ----------------------------------

    def _aggregate_rule_overrides(self, events: List[FeedbackEvent]
                                    ) -> Dict[str, Dict[str, Any]]:
        """Crowdsourced verdict corrections + approved rule_overrides."""
        out: Dict[str, Dict[str, Any]] = {}

        # Crowdsourced: count (fit_tier, risk_level) corrections
        counter: Dict[tuple, Counter] = defaultdict(Counter)
        event_ids: Dict[tuple, List[str]] = defaultdict(list)
        for e in events:
            if e.event_type != "verdict_correction" or not e.applied_at:
                continue
            ov = e.original_value or {}
            cv = e.corrected_value or {}
            if not isinstance(ov, dict) or not isinstance(cv, dict):
                continue
            ft, rl = ov.get("fit_tier"), ov.get("risk_level")
            nv = cv.get("verdict")
            if not (ft and rl and nv):
                continue
            counter[(ft, rl)][nv] += 1
            event_ids[(ft, rl)].append(e.event_id)
        for (ft, rl), c in counter.items():
            top_verdict, votes = c.most_common(1)[0]
            if votes >= AUTO_APPLY_THRESHOLD:
                out[f"{ft}|{rl}"] = {
                    "fit_tier": ft,
                    "risk_level": rl,
                    "new_verdict": top_verdict,
                    "evidence_votes": votes,
                    "source": "verdict_correction_threshold",
                    "event_ids": event_ids[(ft, rl)][:10],
                }

        # Explicit (approved) rule_overrides — take precedence over
        # crowdsourced for the same cell.
        for e in events:
            if e.event_type != "rule_override" or not e.approved_at:
                continue
            cv = e.corrected_value or {}
            if not isinstance(cv, dict):
                continue
            ft, rl, nv = cv.get("fit_tier"), cv.get("risk_level"), cv.get("new_verdict")
            if ft and rl and nv:
                out[f"{ft}|{rl}"] = {
                    "fit_tier": ft,
                    "risk_level": rl,
                    "new_verdict": nv,
                    "evidence_votes": 1,
                    "source": "rule_override_approved",
                    "event_ids": [e.event_id],
                    "approved_by": e.approved_by,
                }
        return out

    def _aggregate_confidence(self, events: List[FeedbackEvent]
                                ) -> Dict[str, float]:
        """Map confidence level → multiplier based on hit rate."""
        level_hits = defaultdict(lambda: {"correct": 0, "wrong": 0})
        for e in events:
            if e.event_type != "confidence_calibration" or not e.applied_at:
                continue
            ov = e.original_value or {}
            cv = e.corrected_value or {}
            if not isinstance(ov, dict) or not isinstance(cv, dict):
                continue
            lvl = ov.get("level")
            was_correct = cv.get("was_correct")
            if lvl and was_correct is not None:
                if was_correct:
                    level_hits[lvl]["correct"] += 1
                else:
                    level_hits[lvl]["wrong"] += 1
        multipliers: Dict[str, float] = {}
        for lvl, d in level_hits.items():
            total = d["correct"] + d["wrong"]
            if total >= CONFIDENCE_RECALIBRATION_N:
                hit_rate = d["correct"] / total
                # Cap below 1.0; never amplify — confidence calibration
                # can only de-rate trust, never inflate it.
                multipliers[lvl] = round(max(0.3, min(1.0, hit_rate + 0.1)), 2)
        return multipliers

    def _aggregate_causal(self, events: List[FeedbackEvent]
                            ) -> Dict[str, Dict[str, Any]]:
        counter: Dict[str, Counter] = defaultdict(Counter)
        event_ids: Dict[str, List[str]] = defaultdict(list)
        for e in events:
            if e.event_type != "causal_update" or not e.applied_at:
                continue
            cv = e.corrected_value or {}
            if not isinstance(cv, dict):
                continue
            interv, new_status = cv.get("intervention"), cv.get("new_status")
            if interv and new_status:
                counter[interv][new_status] += 1
                event_ids[interv].append(e.event_id)
        out: Dict[str, Dict[str, Any]] = {}
        for interv, c in counter.items():
            top_status, votes = c.most_common(1)[0]
            if votes >= CAUSAL_STATUS_FLIP_N:
                out[interv] = {
                    "new_status": top_status,
                    "votes": votes,
                    "event_ids": event_ids[interv][:10],
                }
        return out

    def _aggregate_new_interventions(self, events: List[FeedbackEvent]
                                       ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for e in events:
            if e.event_type != "new_intervention" or not e.approved_at:
                continue
            cv = e.corrected_value or {}
            if isinstance(cv, dict) and cv.get("name"):
                out[cv["name"]] = {**cv, "event_id": e.event_id}
        return out

    # ---- utilities ---------------------------------------------------

    def _compute_state_hash(self, applied_ids: set) -> str:
        joined = ",".join(sorted(applied_ids))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]

    def _version_from_store(self, events: List[FeedbackEvent]) -> str:
        applied = [e for e in events if e.applied_at]
        if not applied:
            return "v0-empty"
        return f"adaptive-v{len(applied)}"

    def get_review_priority(self, *,
                              confidence_level: Optional[str] = None,
                              conformal_width: Optional[float] = None,
                              consensus_class: Optional[str] = None
                              ) -> Dict[str, Any]:
        """Return priority + reasons for a specific memo."""
        reasons: List[str] = []
        score = 0
        if confidence_level == "low":
            reasons.append("low_confidence")
            score += 2
        elif confidence_level == "medium":
            reasons.append("medium_confidence")
            score += 1
        if conformal_width is not None and conformal_width > 0.7:
            reasons.append("wide_conformal")
            score += 2
        if consensus_class == "NO_CONSENSUS":
            reasons.append("split_consensus")
            score += 3
        elif consensus_class == "MAJORITY_CONSENSUS":
            reasons.append("majority_but_split")
            score += 1
        if score >= 3:
            priority = "HIGH"
        elif score >= 1:
            priority = "NORMAL"
        else:
            priority = "LOW"
        return {"priority": priority, "reasons": reasons, "score": score}


# =============================================================================
# CAPTURE HELPERS — convenience constructors for each event type
# =============================================================================

def capture_verdict_correction(
    store: FeedbackStore, *,
    memo_id: str, department: str,
    fit_tier: str, risk_level: str,
    original_verdict: str, corrected_verdict: str,
    user_id: str, user_role: str, rationale: str,
    confidence_in_correction: str = "medium",
) -> FeedbackEvent:
    event = FeedbackEvent(
        event_id=_new_event_id(),
        timestamp=_now_iso(),
        user_id=user_id, user_role=user_role,
        event_type="verdict_correction",
        rationale=rationale,
        confidence_in_correction=confidence_in_correction,
        memo_id=memo_id, department=department,
        original_value={"fit_tier": fit_tier, "risk_level": risk_level,
                          "verdict": original_verdict},
        corrected_value={"verdict": corrected_verdict},
        # Verdict corrections are applied to the store immediately; the
        # learning engine decides via threshold voting when to act on them.
        applied_at=_now_iso(),
    )
    return store.append(event)


def capture_rule_override(
    store: FeedbackStore, *,
    fit_tier: str, risk_level: str, new_verdict: str,
    user_id: str, user_role: str, rationale: str,
    approved_by: Optional[str] = None,
    confidence_in_correction: str = "high",
) -> FeedbackEvent:
    event = FeedbackEvent(
        event_id=_new_event_id(),
        timestamp=_now_iso(),
        user_id=user_id, user_role=user_role,
        event_type="rule_override",
        rationale=rationale,
        confidence_in_correction=confidence_in_correction,
        corrected_value={"fit_tier": fit_tier, "risk_level": risk_level,
                          "new_verdict": new_verdict},
        applied_at=_now_iso(),
        approved_at=_now_iso() if approved_by else None,
        approved_by=approved_by,
    )
    return store.append(event)


def capture_confidence_calibration(
    store: FeedbackStore, *,
    memo_id: str, original_level: str, was_correct: bool,
    user_id: str, user_role: str, rationale: str = "",
) -> FeedbackEvent:
    event = FeedbackEvent(
        event_id=_new_event_id(),
        timestamp=_now_iso(),
        user_id=user_id, user_role=user_role,
        event_type="confidence_calibration",
        rationale=rationale,
        memo_id=memo_id,
        original_value={"level": original_level},
        corrected_value={"was_correct": was_correct},
        applied_at=_now_iso(),
    )
    return store.append(event)


def capture_causal_update(
    store: FeedbackStore, *,
    intervention: str, new_status: str,
    user_id: str, user_role: str, rationale: str,
) -> FeedbackEvent:
    if new_status not in ("CAUSAL", "MIXED", "CORRELATIONAL", "NONE"):
        raise ValueError(
            f"invalid causal status '{new_status}' "
            "— must be one of CAUSAL / MIXED / CORRELATIONAL / NONE"
        )
    event = FeedbackEvent(
        event_id=_new_event_id(),
        timestamp=_now_iso(),
        user_id=user_id, user_role=user_role,
        event_type="causal_update",
        rationale=rationale,
        corrected_value={"intervention": intervention,
                          "new_status": new_status},
        applied_at=_now_iso(),
    )
    return store.append(event)


def capture_new_intervention(
    store: FeedbackStore, *,
    name: str, spec: Dict[str, Any],
    user_id: str, user_role: str, rationale: str,
    approved_by: Optional[str] = None,
) -> FeedbackEvent:
    event = FeedbackEvent(
        event_id=_new_event_id(),
        timestamp=_now_iso(),
        user_id=user_id, user_role=user_role,
        event_type="new_intervention",
        rationale=rationale,
        corrected_value={"name": name, **spec},
        applied_at=_now_iso(),
        approved_at=_now_iso() if approved_by else None,
        approved_by=approved_by,
    )
    return store.append(event)


def capture_general_comment(
    store: FeedbackStore, *,
    memo_id: Optional[str], comment: str,
    user_id: str, user_role: str,
    department: Optional[str] = None,
) -> FeedbackEvent:
    event = FeedbackEvent(
        event_id=_new_event_id(),
        timestamp=_now_iso(),
        user_id=user_id, user_role=user_role,
        event_type="general_comment",
        rationale=comment,
        memo_id=memo_id, department=department,
    )
    return store.append(event)


# =============================================================================
# ROLLBACK
# =============================================================================

def rollback_last_adjustment(store: FeedbackStore) -> Optional[FeedbackEvent]:
    """Unapply the most recent applied event. Returns the event that was
    rolled back, or None if nothing was applied."""
    events = store.load_all()
    applied = [e for e in events if e.applied_at]
    if not applied:
        return None
    latest = applied[-1]
    store.mark_applied(latest.event_id, applied=False)
    return latest


# =============================================================================
# ENV-VAR TOGGLE
# =============================================================================

def is_learning_enabled_default() -> bool:
    """Read WORKFORCE_BRAIN_LEARNING env var. Accepts on/off/true/false/1/0.

    Used as the default for build_brain when no explicit flag is passed.
    Allows system-wide static mode via a single env var in production
    deployments (e.g., for the regulatory snapshot mode)."""
    v = os.getenv("WORKFORCE_BRAIN_LEARNING", "on").lower().strip()
    return v not in ("off", "0", "false", "static", "no")
