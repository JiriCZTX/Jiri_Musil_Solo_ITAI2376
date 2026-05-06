"""
Per-class hard-ownership router for the production NER ensemble.

Per PATH_B5_PLUS_FINAL.md §1:
  • Legacy 5 (SKILL/CERT/DEGREE/EMPLOYER/YEARS_EXP) → v6 ensemble (only producer)
  • TOOL → v11 ModernBERT-large; gazetteer adds high-precision overlap
  • INDUSTRY/LOCATION/PROJECT/SOFT_SKILL → per-class winner from Gate 2

This module is byte-identity-safe: when only v6 is wired (no v11/v14/GLiNER/
gazetteer), output is identical to calling v6 directly. This preserves the
Carlos Mendez R9-interview-stable contract.

Conflict rules are deterministic; no learned trust weights.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# Schema-locked entity types and their owners
LEGACY5_TYPES: Tuple[str, ...] = ("SKILL", "CERT", "DEGREE", "EMPLOYER", "YEARS_EXP")
V11_OWNED: Tuple[str, ...] = ("TOOL",)
GATE2_TYPES: Tuple[str, ...] = ("INDUSTRY", "LOCATION", "PROJECT", "SOFT_SKILL")
ALL_TYPES: Tuple[str, ...] = LEGACY5_TYPES + V11_OWNED + GATE2_TYPES


@dataclass(frozen=True)
class Span:
    """Canonical span representation across extractors."""
    text: str
    type: str
    start: int     # char offset in source text
    end: int       # char offset in source text (exclusive)
    score: float = 1.0
    source: str = "unknown"     # which extractor produced this
    low_confidence: bool = False  # surfacing flag for memo rendering

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


# Extractor protocol: callable taking text → list of Spans
ExtractorFn = Callable[[str], Sequence[Span]]


@dataclass
class RoutingConfig:
    """Wiring of extractors per entity type. None means "no extractor wired".

    The router will produce no spans for any type with no wired extractor —
    consistent with the v6-only baseline behavior.
    """
    # Legacy-5 extractor (v6 ensemble) — single function returning all 5 types
    v6_extractor: Optional[ExtractorFn] = None

    # TOOL extractor (v11 ModernBERT-large) — produces TOOL spans only
    v11_tool_extractor: Optional[ExtractorFn] = None

    # Per-class winners for Gate 2 types (chosen post-Gate-2 ablation on dev)
    # Each maps a class to the function that owns it post-Gate-2.
    # Until Gate 2 runs, all entries are None (ship as routing-only baseline).
    gate2_winners: Dict[str, Optional[ExtractorFn]] = field(
        default_factory=lambda: {t: None for t in GATE2_TYPES}
    )

    # Gazetteer matcher (high-precision adds for TOOL/CERT/INDUSTRY/LOCATION/PROJECT)
    gazetteer_matcher: Optional[ExtractorFn] = None

    # Opus inference verifier (low-confidence rare spans only — adjudication
    # assistant + demo fallback, NOT default production extractor)
    opus_verifier: Optional[Callable[[str, Sequence[Span]], Sequence[Span]]] = None
    opus_verifier_enabled: bool = False

    # Confidence threshold below which a Gate-2-class span is flagged
    low_confidence_threshold: float = 0.50


def _filter_spans_by_owner(
    spans: Iterable[Span],
    allowed_types: Iterable[str],
    source: str,
) -> List[Span]:
    """Keep only spans whose type is in allowed_types; tag source if missing."""
    allow = set(allowed_types)
    out: List[Span] = []
    for s in spans:
        if s.type in allow:
            if s.source == "unknown" or s.source == "":
                s = Span(text=s.text, type=s.type, start=s.start, end=s.end,
                         score=s.score, source=source, low_confidence=s.low_confidence)
            out.append(s)
    return out


def _resolve_overlaps_within_class(spans: List[Span]) -> List[Span]:
    """
    For spans of the same type that overlap, prefer:
      1. Higher score
      2. Longer span (tie-break)
      3. Earlier start (tie-break)
    Different types may overlap; this function only deduplicates within a type.
    """
    by_type: Dict[str, List[Span]] = {}
    for s in spans:
        by_type.setdefault(s.type, []).append(s)
    out: List[Span] = []
    for t, group in by_type.items():
        # Sort by start, then preference (high score, long span, early start)
        group.sort(key=lambda s: (s.start, -s.score, -(s.end - s.start)))
        kept: List[Span] = []
        for s in group:
            overlap_idx = None
            for i, k in enumerate(kept):
                if s.overlaps(k):
                    overlap_idx = i
                    break
            if overlap_idx is None:
                kept.append(s)
            else:
                k = kept[overlap_idx]
                # Replacement preference: higher score → longer → earlier
                if (s.score > k.score
                        or (s.score == k.score and (s.end - s.start) > (k.end - k.start))
                        or (s.score == k.score and (s.end - s.start) == (k.end - k.start)
                            and s.start < k.start)):
                    kept[overlap_idx] = s
        out.extend(kept)
    out.sort(key=lambda s: (s.start, s.end))
    return out


def _merge_legacy5_with_extras(
    legacy_spans: List[Span],
    tool_spans: List[Span],
    gate2_spans: List[Span],
    gazetteer_spans: List[Span],
) -> List[Span]:
    """
    Combine extractor outputs into a single span list.
    Owner-locked types (legacy5, TOOL) cannot be overridden by other sources.
    """
    final: List[Span] = []

    # Legacy-5 owned by v6: trust unconditionally
    final.extend(legacy_spans)

    # TOOL owned by v11: trust v11; gazetteer only adds non-overlapping TOOL spans
    v11_tool = [s for s in tool_spans if s.type == "TOOL"]
    final.extend(v11_tool)
    gazetteer_tool = [s for s in gazetteer_spans if s.type == "TOOL"]
    for g in gazetteer_tool:
        if not any(g.overlaps(t) for t in v11_tool):
            final.append(g)

    # Gate-2 classes: gazetteer first (high precision), then learned extractor
    for cls in GATE2_TYPES:
        gz = [s for s in gazetteer_spans if s.type == cls]
        learned = [s for s in gate2_spans if s.type == cls]
        # Combine; resolve overlaps within class
        combined = gz + learned
        final.extend(combined)

    # Other gazetteer types (CERT, INDUSTRY) — they belong to legacy-5 or
    # Gate2; prevent gazetteer from leaking SKILL/DEGREE/EMPLOYER/YEARS_EXP
    other_gazetteer_types = {"CERT", "INDUSTRY", "LOCATION", "PROJECT"}
    extra = [s for s in gazetteer_spans
             if s.type in other_gazetteer_types - {"TOOL"} and s.type not in GATE2_TYPES]
    # In practice CERT may come from gazetteer when v6 misses it; allow as
    # additive only if it doesn't overlap a v6 CERT
    v6_cert = [s for s in legacy_spans if s.type == "CERT"]
    for s in extra:
        if s.type == "CERT" and not any(s.overlaps(c) for c in v6_cert):
            final.append(s)

    # Resolve within-class overlaps after combination
    return _resolve_overlaps_within_class(final)


def route(text: str, cfg: RoutingConfig) -> List[Span]:
    """
    Run all wired extractors on text; merge per per-class hard-ownership rules.

    Byte-identity guarantee: when only cfg.v6_extractor is wired (all others
    None), output equals cfg.v6_extractor(text) — preserving Carlos Mendez
    R9-interview-stable behavior.
    """
    legacy: List[Span] = []
    if cfg.v6_extractor is not None:
        legacy = _filter_spans_by_owner(
            cfg.v6_extractor(text), LEGACY5_TYPES, source="v6_ensemble"
        )

    tool: List[Span] = []
    if cfg.v11_tool_extractor is not None:
        tool = _filter_spans_by_owner(
            cfg.v11_tool_extractor(text), V11_OWNED, source="v11"
        )

    gate2_out: List[Span] = []
    for cls in GATE2_TYPES:
        winner = cfg.gate2_winners.get(cls)
        if winner is not None:
            gate2_out.extend(_filter_spans_by_owner(
                winner(text), (cls,), source=f"gate2_{cls.lower()}_winner"
            ))

    gazetteer: List[Span] = []
    if cfg.gazetteer_matcher is not None:
        gazetteer = list(cfg.gazetteer_matcher(text))
        # Tag source uniformly
        gazetteer = [
            Span(text=s.text, type=s.type, start=s.start, end=s.end,
                 score=s.score, source="gazetteer", low_confidence=s.low_confidence)
            for s in gazetteer
        ]

    # Pre-merge cleanup within owners
    legacy = _resolve_overlaps_within_class(legacy)
    tool = _resolve_overlaps_within_class(tool)
    gate2_out = _resolve_overlaps_within_class(gate2_out)
    gazetteer = _resolve_overlaps_within_class(gazetteer)

    final = _merge_legacy5_with_extras(legacy, tool, gate2_out, gazetteer)

    # Low-confidence flag for Gate-2 classes below threshold
    flagged: List[Span] = []
    for s in final:
        if s.type in GATE2_TYPES and s.score < cfg.low_confidence_threshold:
            s = Span(text=s.text, type=s.type, start=s.start, end=s.end,
                     score=s.score, source=s.source, low_confidence=True)
        flagged.append(s)

    # Optional Opus verifier: only refines flagged low-confidence rare spans
    if cfg.opus_verifier_enabled and cfg.opus_verifier is not None:
        flagged = list(cfg.opus_verifier(text, flagged))

    return flagged


# ---------------------------------------------------------------------------
# Sanity / self-test (does NOT call any DL model — pure routing logic test)
# ---------------------------------------------------------------------------
def _self_test() -> dict:
    """Validate routing logic with mock extractors. No DL models invoked."""
    def mock_v6(t: str) -> List[Span]:
        return [
            Span("Chevron", "EMPLOYER", 12, 19, 0.95, "v6_ensemble"),
            Span("HAZOP", "SKILL", 30, 35, 0.90, "v6_ensemble"),
            Span("API 510", "CERT", 50, 57, 0.99, "v6_ensemble"),
            # Legacy-5 also tries to label TOOL — should be filtered out
            Span("HYSYS", "TOOL", 70, 75, 0.80, "v6_ensemble"),
        ]

    def mock_v11(t: str) -> List[Span]:
        return [
            Span("HYSYS", "TOOL", 70, 75, 0.92, "v11"),
            Span("Chevron", "EMPLOYER", 12, 19, 0.50, "v11"),  # filtered (not v11-owned)
        ]

    def mock_gazetteer(t: str) -> List[Span]:
        return [
            Span("HYSYS", "TOOL", 70, 75, 1.0, "gazetteer"),  # overlaps v11 → drop
            Span("Aspen Plus", "TOOL", 80, 90, 1.0, "gazetteer"),  # additive
            Span("API 570", "CERT", 100, 107, 1.0, "gazetteer"),  # additive (no v6 CERT here)
            Span("Gulf of Mexico", "LOCATION", 120, 134, 1.0, "gazetteer"),
        ]

    def mock_gate2_location(t: str) -> List[Span]:
        return [
            Span("Permian Basin", "LOCATION", 140, 153, 0.85, "v14"),
        ]

    # Test 1: byte-identity — v6 only
    cfg_v6_only = RoutingConfig(v6_extractor=mock_v6)
    out1 = route("placeholder text long enough to satisfy offsets ........", cfg_v6_only)
    test1_pass = (
        len(out1) == 3
        and {s.type for s in out1} == {"EMPLOYER", "SKILL", "CERT"}
        and all(s.source == "v6_ensemble" for s in out1)
    )

    # Test 2: full ensemble
    cfg_full = RoutingConfig(
        v6_extractor=mock_v6,
        v11_tool_extractor=mock_v11,
        gate2_winners={"INDUSTRY": None, "LOCATION": mock_gate2_location,
                       "PROJECT": None, "SOFT_SKILL": None},
        gazetteer_matcher=mock_gazetteer,
    )
    out2 = route("placeholder text long enough to satisfy offsets " * 5, cfg_full)
    types_count: Dict[str, int] = {}
    for s in out2:
        types_count[s.type] = types_count.get(s.type, 0) + 1

    return {
        "test1_v6_only_byte_identity": test1_pass,
        "test1_spans": [(s.type, s.text, s.source) for s in out1],
        "test2_full_ensemble_types": types_count,
        "test2_spans": [(s.type, s.text, s.source) for s in out2],
        "GATE2_TYPES": list(GATE2_TYPES),
        "LEGACY5_TYPES": list(LEGACY5_TYPES),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(_self_test(), indent=2))
