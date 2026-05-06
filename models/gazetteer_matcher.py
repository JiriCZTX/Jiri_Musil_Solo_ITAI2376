"""
Deterministic high-precision gazetteer matcher (PATH_B5_PLUS_FINAL §1.③ +
reviewer correction §6).

Schema, per reviewer §6 — every gazetteer entry MUST carry:
    canonical_name      str
    aliases             tuple[str, ...]
    entity_type         one of {SKILL, TOOL, CERT, INDUSTRY, LOCATION, PROJECT, SOFT_SKILL}
    source_taxonomy     short tag of the taxonomy that contributed the entry
                        (e.g. "iadc", "asme", "spe", "irena", "cewd", "esco_curated", "manual")
    precision_tier      "TIER_1_HIGH"   — peer-curated, unambiguous (ships as is)
                        "TIER_2_MEDIUM" — domain-specific but with overlap risk; needs context
                        "TIER_3_LOW"    — broad / generic terms (ESCO, generic O*NET);
                                          NEVER ships as a span — kept as candidate-only
    match_mode          "WORD_BOUNDARY" — \\b...\\b regex on canonical + aliases
                        "EXACT"         — exact token-sequence match (no substring)
                        "PHRASE"        — multi-word phrase match (case-insensitive)
                        "REGEX"         — caller-supplied regex (advanced; explicit only)
    context_required    Optional[tuple[str, ...]] — if present, the match is accepted
                        only when at least one of these tokens appears within
                        ±60 chars of the match. Used for ambiguous tokens like
                        "PE" (Professional Engineer credential vs. "PE pipe"
                        material vs. "PE-IT" Italian abbreviation).

Hard rule from reviewer §6:
    "Do not auto-promote broad ESCO/O*NET terms to high-precision spans."

We enforce this by REQUIRING `precision_tier == "TIER_1_HIGH"` to ship a span.
TIER_2_MEDIUM ships only when context_required is satisfied. TIER_3_LOW NEVER
ships from this matcher; it can only enter the pipeline as a candidate via
`pipeline.jd_lf_agreement`, where it must be voted in by ≥2 independent LFs.

The matcher returns a list of `Span` (compatible with models.per_class_router)
plus per-source provenance so downstream consumers can audit which taxonomy
contributed each match.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -- minimal span type aligned with models.per_class_router.Span ---------------
@dataclass(frozen=True)
class Span:
    text: str
    type: str
    start: int
    end: int
    score: float = 1.0
    source: str = "gazetteer"
    low_confidence: bool = False


# -- Allowed entity types (matches ENTITY_SCHEMA_V7.md) -------------------------
ALLOWED_TYPES = (
    "SKILL", "TOOL", "CERT", "DEGREE", "EMPLOYER", "YEARS_EXP",
    "INDUSTRY", "LOCATION", "PROJECT", "SOFT_SKILL",
)

# -- Precision tiers -----------------------------------------------------------
TIER_1 = "TIER_1_HIGH"     # ships as a span without context check
TIER_2 = "TIER_2_MEDIUM"   # ships only when context_required tokens are nearby
TIER_3 = "TIER_3_LOW"      # never ships from this matcher; candidate-only
ALLOWED_TIERS = (TIER_1, TIER_2, TIER_3)

# -- Match modes ---------------------------------------------------------------
MATCH_WORD_BOUNDARY = "WORD_BOUNDARY"
MATCH_EXACT = "EXACT"
MATCH_PHRASE = "PHRASE"
MATCH_REGEX = "REGEX"
ALLOWED_MATCH_MODES = (MATCH_WORD_BOUNDARY, MATCH_EXACT, MATCH_PHRASE, MATCH_REGEX)


# -----------------------------------------------------------------------------
# Entry dataclass — the §6 schema
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GazetteerEntry:
    canonical_name: str
    aliases: Tuple[str, ...]
    entity_type: str
    source_taxonomy: str
    precision_tier: str
    match_mode: str
    context_required: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:  # type: ignore[misc]
        if self.entity_type not in ALLOWED_TYPES:
            raise ValueError(
                f"GazetteerEntry: invalid entity_type {self.entity_type!r}; "
                f"allowed: {ALLOWED_TYPES}"
            )
        if self.precision_tier not in ALLOWED_TIERS:
            raise ValueError(
                f"GazetteerEntry: invalid precision_tier {self.precision_tier!r}; "
                f"allowed: {ALLOWED_TIERS}"
            )
        if self.match_mode not in ALLOWED_MATCH_MODES:
            raise ValueError(
                f"GazetteerEntry: invalid match_mode {self.match_mode!r}; "
                f"allowed: {ALLOWED_MATCH_MODES}"
            )
        if self.precision_tier == TIER_2 and not self.context_required:
            raise ValueError(
                f"GazetteerEntry {self.canonical_name!r} is TIER_2_MEDIUM but "
                f"has no context_required tokens — TIER_2 requires context."
            )

    @property
    def all_forms(self) -> Tuple[str, ...]:
        seen = set()
        out: List[str] = []
        for s in (self.canonical_name, *self.aliases):
            k = s.lower()
            if k not in seen:
                seen.add(k)
                out.append(s)
        return tuple(out)


# -----------------------------------------------------------------------------
# Curated TIER_1 / TIER_2 entries (shipped today)
# -----------------------------------------------------------------------------
# Source taxonomies on disk (v7 wired):
#   cewd  — CEWD Energy Competency Model
#   spe   — SPE Petroleum Eng. Competency Model
#   iadc  — IADC Drilling Taxonomy
#   irena — IRENA Renewable Skills
#   asme  — ASME Body of Knowledge
# Plus targeted manual entries for unambiguous TOOL / CERT / LOCATION / PROJECT.
# ESCO entries are NOT auto-promoted to spans — see _esco_candidates() below.
# -----------------------------------------------------------------------------
_CURATED_ENTRIES: Tuple[GazetteerEntry, ...] = (
    # ===== TOOL (high precision; vendor + product names) =====
    GazetteerEntry("HYSYS", ("AspenTech HYSYS", "Aspen HYSYS"),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("Aspen Plus", ("AspenPlus",),
                   "TOOL", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("PHAST", ("DNV PHAST",),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("PIPESIM", ("Pipesim",),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("OLGA", ("OLGA simulator",),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY,
                   context_required=("flow", "multiphase", "simulation", "transient", "well")),
    # OLGA is TIER_2: also a Norwegian first name; require flow-domain context.
    GazetteerEntry("Bently Nevada", ("Bently Nevada System 1",),
                   "TOOL", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("AVEVA System Platform", ("AVEVA",),
                   "TOOL", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("OSIsoft PI", ("PI System", "OSIsoft AF"),
                   "TOOL", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Rockwell RSLogix", ("RSLogix 5000", "RSLogix"),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("Aspen Process Explorer", ("Aspen ProcExplorer",),
                   "TOOL", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("AutoCAD", (),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("SAP", ("SAP S/4HANA", "SAP ECC"),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY,
                   context_required=("ERP", "module", "implementation", "system",
                                     "MM", "PM", "FI", "SCM", "config", "implement")),
    # SAP must be context-required (TIER_2 effectively): "sap up the energy" etc.
    GazetteerEntry("Power BI", ("PowerBI", "MS Power BI", "Microsoft Power BI"),
                   "TOOL", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Microsoft Excel", ("MS Excel",),
                   "TOOL", "manual", TIER_1, MATCH_PHRASE),
    # "Excel" alone is risky (verb / generic noun) — TIER_2 with context.
    GazetteerEntry("Excel", (),
                   "TOOL", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("Microsoft", "spreadsheet", "VLOOKUP",
                                     "macro", "macros", "pivot", "Power Query",
                                     "VBA", "formulas", "workbook")),
    # "Python" alone is risky (snake / Monty Python / first name) — TIER_2 w/ ctx.
    GazetteerEntry("Python", (),
                   "TOOL", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("scripting", "programming", "code", "data",
                                     "ML", "machine learning", "ETL", "pandas",
                                     "numpy", "automation", "library", "Jupyter",
                                     "developer", "engineering")),
    GazetteerEntry("MATLAB", (),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("Tableau", (),
                   "TOOL", "manual", TIER_1, MATCH_WORD_BOUNDARY,
                   context_required=("dashboard", "BI", "visualization",
                                     "Tableau Desktop", "Tableau Server", "viz")),

    # ===== CERT (high precision; coded credentials) =====
    GazetteerEntry("API 510", ("API-510",),
                   "CERT", "asme", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("API 570", ("API-570",),
                   "CERT", "asme", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("API 653", ("API-653",),
                   "CERT", "asme", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("ASME BPVC Section VIII", ("BPVC Section VIII", "ASME Section VIII"),
                   "CERT", "asme", TIER_1, MATCH_PHRASE),
    GazetteerEntry("NEBOSH IGC", ("NEBOSH International General Certificate",),
                   "CERT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("PMP", ("Project Management Professional",),
                   "CERT", "manual", TIER_1, MATCH_WORD_BOUNDARY,
                   context_required=("certified", "certification", "credential",
                                     "PMI", "Project Management")),
    GazetteerEntry("CSP", ("Certified Safety Professional",),
                   "CERT", "manual", TIER_1, MATCH_WORD_BOUNDARY,
                   context_required=("safety", "BCSP", "Certified")),
    GazetteerEntry("OSHA 30", ("OSHA 30-hour",),
                   "CERT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("IADC WellCAP", ("WellCAP",),
                   "CERT", "iadc", TIER_1, MATCH_PHRASE),
    GazetteerEntry("IADC WellSharp", ("WellSharp",),
                   "CERT", "iadc", TIER_1, MATCH_PHRASE),
    GazetteerEntry("PE License", ("Professional Engineer License", "P.E. License"),
                   "CERT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("GWO Basic Safety", ("GWO BST",),
                   "CERT", "irena", TIER_1, MATCH_PHRASE),

    # ===== INDUSTRY (medium precision; many require context) =====
    GazetteerEntry("upstream oil and gas", ("upstream O&G", "upstream oil & gas"),
                   "INDUSTRY", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("midstream", (),
                   "INDUSTRY", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("pipeline", "gas", "oil", "LNG", "compressor")),
    GazetteerEntry("downstream", (),
                   "INDUSTRY", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("refining", "petrochemical", "marketing", "fuels")),
    GazetteerEntry("LNG", ("liquefied natural gas",),
                   "INDUSTRY", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("offshore wind", ("offshore wind farm",),
                   "INDUSTRY", "irena", TIER_1, MATCH_PHRASE),
    GazetteerEntry("onshore wind", (),
                   "INDUSTRY", "irena", TIER_1, MATCH_PHRASE),
    GazetteerEntry("solar PV", ("photovoltaic", "solar photovoltaic"),
                   "INDUSTRY", "irena", TIER_1, MATCH_PHRASE),
    GazetteerEntry("nuclear power", ("nuclear energy",),
                   "INDUSTRY", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("CCUS", ("carbon capture and storage",
                            "carbon capture utilization and storage"),
                   "INDUSTRY", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("hydrogen", ("green hydrogen", "blue hydrogen"),
                   "INDUSTRY", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("electrolyzer", "fuel cell", "production",
                                     "infrastructure", "economy", "blue", "green")),
    GazetteerEntry("battery storage", ("BESS", "battery energy storage"),
                   "INDUSTRY", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("petrochemicals", ("petchem",),
                   "INDUSTRY", "manual", TIER_1, MATCH_WORD_BOUNDARY),

    # ===== LOCATION (basins / offshore / megaprojects-as-location) =====
    GazetteerEntry("Gulf of Mexico", ("GoM",),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Permian Basin", (),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("North Sea", (),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Bakken", ("Bakken Shale", "Bakken Formation"),
                   "LOCATION", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("Eagle Ford", ("Eagle Ford Shale",),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Marcellus", ("Marcellus Shale",),
                   "LOCATION", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("Haynesville", (),
                   "LOCATION", "manual", TIER_1, MATCH_WORD_BOUNDARY),
    GazetteerEntry("Alberta oil sands", ("Athabasca oil sands", "oil sands"),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Houston, TX", ("Houston Texas",),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Rotterdam refinery", (),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Singapore LNG", (),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("St. James Parish, Louisiana", ("St James Parish, Louisiana",
                                                    "St. James Parish"),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Abu Dhabi", (),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Qatar", (),
                   "LOCATION", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("offshore", "LNG", "field", "asset",
                                     "Doha", "QatarEnergy", "Ras Laffan")),
    GazetteerEntry("offshore Brazil", ("offshore brazil",),
                   "LOCATION", "manual", TIER_1, MATCH_PHRASE),

    # ===== PROJECT (named megaprojects / facilities-as-projects) =====
    # Bare "Gorgon" / "HPC" / "Perdido" aliases were demoted: they collide
    # with mythology / High-Performance Computing / "lost" (Spanish). Each
    # bare-form alias became its own TIER_2 entry with context_required.
    GazetteerEntry("Gorgon LNG", ("Gorgon project",),
                   "PROJECT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Gorgon", (),
                   "PROJECT", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("Chevron", "LNG", "Australia",
                                     "offshore", "Pluto", "Wheatstone",
                                     "ExxonMobil", "Shell")),
    GazetteerEntry("Sabine Pass LNG", ("Sabine Pass",),
                   "PROJECT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Hinkley Point C", (),
                   "PROJECT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("HPC", (),
                   "PROJECT", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("Hinkley", "nuclear", "EDF", "EPR",
                                     "Somerset", "reactor", "Sizewell")),
    GazetteerEntry("Thunder Horse", ("Thunder Horse platform",),
                   "PROJECT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Perdido Spar", (),
                   "PROJECT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Perdido", (),
                   "PROJECT", "manual", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("Spar", "Gulf of Mexico", "Shell",
                                     "deepwater", "FPS", "topsides")),
    GazetteerEntry("Olympic Dam", (),
                   "PROJECT", "manual", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Dogger Bank Wind Farm", ("Dogger Bank",),
                   "PROJECT", "irena", TIER_1, MATCH_PHRASE),
    GazetteerEntry("Permian Highway Pipeline", (),
                   "PROJECT", "manual", TIER_1, MATCH_PHRASE),

    # ===== SOFT_SKILL (peer-curated CEWD only — exact phrases) =====
    GazetteerEntry("stakeholder management", (),
                   "SOFT_SKILL", "cewd", TIER_1, MATCH_PHRASE),
    GazetteerEntry("cross-functional leadership", ("cross functional leadership",),
                   "SOFT_SKILL", "cewd", TIER_1, MATCH_PHRASE),
    GazetteerEntry("change management", (),
                   "SOFT_SKILL", "cewd", TIER_2, MATCH_PHRASE,
                   context_required=("organizational", "people", "team", "transformation",
                                     "transition", "ADKAR", "Prosci")),
    # change management is dual-meaning: also a technical SKILL (ITIL change mgmt).
    # Require interpersonal context to ship as SOFT_SKILL.
    GazetteerEntry("conflict resolution", (),
                   "SOFT_SKILL", "cewd", TIER_1, MATCH_PHRASE),
    GazetteerEntry("team building", (),
                   "SOFT_SKILL", "cewd", TIER_1, MATCH_PHRASE),
    GazetteerEntry("mentoring", (),
                   "SOFT_SKILL", "cewd", TIER_2, MATCH_WORD_BOUNDARY,
                   context_required=("team", "junior", "engineers", "career",
                                     "development", "coach", "guidance",
                                     "early-career", "graduates", "leadership")),
)


# -----------------------------------------------------------------------------
# Loader: peer-curated taxonomies from data/raw/{cewd,spe,iadc,irena,asme}/flat.json
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _load_taxonomy_skills(taxonomy: str) -> List[str]:
    """Return the canonical skill phrases for one peer-curated taxonomy."""
    p = RAW_DIR / taxonomy / "flat.json"
    if not p.exists():
        return []
    with p.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        out: List[str] = []
        for v in data.values():
            if isinstance(v, list):
                out.extend(x if isinstance(x, str) else x.get("skill", "") for x in v)
            elif isinstance(v, str):
                out.append(v)
        return [s for s in out if s]
    if isinstance(data, list):
        out2: List[str] = []
        for item in data:
            if isinstance(item, str):
                out2.append(item)
            elif isinstance(item, dict):
                # SPE/IADC/IRENA/ASME format: {"skill": "...", "discipline": "..."}
                out2.append(item.get("skill", ""))
        return [s for s in out2 if s]
    return []


def _peer_curated_skill_entries() -> List[GazetteerEntry]:
    """
    Build TIER_1 SKILL gazetteer entries from peer-curated taxonomies.
    Filtering: keep only multi-word phrases (≥2 tokens) to avoid mass false-
    positives from single-word terms like "leadership" (which is SOFT_SKILL,
    not SKILL) or generic words like "communication".
    """
    entries: List[GazetteerEntry] = []
    seen: set = set()
    for taxonomy in ("spe", "iadc", "irena", "asme"):
        skills = _load_taxonomy_skills(taxonomy)
        for s in skills:
            s = s.strip()
            if len(s.split()) < 2:
                continue  # single-word entries are too risky for high precision
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append(GazetteerEntry(
                canonical_name=s,
                aliases=(),
                entity_type="SKILL",
                source_taxonomy=taxonomy,
                precision_tier=TIER_1,
                match_mode=MATCH_PHRASE,
            ))
    return entries


# -----------------------------------------------------------------------------
# ESCO: candidate-only (TIER_3_LOW) — explicitly NOT promoted to spans
# -----------------------------------------------------------------------------
# URIs whose path segment marks them as ISCED-F education-field classifiers
# rather than skills. They came in via the legacy alphabet /search fallback
# and must NOT pollute the SKILL TIER_3 candidate pool. The cleaner
# scripts/pull_esco_partial.py drops these at write time; the load-time
# filter below is defense-in-depth in case a stale on-disk artifact still
# carries ISCED-F entries.
_ESCO_ISCED_F_PREFIX = "http://data.europa.eu/esco/isced-f/"


def _esco_candidates() -> List[GazetteerEntry]:
    """
    ESCO entries enter the gazetteer as TIER_3_LOW candidates only.
    Per reviewer §6: "Do not auto-promote broad ESCO/O*NET terms to high-
    precision spans." These entries are loaded so downstream code (LF agreement)
    can use them as one of the labeling functions, but the matcher itself
    REFUSES to ship TIER_3 spans (see _ship_filter below).

    Loader prefers ``esco_candidate_skills_partial_en_v2.json`` — the
    *partial* subset (~16-17% via narrowerConcept walk + alphabet /search
    fallback, ISCED-F filtered, deduped, produced by
    ``scripts/pull_esco_partial.py``). Falls back to the legacy 200-entry
    file (``esco_skills_full_en.json``) when the partial v2 artifact is not
    present. The newer file uses a dict-wrapped schema
    (``{"_meta": {...}, "records": [...]}``); the legacy file is a flat list.
    Both are accepted here.

    Defense-in-depth: any record whose URI starts with the ISCED-F prefix is
    dropped at load time even if it slipped through the artifact builder.
    This keeps the SKILL TIER_3 candidate pool free of academic-field codes.
    """
    p_partial = RAW_DIR / "esco" / "esco_candidate_skills_partial_en_v2.json"
    p_legacy = RAW_DIR / "esco" / "esco_skills_full_en.json"
    p = p_partial if p_partial.exists() else p_legacy
    if not p.exists():
        return []
    with p.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        items = data["records"]
    elif isinstance(data, list):
        items = data
    else:
        return []
    out: List[GazetteerEntry] = []
    seen: set = set()
    for item in items:
        # Defense-in-depth: drop ISCED-F education-field URIs.
        uri = item.get("uri") or ""
        if isinstance(uri, str) and uri.startswith(_ESCO_ISCED_F_PREFIX):
            continue
        title = (item.get("title") or "").strip()
        if not title or len(title.split()) < 2:
            continue
        if title.lower() in seen:
            continue
        seen.add(title.lower())
        alts: Tuple[str, ...] = ()
        alt_block = item.get("alternativeLabel")
        if isinstance(alt_block, dict):
            en_alts = alt_block.get("en") or []
            if isinstance(en_alts, list):
                alts = tuple(s for s in en_alts
                             if isinstance(s, str) and s and s.lower() != title.lower())
        # TIER_3_LOW: matcher refuses to ship these via .match().
        out.append(GazetteerEntry(
            canonical_name=title,
            aliases=alts,
            entity_type="SKILL",
            source_taxonomy="esco",
            precision_tier=TIER_3,
            match_mode=MATCH_PHRASE,
        ))
    return out


# -----------------------------------------------------------------------------
# Compiled regex per entry (cached on first build)
# -----------------------------------------------------------------------------
def _compile_pattern(entry: GazetteerEntry) -> re.Pattern[str]:
    forms = entry.all_forms
    escaped = sorted(set(re.escape(f) for f in forms), key=len, reverse=True)
    body = "(?:" + "|".join(escaped) + ")"
    if entry.match_mode == MATCH_WORD_BOUNDARY:
        return re.compile(rf"\b{body}\b", re.IGNORECASE)
    if entry.match_mode == MATCH_PHRASE:
        # Collapse whitespace-tolerant phrase
        ws_body = "(?:" + "|".join(
            re.escape(f).replace(r"\ ", r"\s+") for f in forms
        ) + ")"
        return re.compile(rf"\b{ws_body}\b", re.IGNORECASE)
    if entry.match_mode == MATCH_EXACT:
        return re.compile(rf"(?<!\w){body}(?!\w)")
    if entry.match_mode == MATCH_REGEX:
        return re.compile(forms[0], re.IGNORECASE)
    raise ValueError(f"unsupported match_mode: {entry.match_mode}")


@dataclass
class GazetteerMatcher:
    entries: Tuple[GazetteerEntry, ...]
    _compiled: Tuple[Tuple[GazetteerEntry, re.Pattern[str]], ...] = field(
        init=False, default_factory=tuple
    )

    def __post_init__(self) -> None:  # type: ignore[misc]
        compiled = tuple((e, _compile_pattern(e)) for e in self.entries)
        # dataclass-frozen=False so we can assign here
        object.__setattr__(self, "_compiled", compiled)

    def match(self, text: str) -> List[Span]:
        """Return Spans for every TIER_1_HIGH match plus context-validated
        TIER_2_MEDIUM matches. TIER_3_LOW NEVER ships — caller must use
        candidate paths if it wants ESCO etc.

        context_required is enforced for ALL tiers when present (not just
        TIER_2). This lets a TIER_1 entry with a context guard (e.g. OLGA,
        SAP) ship with the same precision discipline as TIER_2.

        Containment dedup (cross-type): when a shorter span is fully
        contained inside a longer span of the same entity_type (e.g. bare
        "Gorgon" inside "Gorgon LNG" both → PROJECT), drop the shorter one.
        Different-type containment is preserved (e.g. INDUSTRY "LNG"
        inside PROJECT "Gorgon LNG" — different types, keep both)."""
        spans: List[Span] = []
        seen_ranges: set = set()  # (start, end, type) dedup
        for entry, pat in self._compiled:
            if not _ship_filter(entry):
                continue
            for m in pat.finditer(text):
                start, end = m.start(), m.end()
                if entry.context_required and not _context_ok(
                    text, start, end, entry.context_required
                ):
                    continue
                key = (start, end, entry.entity_type)
                if key in seen_ranges:
                    continue
                seen_ranges.add(key)
                spans.append(Span(
                    text=text[start:end],
                    type=entry.entity_type,
                    start=start,
                    end=end,
                    score=1.0 if entry.precision_tier == TIER_1 else 0.85,
                    source=f"gazetteer:{entry.source_taxonomy}",
                ))

        # Containment dedup (Day-2 launch correction #4): cross-type, not
        # only same-type. When a span is fully contained inside a strictly
        # longer span — REGARDLESS of type — drop the shorter, more-generic
        # one. This is what makes "Gorgon LNG" (PROJECT) win over its
        # nested "LNG" (INDUSTRY), and "Microsoft Excel" win over "Excel".
        # Equal-length distinct spans are kept (they don't fit one inside
        # the other so the rule doesn't fire).
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        kept: List[Span] = []
        for s in spans:
            redundant = False
            for k in kept:
                if k.start <= s.start and s.end <= k.end and \
                        (k.end - k.start) > (s.end - s.start):
                    # k is strictly longer and fully contains s (any type).
                    redundant = True
                    break
            if not redundant:
                # Also remove anything in `kept` that this NEW longer span swallows
                kept = [
                    k for k in kept
                    if not (s.start <= k.start and k.end <= s.end
                            and (s.end - s.start) > (k.end - k.start))
                ]
                kept.append(s)
        kept.sort(key=lambda s: (s.start, s.end))
        return kept

    def candidates(self, text: str) -> List[Span]:
        """Return TIER_3_LOW *candidate* matches — NOT shipped as spans by the
        production router. Used by `pipeline.jd_lf_agreement` as one of the
        labeling functions."""
        out: List[Span] = []
        for entry, pat in self._compiled:
            if entry.precision_tier != TIER_3:
                continue
            for m in pat.finditer(text):
                out.append(Span(
                    text=text[m.start():m.end()],
                    type=entry.entity_type,
                    start=m.start(),
                    end=m.end(),
                    score=0.50,
                    source=f"gazetteer_candidate:{entry.source_taxonomy}",
                    low_confidence=True,
                ))
        out.sort(key=lambda s: (s.start, s.end))
        return out


def _ship_filter(entry: GazetteerEntry) -> bool:
    """Return True if this entry is allowed to ship as a span via .match().
    Hard rule from reviewer §6: TIER_3_LOW NEVER ships."""
    return entry.precision_tier in (TIER_1, TIER_2)


def _context_ok(text: str, start: int, end: int,
                context_tokens: Sequence[str]) -> bool:
    """Check that at least one context token appears within ±60 chars of
    the match. Case-insensitive substring check."""
    if not context_tokens:
        return True
    window_start = max(0, start - 60)
    window_end = min(len(text), end + 60)
    window = text[window_start:window_end].lower()
    return any(tok.lower() in window for tok in context_tokens)


# -----------------------------------------------------------------------------
# Broad-term blacklist + per-canonical demotions (reviewer follow-up #1)
# -----------------------------------------------------------------------------
# Any ship-eligible (TIER_1 / TIER_2) entry whose canonical_name OR an alias
# exactly matches one of these is demoted to TIER_3_LOW (candidate-only) at
# matcher-build time. This is enforced HERE rather than at the entry literal
# so the demotion list is auditable in one place.
BROAD_TERMS_BLACKLIST: Tuple[str, ...] = (
    "energy", "power", "operations", "leadership", "management",
    "communication", "project management", "agile",
    # Dual-use HR concepts that creep into peer-curated SKILL lists
    "performance management", "professional development", "general management",
)

# Per-canonical demotions: domain-specific, but better lived in another type.
# Skill canonicals that already exist as a curated higher-priority type
# (e.g., "stakeholder management" exists as a curated SOFT_SKILL → SKILL dup
# from SPE list is dropped).
_CURATED_HIGHER_PRIORITY_BY_NAME: Dict[str, str] = {
    e.canonical_name.lower().strip(): e.entity_type
    for e in _CURATED_ENTRIES
    if e.entity_type in ("SOFT_SKILL", "TOOL", "CERT", "INDUSTRY",
                         "LOCATION", "PROJECT")
}


def _is_blacklisted_form(forms: Sequence[str]) -> bool:
    s = {f.lower().strip() for f in forms}
    return bool(s & set(BROAD_TERMS_BLACKLIST))


def _apply_blacklist_and_dedup(
    entries: Sequence[GazetteerEntry],
) -> List[GazetteerEntry]:
    """
    Two-pass cleanup applied at matcher build time:
      1. BROAD_TERMS_BLACKLIST: demote any entry whose canonical_name or any
         alias exactly matches one of the dangerous broadcasts. The entry
         survives as TIER_3_LOW so JD-LF candidates can still see it.
      2. Higher-priority dedup: peer-curated SKILL entries whose canonical
         already exists as a curated higher-priority type (e.g.
         SOFT_SKILL "stakeholder management" wins over SPE-SKILL
         "stakeholder management") are dropped entirely from the SKILL bucket.
    """
    out: List[GazetteerEntry] = []
    for e in entries:
        forms = (e.canonical_name, *e.aliases)
        # (1) blacklist demotion
        if e.precision_tier in (TIER_1, TIER_2) and _is_blacklisted_form(forms):
            out.append(GazetteerEntry(
                canonical_name=e.canonical_name,
                aliases=e.aliases,
                entity_type=e.entity_type,
                source_taxonomy=e.source_taxonomy,
                precision_tier=TIER_3,
                match_mode=e.match_mode,
                # context_required is only required for TIER_2; OK to drop
                context_required=None,
            ))
            continue
        # (2) higher-priority dedup
        if e.entity_type == "SKILL":
            other_type = _CURATED_HIGHER_PRIORITY_BY_NAME.get(
                e.canonical_name.lower().strip()
            )
            if other_type is not None and other_type != "SKILL":
                # Drop the duplicate SKILL — the curated higher-priority entry
                # already covers this surface form.
                continue
        out.append(e)
    return out


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------
def build_default_matcher(*, include_peer_skills: bool = True,
                          include_esco_candidates: bool = True) -> GazetteerMatcher:
    """
    Build the default production matcher from:
      - The curated TIER_1 / TIER_2 entries above.
      - Peer-curated TIER_1 SKILLs from SPE/IADC/IRENA/ASME (multi-word only).
      - ESCO entries as TIER_3_LOW candidates (kept for LF agreement use).

    Applies BROAD_TERMS_BLACKLIST demotion + higher-priority dedup at the end.
    """
    entries: List[GazetteerEntry] = list(_CURATED_ENTRIES)
    if include_peer_skills:
        entries.extend(_peer_curated_skill_entries())
    if include_esco_candidates:
        entries.extend(_esco_candidates())
    cleaned = _apply_blacklist_and_dedup(entries)
    return GazetteerMatcher(entries=tuple(cleaned))


# -----------------------------------------------------------------------------
# Self-test (does not depend on any DL model)
# -----------------------------------------------------------------------------
def _self_test() -> dict:
    matcher = build_default_matcher()
    # Each fixture: (text, {entity_type: [expected_substrings_in_any_order]}).
    # Empty list means the matcher MUST NOT emit any span of that type.
    fixtures = [
        # --- previously-passing baseline -----------------------------------
        ("12 years at Chevron, HYSYS for process simulation, Aspen Plus dynamic "
         "modeling, ASME BPVC Section VIII certified.",
         {"TOOL": ["HYSYS", "Aspen Plus"], "CERT": ["ASME BPVC Section VIII"]}),
        ("Worked the Permian Basin and Eagle Ford Shale, Houston, TX office.",
         {"LOCATION": ["Permian Basin", "Eagle Ford", "Houston, TX"]}),
        ("Lead engineer on Gorgon LNG and the Dogger Bank Wind Farm expansion.",
         {"PROJECT": ["Gorgon LNG", "Dogger Bank Wind Farm"]}),
        ("Strong stakeholder management and cross-functional leadership.",
         {"SOFT_SKILL": ["stakeholder management", "cross-functional leadership"]}),
        ("Maria Olga is the team lead.",  # OLGA must NOT match — no flow context
         {"TOOL": []}),
        ("OLGA multiphase flow simulator transient runs on the deepwater well.",
         {"TOOL": ["OLGA"]}),
        ("midstream pipeline operations across the Bakken.",
         {"INDUSTRY": ["midstream"], "LOCATION": ["Bakken"]}),
        ("midstream of the river bend, kayaking trip last weekend.",  # NO INDUSTRY ship
         {"INDUSTRY": []}),
        ("API 510 inspector, API 570 piping, OSHA 30-hour cert.",
         {"CERT": ["API 510", "API 570", "OSHA 30"]}),

        # --- adversarial: API alone is NOT a CERT ---------------------------
        ("API standards interpretation across upstream operations.",
         {"CERT": []}),  # bare "API" must NOT match any CERT
        ("Worked closely with API consultants on regulatory compliance.",
         {"CERT": []}),

        # --- adversarial: LNG industry vs Gorgon LNG project ----------------
        # "midstream" also matches INDUSTRY here because "pipeline" is in its
        # context_required list — that's the correct behaviour. We assert that
        # bare "LNG" matches INDUSTRY (NOT PROJECT) and the project list stays empty.
        ("LNG market grew 12% YoY across midstream pipeline operations.",
         {"INDUSTRY": ["LNG", "midstream"], "PROJECT": []}),
        ("Lead engineer on Gorgon LNG, supporting Sabine Pass LNG commissioning.",
         {"PROJECT": ["Gorgon LNG", "Sabine Pass LNG"]}),
        # St. James Parish LNG terminal: LOCATION + (no PROJECT alias for it)
        ("Plant Manager at the St. James Parish, Louisiana LNG terminal expansion.",
         {"LOCATION": ["St. James Parish, Louisiana"]}),

        # --- adversarial: Power BI vs power ---------------------------------
        ("Built Power BI dashboards for the executive team across operations.",
         {"TOOL": ["Power BI"]}),
        ("nuclear power plant operations and grid stability.",
         # "nuclear power" matches INDUSTRY; bare "power" does not.
         {"INDUSTRY": ["nuclear power"], "TOOL": []}),
        ("She has serious staying power on long shifts at the rig.",
         # "power" alone must NOT match any type
         {"INDUSTRY": [], "TOOL": []}),

        # --- adversarial: Python with vs without context --------------------
        ("Built ETL pipelines in Python using pandas for data engineering.",
         {"TOOL": ["Python"]}),
        ("Saw a python at the petting zoo last weekend.",
         {"TOOL": []}),  # snake context — TIER_2 context check rejects

        # --- adversarial: Excel with vs without context ---------------------
        # Same-type containment drops bare "Excel" inside "Microsoft Excel".
        ("Microsoft Excel for VBA macros and pivot reporting.",
         {"TOOL": ["Microsoft Excel"]}),
        ("Excel pivot tables and VBA macros for monthly reporting.",
         {"TOOL": ["Excel"]}),  # bare Excel + spreadsheet context fires
        ("She tends to excel under pressure during incident response drills.",
         {"TOOL": []}),  # verb context — TIER_2 context check rejects

        # --- adversarial: broad-term blacklist enforcement ------------------
        # "energy" / "power" / "leadership" / "management" / "communication"
        # / "project management" / "Agile" must never ship as a span.
        ("energy at Chevron with strong communication and project management "
         "skills, supporting power, operations, leadership, and management "
         "across teams using Agile.",
         {"SKILL": [], "SOFT_SKILL": [], "INDUSTRY": [], "TOOL": []}),

        # --- adversarial: HPC needs nuclear context (vs High-Performance Computing) ---
        ("Engineer on Hinkley Point C nuclear reactor commissioning, EDF.",
         {"PROJECT": ["Hinkley Point C"]}),  # only the canonical phrase appears
        ("Worked on the HPC nuclear EDF Somerset project, EPR reactor design.",
         {"PROJECT": ["HPC"]}),  # bare HPC + nuclear context fires
        ("Optimised HPC GPU clusters for ML training on AWS.",
         {"PROJECT": []}),  # HPC = High-Performance Computing here, must NOT match
        # --- adversarial: Perdido needs offshore/Spar context (vs "lost") ---
        # "Perdido Spar" is canonical (PHRASE); same-type containment dedup
        # drops the bare "Perdido" inside that span.
        ("Perdido Spar topsides, Shell, Gulf of Mexico deepwater operations.",
         {"PROJECT": ["Perdido Spar"]}),
        ("Es perdido en el desierto.",   # Spanish for "lost in the desert"
         {"PROJECT": []}),
        # --- adversarial: Gorgon needs LNG/Australia context (vs mythology) ---
        # Same-type containment drops bare "Gorgon" when "Gorgon LNG" matches.
        ("Lead engineer on Gorgon LNG Australia commissioning, Chevron operator.",
         {"PROJECT": ["Gorgon LNG"]}),
        ("In Greek mythology, the Gorgon Medusa turned men to stone.",
         {"PROJECT": []}),
        # --- adversarial: mentoring needs leadership/development context ---
        ("Strong mentoring of junior engineers across the team.",
         {"SOFT_SKILL": ["mentoring"]}),
        ("The trainer was mentoring the puppy with positive reinforcement.",
         {"SOFT_SKILL": []}),

        # --- Day-2 launch correction #4: cross-type containment fixtures ---
        # 1. Gorgon LNG (PROJECT) vs LNG (INDUSTRY) — cross-type containment
        #    drops the inner INDUSTRY span when the outer PROJECT span fires.
        ("Lead engineer on the Gorgon LNG commissioning team, Chevron Australia.",
         {"PROJECT": ["Gorgon LNG"], "INDUSTRY": []}),
        # The same span elsewhere in the doc, far from the PROJECT match,
        # still ships as INDUSTRY.
        ("LNG market grew 12%. Separately, the Gorgon LNG project launched.",
         {"PROJECT": ["Gorgon LNG"], "INDUSTRY": ["LNG"]}),

        # 2. API 570 (CERT) vs bare API — bare "API" is NOT in the gazetteer
        #    so it never ships, while "API 570" matches CERT.
        ("Holds API 570 piping inspection certification.",
         {"CERT": ["API 570"]}),
        ("Worked with API consultants on the standards update.",
         {"CERT": []}),  # bare "API" without digit must NOT ship

        # 3. Microsoft Excel (TOOL) vs Excel (TOOL) — same type, containment
        #    dedup drops the contained span.
        ("Microsoft Excel for VBA macros, pivot reporting and workbook design.",
         {"TOOL": ["Microsoft Excel"]}),

        # 4. Hinkley Point C (PROJECT) vs HPC (PROJECT) — different match
        #    sites in text, both ship; same-type but no containment.
        ("Worked at Hinkley Point C as well as the HPC nuclear EDF Somerset programme.",
         {"PROJECT": ["Hinkley Point C", "HPC"]}),
    ]
    issues: List[str] = []
    for text, expected in fixtures:
        spans = matcher.match(text)
        for etype, want in expected.items():
            got = sorted({s.text for s in spans if s.type == etype}, key=str.lower)
            want_norm = sorted({w for w in want}, key=str.lower)
            got_lower = {g.lower() for g in got}
            want_lower = {w.lower() for w in want_norm}
            if want_lower != got_lower:
                issues.append(
                    f"text={text!r} type={etype}: want={want_norm} got={got}"
                )
    # Also verify TIER_3_LOW does not ship
    esco_ship_count = sum(
        1 for s in matcher.match(
            "data analytics, machine learning, financial reporting, "
            "human resources management, project management."
        )
        if s.source.startswith("gazetteer:esco")
    )
    if esco_ship_count > 0:
        issues.append(f"ESCO TIER_3 leaked {esco_ship_count} ships — must be 0")

    # Verify blacklist demotion happened: no ship-eligible canonical_name
    # equals a BROAD_TERMS_BLACKLIST entry.
    bl = set(BROAD_TERMS_BLACKLIST)
    leaked_blacklist = [e.canonical_name for e in matcher.entries
                        if e.precision_tier in (TIER_1, TIER_2)
                        and e.canonical_name.lower().strip() in bl]
    if leaked_blacklist:
        issues.append(f"BROAD_TERMS leaked into ship-eligible: {leaked_blacklist}")

    return {
        "n_entries": len(matcher.entries),
        "n_ship_eligible": sum(1 for e in matcher.entries
                                if e.precision_tier in (TIER_1, TIER_2)),
        "n_tier3_candidates": sum(1 for e in matcher.entries
                                   if e.precision_tier == TIER_3),
        "fixtures_tested": len(fixtures),
        "issues": issues,
        "esco_ship_count": esco_ship_count,
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(_self_test(), indent=2))
