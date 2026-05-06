"""
CrewAI Custom Tools for Agent 1: Talent Intelligence Agent.

Wraps the DistilBERT NER model and SBERT matcher as callable tools
that CrewAI can invoke during its ReAct reasoning loop.

The LLM brain (Agent Brain) uses these tools to:
1. Extract entities from resumes (NER tool)
2. Match candidates to jobs (SBERT tool)
3. Diff candidate vs job requirements to surface concrete skill gaps (gap tool)
4. Look up standardized skills (O*NET tool)
"""
import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from crewai.tools import tool
from typing import Optional


# Global model instances (initialized in main.py)
_ner_engine = None
_sbert_matcher = None


# Noise filters for entities extracted from job descriptions.
# The NER model was trained on resumes; JDs use different phrasing patterns
# ("Experience with X", "Ideal candidate", trailing " skills") that leak into
# extracted spans. These filters clean JD-side spans before the SBERT diff.
_JD_NOISE_SPANS = {
    "ideal candidate", "candidate", "ideal", "seeking", "looking for",
    "role", "position", "opening", "opportunity", "applicants",
    "experience", "background", "knowledge", "expertise", "familiarity",
    "skills", "skill", "ability", "proficiency", "proficient",
    "required", "preferred", "plus", "must", "must have", "strong",
    "hands-on", "hands on", "ideal for", "we are hiring", "we are seeking",
}
_JD_PREFIX_STRIPS = [
    "experience with ", "experience in ", "experience of ",
    "familiarity with ", "familiar with ", "proficiency in ",
    "proficient in ", "knowledge of ", "expertise in ", "expert in ",
    "skilled in ", "skills in ", "strong ", "hands-on experience with ",
    "hands on experience with ", "must have ", "must be proficient in ",
    "must be experienced in ", "must hold ",
]
_JD_SUFFIX_STRIPS = [
    " skills", " experience", " ability", " abilities", " required",
    " preferred", " a plus", " is a plus", " required.", " preferred.",
]


def _clean_job_spans(span_list, job_title=None):
    """Strip JD-style phrasing noise from extracted job entity spans."""
    cleaned = []
    seen = set()
    title_norm = job_title.lower().strip() if job_title else None
    for s in span_list:
        text = re.sub(r'\s+', ' ', s).strip().strip('.,;:')
        low = text.lower()
        # Drop the job title if it leaked in as a requirement
        if title_norm and (low == title_norm or low in title_norm):
            continue
        # Strip leading phrasing patterns
        for p in _JD_PREFIX_STRIPS:
            if low.startswith(p):
                text = text[len(p):]
                low = text.lower()
                break
        # Strip trailing phrasing patterns
        for suf in _JD_SUFFIX_STRIPS:
            if low.endswith(suf):
                text = text[:-len(suf)].rstrip(' .,;:')
                low = text.lower()
                break
        # Drop pure noise spans
        if low in _JD_NOISE_SPANS:
            continue
        # Drop empty or single-char residue
        if len(text.strip()) < 2:
            continue
        # Dedupe (case-insensitive)
        key = low
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text.strip())
    return cleaned


def _extract_job_title(job_text):
    """Best-effort first-line job title — used to filter title leaks from NER."""
    first = job_text.strip().splitlines()[0] if job_text.strip() else ""
    # Handle "Title — Context" or "Title - Context" headers
    for sep in (" — ", " - "):
        if sep in first:
            first = first.split(sep)[0]
            break
    return first.strip()


def set_ner_engine(engine):
    global _ner_engine
    _ner_engine = engine


def set_sbert_matcher(matcher):
    global _sbert_matcher
    _sbert_matcher = matcher


@tool("Resume Entity Extractor")
def extract_resume_entities(resume_text: str) -> str:
    """
    Extract named entities from a resume using the fine-tuned DistilBERT NER model.
    Identifies: SKILL, CERT (certifications), DEGREE, EMPLOYER, YEARS_EXP.
    Input: Raw resume text as a string.
    Output: JSON with extracted entities grouped by type.
    """
    if _ner_engine is None:
        return json.dumps({"error": "NER model not initialized"})

    entities = _ner_engine.extract_entities(resume_text)

    summary = {
        "skills": entities.get("SKILL", []),
        "certifications": entities.get("CERT", []),
        "degrees": entities.get("DEGREE", []),
        "employers": entities.get("EMPLOYER", []),
        "years_experience": entities.get("YEARS_EXP", []),
        "total_entities_found": sum(len(v) for v in entities.values()),
    }

    return json.dumps(summary, indent=2)


@tool("Candidate-Job Matcher")
def match_candidate_to_job(candidate_text: str, job_text: str) -> str:
    """
    Compute semantic similarity between a candidate profile and job description.
    Uses Sentence-BERT embeddings and cosine similarity.
    Input: candidate_text (resume or profile), job_text (job description).
    Output: JSON with match score, tier, and recommendation.
    """
    if _sbert_matcher is None:
        return json.dumps({"error": "SBERT matcher not initialized"})

    score = _sbert_matcher.compute_match_score(candidate_text, job_text)
    tier = _sbert_matcher._score_to_tier(score)

    result = {
        "match_score": round(score, 4),
        "match_tier": tier,
        "is_qualified": score >= _sbert_matcher.threshold,
        "recommendation": _generate_recommendation(score, tier),
    }

    return json.dumps(result, indent=2)


# -----------------------------------------------------------------------------
# Best-in-class skill-gap helpers
# -----------------------------------------------------------------------------

# Criticality signals — markers in the JD text near each requirement.
_CRITICAL_PATTERNS = [
    r'\brequired\b', r'\bmust have\b', r'\bmust be\b', r'\bmust hold\b',
    r'\bmandatory\b', r'\bessential\b', r'\brequired\.\b',
]
_PREFERRED_PATTERNS = [
    r'\bpreferred\b', r'\bstrong plus\b', r'\bhighly desired\b',
    r'\bshould have\b', r'\bdesirable\b',
]
_NICE_TO_HAVE_PATTERNS = [
    r'\ba plus\b', r'\bis a plus\b', r'\bnice to have\b',
    r'\bwould be a plus\b', r'\bbonus\b',
]

# Seniority lexicon — ordered so we pick the max level signal found.
# 0=entry, 8=C-suite. Used for both job titles and resume headlines.
_SENIORITY_LEVELS = [
    ("chief", 8), ("cto", 8), ("ceo", 8), ("cfo", 8),
    ("vp of", 7), ("vp ", 7), ("vice president", 7),
    ("director", 6), ("head of", 6),
    ("principal", 5),
    ("lead ", 4), (" lead", 4), ("staff ", 4), ("manager", 4),
    ("supervisor", 4), ("superintendent", 4),
    ("senior ", 3), ("sr.", 3), ("sr ", 3),
    ("mid-level", 2), ("mid level", 2), ("regular", 2),
    ("associate ", 1), ("junior", 1), ("entry-level", 0), ("entry level", 0),
]
_SENIORITY_LABELS = {
    0: "Entry-level", 1: "Junior/Associate", 2: "Mid-level",
    3: "Senior", 4: "Lead/Manager", 5: "Principal",
    6: "Director", 7: "VP", 8: "C-suite",
}


def _detect_criticality(span, job_text_lower, window=120):
    """
    Locate `span` inside the lower-cased JD text and inspect a ±N-char
    window for criticality markers. Returns one of:
      CRITICAL / PREFERRED / NICE_TO_HAVE / STANDARD
    """
    low_span = span.lower()
    pos = job_text_lower.find(low_span)
    if pos < 0:
        # Fallback: try stripping punctuation
        stripped = re.sub(r'[^\w\s]', '', low_span)
        if stripped and stripped != low_span:
            pos = job_text_lower.find(stripped)
    if pos < 0:
        return "STANDARD"
    start = max(0, pos - window)
    end = min(len(job_text_lower), pos + len(span) + window)
    context = job_text_lower[start:end]
    for pat in _CRITICAL_PATTERNS:
        if re.search(pat, context):
            return "CRITICAL"
    for pat in _PREFERRED_PATTERNS:
        if re.search(pat, context):
            return "PREFERRED"
    for pat in _NICE_TO_HAVE_PATTERNS:
        if re.search(pat, context):
            return "NICE_TO_HAVE"
    return "STANDARD"


def _detect_seniority(text, default=2):
    """Return numeric seniority level (0-8) from first ~300 chars of text."""
    head = text[:300].lower()
    best = None
    for kw, lvl in _SENIORITY_LEVELS:
        if kw in head:
            if best is None or lvl > best:
                best = lvl
    return best if best is not None else default


def _seniority_alignment(job_text, candidate_text):
    """Compare seniority levels and report delta + alignment."""
    job_level = _detect_seniority(job_text)
    cand_level = _detect_seniority(candidate_text)
    delta = cand_level - job_level
    if abs(delta) <= 1:
        alignment = "aligned"
    elif delta < -1:
        alignment = "candidate_under_leveled"
    else:
        alignment = "candidate_over_leveled"
    return {
        "job_level": job_level,
        "job_level_label": _SENIORITY_LABELS.get(job_level, "?"),
        "candidate_level": cand_level,
        "candidate_level_label": _SENIORITY_LABELS.get(cand_level, "?"),
        "delta": delta,
        "alignment": alignment,
    }


# -----------------------------------------------------------------------------
# Candidate-side evidence extraction — for each matched/weak item, pull
# the supporting sentence from the resume, detect proficiency, count
# mentions, and extract quantified achievements (%, $, numeric impact).
# Produces auditable, hiring-manager-friendly match evidence.
# -----------------------------------------------------------------------------

# Proficiency modifiers ordered high → low. Longest strings matched first
# (handled at use site).
_PROFICIENCY_HIGH = [
    "expertise in", "expert in", "led ", "designed ", "managed ",
    "principal ", "architected ", "deployed ", "implemented ",
    "drove ", "built ", "optimized ",
]
_PROFICIENCY_MEDIUM = [
    "proficient in", "skilled in", "experienced in", "worked on",
    "core skills", "core skill", "core ",
]
_PROFICIENCY_LOW = [
    "familiar with", "exposure to", "aware of", "some experience",
    "basic ", "introduction to",
]


def _find_supporting_sentence(span, candidate_text):
    """
    Find the sentence inside `candidate_text` that contains `span`.
    Expands from the span's position to the nearest sentence boundaries
    (., !, ?, newline). Returns None if span is not present.
    """
    if not span or not candidate_text:
        return None
    low_text = candidate_text.lower()
    low_span = span.lower()
    pos = low_text.find(low_span)
    if pos < 0:
        return None
    # Expand backward to previous sentence boundary
    start = pos
    while start > 0 and candidate_text[start - 1] not in ".!?\n":
        start -= 1
    # Expand forward to next sentence boundary
    end = pos + len(span)
    while end < len(candidate_text) and candidate_text[end] not in ".!?\n":
        end += 1
    sentence = candidate_text[start:end].strip().strip(".!?,;:")
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence if len(sentence) >= 8 else None


def _detect_proficiency(span, candidate_text, window=80):
    """
    Scan a window BEFORE the span for proficiency markers and return
    'high' / 'medium' / 'low' / None.
    """
    if not span or not candidate_text:
        return None
    low_text = candidate_text.lower()
    low_span = span.lower()
    pos = low_text.find(low_span)
    if pos < 0:
        return None
    start = max(0, pos - window)
    before = low_text[start:pos]
    for m in _PROFICIENCY_HIGH:
        if m in before:
            return "high"
    for m in _PROFICIENCY_MEDIUM:
        if m in before:
            return "medium"
    for m in _PROFICIENCY_LOW:
        if m in before:
            return "low"
    return None


_QUANT_PATTERNS = [
    r"\$[\d.,]+\s?[MBKmbk]?\b",             # $2.1B, $3.2M, $200K
    r"\d+(?:\.\d+)?\s?%",                   # 45%, 15.2%
    r"\b\d+(?:,\d{3})+\s?(?:bpd|MW|MMcf|mcf|bbl|TPD|tph)\b",  # 150,000 bpd, 300 MW
    r"\b\d+\s?(?:MW|GW|bpd|MMcf|bbl|TPD|tph|MWh)\b",          # 800 MW, 300 MW
    r"\b\d+\+?\s?(?:platforms|stations|wells|plants|facilities|sites)\b",  # 4 platforms
]


def _extract_quantifiers(span, candidate_text, window=80):
    """
    Pull percentage / currency / engineering-unit figures near the span.

    Restricted to the *same sentence* as the span (sentence-scoped) and a
    tight ±`window` char radius so a quantifier for skill A doesn't leak
    onto unrelated skill B. Substring dedupe removes nested matches
    (e.g., '150,000 bpd' subsumes '000 bpd').
    """
    if not span or not candidate_text:
        return []
    low_text = candidate_text.lower()
    low_span = span.lower()
    pos = low_text.find(low_span)
    if pos < 0:
        return []
    # Constrain to the same sentence as the span.
    sent_start = pos
    while sent_start > 0 and candidate_text[sent_start - 1] not in ".!?\n":
        sent_start -= 1
    sent_end = pos + len(span)
    while sent_end < len(candidate_text) and candidate_text[sent_end] not in ".!?\n":
        sent_end += 1
    # Also clamp by window radius
    start = max(sent_start, pos - window)
    end = min(sent_end, pos + len(span) + window)
    context = candidate_text[start:end]

    raw_hits = []
    for pat in _QUANT_PATTERNS:
        for m in re.findall(pat, context, flags=re.IGNORECASE):
            raw_hits.append(m.strip())

    # Substring dedupe: if hit A contains hit B, drop B.
    raw_hits.sort(key=len, reverse=True)
    kept = []
    for h in raw_hits:
        if any(h.lower() in k.lower() and h.lower() != k.lower() for k in kept):
            continue
        # Also drop exact-ci duplicates
        if any(h.lower() == k.lower() for k in kept):
            continue
        kept.append(h)
    return kept


def _extract_all_quantifiers(text):
    """
    Extract every quantified figure across the whole resume — captures
    resume-level impact signals (e.g., '$2.1B LNG project', '45% incident
    reduction', '300 MW solar farm') that might not be in the same sentence
    as an extracted skill but still matter to a hiring manager.
    """
    if not text:
        return []
    raw_hits = []
    for pat in _QUANT_PATTERNS:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            raw_hits.append(m.strip())
    raw_hits.sort(key=len, reverse=True)
    kept = []
    for h in raw_hits:
        if any(h.lower() in k.lower() and h.lower() != k.lower() for k in kept):
            continue
        if any(h.lower() == k.lower() for k in kept):
            continue
        kept.append(h)
    return kept


def _extract_evidence_profile(closest_span, candidate_text):
    """
    Build a full evidence profile for a matched/weak item:
      - sentence: the resume sentence that supports the match
      - proficiency: high / medium / low (from modifier words before the span)
      - mention_count: how many times the span appears in the resume
      - quantifiers: % / $ / engineering-unit figures found near the span
    """
    if not closest_span or not candidate_text:
        return None
    sentence = _find_supporting_sentence(closest_span, candidate_text)
    proficiency = _detect_proficiency(closest_span, candidate_text)
    mention_count = candidate_text.lower().count(closest_span.lower())
    quantifiers = _extract_quantifiers(closest_span, candidate_text)
    profile = {}
    if sentence:
        profile["sentence"] = sentence
    if proficiency:
        profile["proficiency"] = proficiency
    if mention_count > 0:
        profile["mention_count"] = mention_count
    if quantifiers:
        profile["quantifiers"] = quantifiers
    return profile or None


# -----------------------------------------------------------------------------
# Energy-domain skill synonyms — precomputed equivalence groups. SBERT
# sometimes scores these below the STRONG threshold because the surface
# forms differ significantly ("HAZOP" vs "process hazard analysis" scores
# ~0.40). When both sides of a potential pair resolve to the same synonym
# group, we force a strong-tier similarity in the Hungarian cost matrix.
# -----------------------------------------------------------------------------

_SKILL_SYNONYMS = {
    "hazop": [
        "hazop", "hazop facilitation", "hazop study", "hazop review",
        "process hazard analysis", "pha", "hazard and operability",
        "hazard operability", "process hazard assessment",
    ],
    "pid_review": [
        "p&id", "p&id review", "p&id markup", "pid review",
        "piping and instrumentation diagram", "piping & instrumentation diagram",
        "piping instrumentation diagram",
    ],
    "scada": [
        "scada", "scada integration", "scada for pipelines",
        "supervisory control", "supervisory control and data acquisition",
    ],
    "plc": [
        "plc", "plc programming", "programmable logic controller",
        "ladder logic",
    ],
    "dcs": [
        "dcs", "dcs programming", "distributed control system",
        "distributed control systems",
    ],
    "ndt": [
        "ndt", "ndt testing", "ndt inspection",
        "non-destructive testing", "nondestructive testing",
    ],
    "process_safety_mgmt": [
        "process safety management", "psm",
        "process safety", "process safety programs",
    ],
    "moc": [
        "moc", "moc management",
        "management of change", "management-of-change",
    ],
    "sil": [
        "sil", "sil verification", "sil classification",
        "safety integrity level",
    ],
    "cathodic_protection": [
        "cathodic protection", "cp system", "impressed current cathodic",
    ],
    "rcm": [
        "rcm", "reliability centered maintenance",
        "reliability-centered maintenance",
    ],
    "feed_studies": [
        "feed", "feed studies", "front end engineering design",
        "front-end engineering design",
    ],
    "rca": [
        "rca", "root cause analysis", "root cause failure analysis",
        "root-cause analysis",
    ],
    "pe_license": [
        "pe", "pe license", "pe licensed", "professional engineer",
        "professional engineer license", "professional engineering license",
    ],
    "pmp_cert": [
        "pmp", "pmp certified", "pmp certification",
        "project management professional",
    ],
    "nebosh": [
        "nebosh", "nebosh certified", "nebosh diploma",
        "nebosh igc", "nebosh general certificate",
    ],
    "csp_cert": [
        "csp", "csp certified", "csp certification",
        "certified safety professional",
    ],
    "api_570": [
        "api 570", "api 570 certified", "api 570 certification",
        "api-570",
    ],
    "api_510": ["api 510", "api 510 certified", "api-510"],
    "api_1169": ["api 1169", "api 1169 certified", "api-1169"],
    "api_653": ["api 653", "api 653 certified", "api-653"],
    "api_571": ["api 571", "api 571 certified", "api-571"],
    "nace_cip": [
        "nace cip", "nace cip level", "nace certified",
        "ampp cip", "coating inspector",
    ],
    "gwo_safety": [
        "gwo", "gwo basic safety", "global wind organisation",
    ],
    "ohsa_30": ["osha 30", "osha 30-hour"],
    "osha_10": ["osha 10", "osha 10-hour"],
}

# Flatten to {alias_lower: group_id} for O(1) lookup
_SYNONYM_LOOKUP = {}
for group_id, aliases in _SKILL_SYNONYMS.items():
    for alias in aliases:
        _SYNONYM_LOOKUP[alias.lower().strip()] = group_id


def _resolve_synonym(span):
    """Return the synonym group ID if span matches any known alias, else None."""
    if not span:
        return None
    low = span.lower().strip()
    # Exact match
    if low in _SYNONYM_LOOKUP:
        return _SYNONYM_LOOKUP[low]
    # Substring check — match the longest alias that is a substring of span
    best = None
    best_len = 0
    for alias, group_id in _SYNONYM_LOOKUP.items():
        if len(alias) < 4:
            continue
        if alias in low and len(alias) > best_len:
            best = group_id
            best_len = len(alias)
    return best


def _explain_match(assignment):
    """
    Generate a compact, auditable one-line explanation of why this
    assignment was classified the way it was. Used by hiring managers
    (or the LLM agent) to trust-but-verify the model's reasoning.
    """
    parts = []
    raw = assignment.get("sbert_similarity", assignment.get("similarity", 0.0))
    parts.append(f"SBERT sim {raw:.2f}")
    if assignment.get("cluster_boost"):
        parts.append(f"+cluster boost (both in {assignment.get('cluster', '?')})")
    if assignment.get("synonym_group"):
        parts.append(f"+synonym group '{assignment['synonym_group']}'")
    if assignment.get("canonical"):
        parts.append(f"+taxonomy anchor '{assignment['canonical']}'")
    src = assignment.get("match_source")
    if src and src not in (None, "same_type"):
        parts.append(f"+cross-type via candidate's {src}")
    evidence = assignment.get("candidate_evidence", {}) or {}
    if evidence.get("proficiency"):
        parts.append(f"+proficiency: {evidence['proficiency']}")
    if evidence.get("mention_count", 0) >= 2:
        parts.append(f"+{evidence['mention_count']}× mentions in resume")
    if evidence.get("quantifiers"):
        parts.append(f"+quantified impact: {', '.join(evidence['quantifiers'][:2])}")
    return "; ".join(parts)


def _optimal_match(required_items, primary_pool, fallback_pool=None,
                   fallback_label=None, match_strong=0.75):
    """
    Hungarian-algorithm 1-to-1 optimal matching between required items and
    the primary pool, preventing a single candidate skill from being claimed
    by multiple requirements. Cross-type fallback (greedy) is applied only
    when the primary same-type assignment is below the STRONG threshold.

    Returns a list of dicts, one per requirement:
      {required, closest, similarity, match_source}
    where match_source ∈ {"same_type", fallback_label, None}.
    """
    n_req = len(required_items)
    if n_req == 0:
        return []

    # Pre-initialize assignment slots (None = unmatched)
    assignments = [
        {"required": r, "closest": None, "similarity": 0.0, "match_source": None}
        for r in required_items
    ]

    req_emb = _sbert_matcher.encode(required_items)

    if primary_pool:
        primary_emb = _sbert_matcher.encode(primary_pool)
        sim_matrix = cosine_similarity(req_emb, primary_emb)

        # Synonym pre-boost: for each (req, candidate) pair that resolves
        # to the same synonym group (e.g., 'HAZOP' ↔ 'process hazard
        # analysis'), lift their similarity to at least 0.92 so Hungarian
        # prefers that assignment. Record which group matched.
        req_groups = [_resolve_synonym(r) for r in required_items]
        pool_groups = [_resolve_synonym(p) for p in primary_pool]
        synonym_pairs = {}  # {req_idx: (group_id, pool_idx)}
        for i, rg in enumerate(req_groups):
            if not rg:
                continue
            for j, pg in enumerate(pool_groups):
                if pg == rg:
                    if sim_matrix[i][j] < 0.92:
                        sim_matrix[i][j] = 0.92
                    synonym_pairs.setdefault(i, (rg, j))

        # linear_sum_assignment minimizes cost → negate similarity to maximize.
        # It handles non-square matrices: assigns min(n_req, n_pool) pairs.
        row_idx, col_idx = linear_sum_assignment(-sim_matrix)
        for r, c in zip(row_idx, col_idx):
            assignments[r]["closest"] = primary_pool[c]
            assignments[r]["similarity"] = float(sim_matrix[r][c])
            assignments[r]["match_source"] = "same_type"
            # Record synonym match if it drove this assignment
            syn_hit = synonym_pairs.get(r)
            if syn_hit and syn_hit[1] == c:
                assignments[r]["synonym_group"] = syn_hit[0]

    # Cross-type fallback: if the assigned sim < STRONG, try the fallback pool
    # for a potentially better match (e.g., SKILL req → candidate CERT).
    if fallback_pool:
        fallback_emb = _sbert_matcher.encode(fallback_pool)
        f_sims = cosine_similarity(req_emb, fallback_emb)
        for i in range(n_req):
            if assignments[i]["similarity"] < match_strong:
                f_j = int(np.argmax(f_sims[i]))
                f_sim = float(f_sims[i][f_j])
                if f_sim > assignments[i]["similarity"]:
                    assignments[i]["closest"] = fallback_pool[f_j]
                    assignments[i]["similarity"] = f_sim
                    assignments[i]["match_source"] = fallback_label

    # Round similarities for display
    for a in assignments:
        a["similarity"] = round(a["similarity"], 3)

    return assignments


def _classify_assignments(assignments, match_strong=0.75, match_weak=0.50):
    """Bucket Hungarian-matched assignments into matched / weak / missing."""
    matched, weak, missing = [], [], []
    for a in assignments:
        if a["similarity"] >= match_strong:
            matched.append(a)
        elif a["similarity"] >= match_weak:
            weak.append(a)
        else:
            missing.append(a)
    return {"matched": matched, "weak": weak, "missing": missing}


def _compute_fit_score(gap_by_type, years_gap, seniority):
    """
    Principled 0-100 hiring-fit score with four weighted components:
      60%  critical-weighted coverage (CRITICAL reqs count 3x, PREFERRED 1.5x,
           STANDARD 1x, NICE_TO_HAVE 0.5x)
      15%  unweighted overall coverage
      15%  years-of-experience delta score
      10%  seniority alignment score

    Returns: dict with composite_fit_score, per-component scores, and a
    recommendation_tier string.
    """
    crit_weights = {"CRITICAL": 3.0, "PREFERRED": 1.5,
                    "STANDARD": 1.0, "NICE_TO_HAVE": 0.5}
    state_scores = {"matched": 1.0, "weak": 0.5, "missing": 0.0}

    weighted_num, weighted_den = 0.0, 0.0
    n_m, n_w, n_mi = 0, 0, 0
    for etype in ["SKILL", "CERT", "DEGREE"]:
        for state, s in state_scores.items():
            for m in gap_by_type[etype][state]:
                w = crit_weights.get(m.get("criticality", "STANDARD"), 1.0)
                weighted_num += w * s
                weighted_den += w
                if state == "matched": n_m += 1
                elif state == "weak":  n_w += 1
                else:                  n_mi += 1
    critical_weighted = (weighted_num / weighted_den * 100) if weighted_den else 100.0

    total = n_m + n_w + n_mi
    overall = ((n_m + 0.5 * n_w) / total * 100) if total else 100.0

    # Years component: 80 at exact match, linear up for surplus, down for shortfall.
    if years_gap.get("meets_requirement") is None:
        years_score = 70.0
    elif years_gap["meets_requirement"]:
        d = years_gap.get("delta_years", 0) or 0
        years_score = min(100.0, 80.0 + d * 2.0)
    else:
        d = years_gap.get("delta_years", 0) or 0  # negative
        years_score = max(0.0, 80.0 + d * 15.0)

    # Seniority component: 100 if aligned, drop 25 per level of mismatch.
    d = abs(seniority.get("delta", 0))
    if d <= 1:
        sen_score = 100.0
    else:
        sen_score = max(0.0, 100.0 - (d - 1) * 25.0)

    composite = (
        0.60 * critical_weighted
        + 0.15 * overall
        + 0.15 * years_score
        + 0.10 * sen_score
    )
    return {
        "composite_fit_score": round(composite, 1),
        "components": {
            "critical_weighted_coverage": round(critical_weighted, 1),
            "overall_coverage": round(overall, 1),
            "years_score": round(years_score, 1),
            "seniority_score": round(sen_score, 1),
        },
        "recommendation_tier": _fit_to_tier(composite),
    }


def _fit_to_tier(score):
    if score >= 85: return "STRONG_HIRE"
    if score >= 70: return "HIRE"
    if score >= 55: return "INTERVIEW"
    if score >= 40: return "CONDITIONAL"
    return "DO_NOT_ADVANCE"


# -----------------------------------------------------------------------------
# Skill-cluster relatedness — captures domain adjacency that raw SBERT
# cosine similarity can underweight. If a job requirement and the matched
# candidate skill live in the same cluster (e.g., both "control_systems"),
# we bump the effective similarity by a small boost. This is what lifts
# "SCADA integration" ↔ "DCS programming" matches that SBERT scores ~0.55
# into the 'matched' tier where they domain-functionally belong.
# -----------------------------------------------------------------------------

_SKILL_CLUSTERS = {
    "control_systems": [
        "scada", "plc", "dcs", "hmi", "instrumentation",
        "cybersecurity for ot", "control loop", "fieldbus",
        "control systems", "automation",
    ],
    "process_safety": [
        "hazop", "pha", "process hazard", "sil verification", "moc",
        "risk assessment", "process safety", "safety culture",
        "sis", "functional safety",
    ],
    "refining_chemistry": [
        "p&id", "turnaround", "distillation", "refining", "refinery",
        "hysys", "aspen plus", "aspen", "amine", "sulfur recovery",
        "fractionation", "dehydration", "process simulation",
    ],
    "power_systems": [
        "relay protection", "arc flash", "load flow", "power factor",
        "harmonic", "etap", "skm", "substation", "transformer",
        "grounding", "switchgear", "cable sizing", "protection",
    ],
    "offshore_subsea": [
        "subsea", "riser", "fpso", "dynamic positioning", "flow assurance",
        "mooring", "deepwater", "well intervention", "marine operations",
        "jack-up", "offshore", "pipesim", "olga",
    ],
    "renewables": [
        "solar inverter", "bess", "wind turbine", "wind resource",
        "pvsyst", "windpro", "wasp", "solar farm", "battery storage",
        "grid interconnection", "wind farm", "solar", "pv installation",
    ],
    "pipeline_integrity": [
        "pipeline integrity", "cathodic protection", "inline inspection",
        "in-line inspection", "leak detection", "pig launcher", "corrosion",
    ],
    "reliability_maintenance": [
        "reliability centered maintenance", "rcm", "root cause",
        "vibration analysis", "bearing failure", "ndt", "welding inspection",
        "predictive maintenance", "rotating equipment", "alignment",
        "maintenance planning",
    ],
    "drilling_reservoir": [
        "drilling", "completion", "reservoir simulation", "waterflood",
        "artificial lift", "wellhead", "reservoir engineering",
        "enhanced oil recovery",
    ],
    "hse_environmental": [
        "nebosh", "osha", "csp", "hse", "environmental impact",
        "wastewater", "air quality", "incident investigation",
        "hazardous waste", "permit to work",
    ],
    "project_management": [
        "pmp", "earned value", "primavera", "microsoft project",
        "vendor management", "budget forecasting", "stakeholder",
        "project management", "feed", "feed studies", "capital project",
    ],
    "nuclear": [
        "reactor", "radiation protection", "nuclear", "relap",
        "boiling water reactor", "emergency core cooling",
    ],
    "digital_data": [
        "machine learning", "digital twin", "iot", "cloud computing",
        "predictive modeling", "data visualization", "tableau", "power bi",
        "statistical process control",
    ],
}


def _skill_cluster(skill):
    """Return the domain cluster for a skill/cert/requirement, or None."""
    if not skill:
        return None
    low = skill.lower()
    for cluster, kws in _SKILL_CLUSTERS.items():
        for kw in kws:
            if kw in low:
                return cluster
    return None


# -----------------------------------------------------------------------------
# Interview-question templates — what to actually ask the candidate for each
# top gap. Short, specific, probes for depth vs keyword familiarity.
# -----------------------------------------------------------------------------

_INTERVIEW_TEMPLATES = {
    "hazop": "Walk me through a HAZOP you facilitated — top 3 findings and how you drove closure.",
    "api 570": "Describe your API 570 inspection workflow for a piping circuit with known thinning history.",
    "api 510": "How do you plan an API 510 pressure-vessel inspection — internal vs on-stream?",
    "api 1169": "Describe your role in a pipeline construction project applying API 1169.",
    "api 571": "Tell me about a damage-mechanism review (API 571) that changed an inspection plan.",
    "api 653": "Walk me through an API 653 tank inspection — settlement, corrosion, out-of-roundness.",
    "p&id": "Describe a P&ID review where you caught a significant design issue before construction.",
    "scada": "How would you design SCADA cybersecurity for 50+ remote pipeline stations?",
    "plc programming": "Describe a PLC migration you led — platforms, cutover strategy, what broke.",
    "dcs": "Walk me through a DCS upgrade you managed. What was the cutover plan and downtime?",
    "subsea": "Tell me about a subsea tieback design challenge — how you solved flow assurance.",
    "riser": "Describe a riser analysis — governing load cases and fatigue considerations.",
    "flow assurance": "Walk me through a flow assurance study for a deepwater tieback — tools, deliverables.",
    "wind turbine": "Describe a wind commissioning project — biggest risk items and mitigations.",
    "solar": "How do you approach solar inverter commissioning for a utility-scale facility?",
    "relay protection": "Walk me through a relay coordination study — settings and selectivity.",
    "arc flash": "Describe an arc-flash hazard analysis — how you reduced incident energy.",
    "reliability centered maintenance": "How have you built an RCM program — biggest wins?",
    "vibration analysis": "Describe a bearing failure you diagnosed via vibration analysis.",
    "pipeline integrity": "Walk me through your pipeline integrity plan — intervals and risk model.",
    "cathodic protection": "Describe a CP system you designed for a buried pipeline.",
    "welding inspection": "How do you plan and execute weld QA on a structural project?",
    "pe license": "Tell me about your PE licensure — hardest project you stamped.",
    "pmp": "Describe a project you led end-to-end — scope, cost, schedule, risk.",
    "nebosh": "Describe the safety management structure you've implemented on a major asset.",
    "risk assessment": "Walk me through a quantitative risk assessment — methodology and deliverables.",
    "turnaround": "Describe a major turnaround — what went well, what went wrong, what you'd change.",
    "hysys": "Describe a process simulation you built in HYSYS — key design decision it drove.",
    "etap": "Walk me through a load-flow / short-circuit study in ETAP.",
    "feed": "Describe a FEED study you led — scope, estimate class, deliverables.",
    "lng": "Describe your LNG experience — liquefaction, storage, or shipping?",
    "moc": "Walk me through a management-of-change you shepherded through the system.",
    "sil": "Describe a SIL verification — how you determined the target SIL.",
    "drilling": "Describe a drilling challenge — pressure, directional, or cost optimization.",
    "commissioning": "Walk me through a commissioning plan you led — sequence, punch list, handover.",
    "nuclear": "Describe your nuclear experience — reactor type, safety systems you worked on.",
    "reservoir simulation": "Describe a reservoir simulation study — objective, model, outcomes.",
    "corrosion": "Walk me through a corrosion mitigation strategy you designed.",
}


def _interview_question(required, etype):
    """Return a specific technical interview question for this gap."""
    if not required:
        return None
    low = required.lower()
    # Longest-keyword-first so 'plc programming' beats 'plc'
    for kw in sorted(_INTERVIEW_TEMPLATES.keys(), key=lambda k: -len(k)):
        if kw in low:
            return _INTERVIEW_TEMPLATES[kw]
    if etype == "CERT":
        return f"How did you prepare for the {required} credential and apply it in practice?"
    if etype == "SKILL":
        return f"Walk me through a project where {required} was critical to success."
    if etype == "DEGREE":
        return f"How has your {required} background shaped your engineering approach?"
    return f"Tell me about your experience with {required}."


# -----------------------------------------------------------------------------
# Taxonomy canonicalization — anchors each extracted entity to a known
# node in the v2 energy taxonomy (624 skills / 51 certs / 121 degrees /
# 285 employers). Used only for annotation + confidence; matching still
# runs on surface forms so taxonomy coverage gaps don't suppress matches.
# -----------------------------------------------------------------------------

_TAXO_SKILLS: dict = {}
_TAXO_CERTS: dict = {}
_TAXO_DEGREES: dict = {}
_TAXO_EMPLOYERS: dict = {}

try:
    from data.taxonomy.energy_taxonomy_v2 import (
        SKILLS as _V2_SKILLS,
        CERTIFICATIONS as _V2_CERTS,
        DEGREES as _V2_DEGREES,
        EMPLOYERS as _V2_EMPLOYERS,
    )
    # Build {normalized_lowercase_form: canonical_surface_form} lookups
    def _taxo_norm(s):
        return re.sub(r'[^\w\s&]', '', s.lower()).strip()
    _TAXO_SKILLS = {_taxo_norm(s): s for s in _V2_SKILLS}
    _TAXO_CERTS = {_taxo_norm(s): s for s in _V2_CERTS}
    _TAXO_DEGREES = {_taxo_norm(s): s for s in _V2_DEGREES}
    _TAXO_EMPLOYERS = {_taxo_norm(s): s for s in _V2_EMPLOYERS}
except Exception:
    # Graceful fallback if taxonomy hasn't been generated yet.
    pass


def _canonicalize(span, etype):
    """
    Return the v2-taxonomy canonical form for `span`, or None.

    Strategy:
      1. Normalize span (lowercase, strip punctuation, drop
         'certified'/'license'/'certification' suffixes).
      2. Exact lookup in the per-type taxonomy dict.
      3. Substring fallback: span ⊆ canonical or canonical ⊆ span.
    """
    pool = {
        "SKILL": _TAXO_SKILLS,
        "CERT": _TAXO_CERTS,
        "DEGREE": _TAXO_DEGREES,
        "EMPLOYER": _TAXO_EMPLOYERS,
    }.get(etype, {})
    if not pool or not span:
        return None
    norm = span.lower().strip()
    for suf in [" certified", " certification", " license", " certificate"]:
        if norm.endswith(suf):
            norm = norm[:-len(suf)]
    norm = re.sub(r'[^\w\s&]', '', norm).strip()
    if not norm:
        return None
    # Exact match
    if norm in pool:
        return pool[norm]
    # Substring fallback — only for reasonably long spans to avoid false hits
    if len(norm) >= 4:
        for key, canonical in pool.items():
            if not key or len(key) < 4:
                continue
            if norm == key or norm in key or key in norm:
                return canonical
    return None


# -----------------------------------------------------------------------------
# Employer pedigree — tier-1 energy employers the candidate has worked at.
# -----------------------------------------------------------------------------

_TIER_1_ENERGY_EMPLOYERS = {
    # Super-majors + NOCs
    "chevron", "shell", "exxon", "exxonmobil", "bp", "totalenergies", "total",
    "equinor", "saudi aramco", "aramco", "adnoc", "petronas", "petrobras",
    "conocophillips", "eni", "repsol",
    # Oilfield services + EPCs
    "baker hughes", "halliburton", "schlumberger", "slb",
    "technipfmc", "saipem", "subsea7", "mcdermott", "nov",
    "bechtel", "fluor", "kbr", "jacobs", "wood plc", "wood",
    # Industrial automation + power OEMs
    "ge vernova", "siemens energy", "siemens", "abb", "rockwell", "emerson",
    "honeywell uop", "honeywell", "yokogawa", "schneider electric",
    # Utilities / gentailers
    "nextera", "duke energy", "exelon", "southern company", "sempra",
    "tva", "bpa", "dominion", "aes",
    # Renewables + storage
    "vestas", "\u00f8rsted", "orsted", "first solar", "enphase", "sunpower",
    # Midstream
    "enterprise products", "williams companies", "williams", "kinder morgan",
    "enbridge", "energy transfer",
}


def _employer_pedigree(candidate_employers):
    """Flag tier-1 energy employers in the candidate's history."""
    hits = []
    for emp in candidate_employers or []:
        low = emp.lower()
        for t1 in _TIER_1_ENERGY_EMPLOYERS:
            if t1 in low:
                hits.append({"employer": emp, "tier": 1, "matched_canonical": t1})
                break
    return {
        "tier_1_count": len(hits),
        "total_employers_extracted": len(candidate_employers or []),
        "tier_1_employers": hits,
    }


# -----------------------------------------------------------------------------
# Confidence signal — how trustworthy is this analysis?
# -----------------------------------------------------------------------------

def _compute_confidence(cand_ents, job_ents, gap_by_type, taxo_hit_count):
    """
    Compute a confidence level for the analysis.

    Factors:
      - Candidate entity extraction count (more = more reliable)
      - JD requirement extraction count (≥3 for meaningful diff)
      - Strength distribution of matched items (≥0.85 = highly reliable)
      - Taxonomy hit count (canonicalized entities = gold-standard matches)
    """
    reasons = []
    score = 1.0

    cand_total = sum(len(v) for v in cand_ents.values())
    if cand_total < 4:
        score -= 0.30
        reasons.append(
            f"Only {cand_total} entities extracted from candidate — "
            "limited signal; some gaps may be extraction artifacts."
        )
    elif cand_total < 8:
        score -= 0.10
        reasons.append(
            f"Moderate candidate coverage ({cand_total} entities) — "
            "verify key skills in interview."
        )

    job_total = sum(len(job_ents.get(k, [])) for k in ["SKILL", "CERT", "DEGREE"])
    if job_total < 3:
        score -= 0.20
        reasons.append(
            f"Only {job_total} requirements extracted from JD — "
            "diff signal is sparse."
        )

    matched_count = sum(len(gap_by_type[e]["matched"]) for e in gap_by_type)
    strong_count = sum(
        1 for e in gap_by_type for m in gap_by_type[e]["matched"]
        if m["similarity"] >= 0.85
    )
    if matched_count > 0 and (strong_count / matched_count) < 0.6:
        score -= 0.10
        reasons.append(
            "Several matches are near the threshold — "
            "verify borderline items in interview."
        )

    if taxo_hit_count > 0:
        reasons.append(
            f"{taxo_hit_count} entities canonicalized to v2 taxonomy — "
            "high confidence on those specific matches."
        )

    score = max(0.0, min(1.0, score))
    if score >= 0.80:
        level = "high"
    elif score >= 0.50:
        level = "medium"
    else:
        level = "low"
    return {"level": level, "score": round(score, 2), "reasons": reasons}


# -----------------------------------------------------------------------------
# Counterfactual coaching paths — "if gap X closes, fit score goes from A to B".
# -----------------------------------------------------------------------------

def _counterfactual_paths(gap_by_type, years_gap, seniority, current_fit,
                          top_n=3):
    """
    For each top missing requirement, simulate closing it and recompute
    the fit score. Returns a ranked list of coaching paths by fit delta.
    """
    crit_order = {"CRITICAL": 0, "PREFERRED": 1, "STANDARD": 2, "NICE_TO_HAVE": 3}
    all_missing = []
    for etype in ["SKILL", "CERT", "DEGREE"]:
        for m in gap_by_type[etype]["missing"]:
            all_missing.append((etype, m))
    # Prioritize by criticality — closing a CRITICAL gap yields the biggest lift
    all_missing.sort(key=lambda x: crit_order.get(x[1].get("criticality", "STANDARD"), 2))

    paths = []
    for etype, m in all_missing[:top_n]:
        # Deep-ish copy of gap_by_type so we can mutate without side-effects
        sim_gaps = {
            k: {s: list(v) for s, v in b.items()}
            for k, b in gap_by_type.items()
        }
        sim_gaps[etype]["missing"] = [
            x for x in sim_gaps[etype]["missing"]
            if x["required"] != m["required"]
        ]
        simulated = dict(m)
        simulated["similarity"] = 1.0
        simulated["match_source"] = "simulated_closure"
        sim_gaps[etype]["matched"].append(simulated)

        new_fit = _compute_fit_score(sim_gaps, years_gap, seniority)
        delta = new_fit["composite_fit_score"] - current_fit
        paths.append({
            "if_candidate_closes_gap": m["required"],
            "gap_type": etype.lower(),
            "criticality": m.get("criticality", "STANDARD"),
            "action_to_close": m.get("action_suggestion", ""),
            "current_fit": current_fit,
            "projected_fit": new_fit["composite_fit_score"],
            "delta": round(delta, 1),
            "projected_tier": new_fit["recommendation_tier"],
        })

    # Sort by impact (highest delta first) — most-valuable coaching first
    paths.sort(key=lambda p: -p["delta"])
    return paths


def _suggest_action(required, etype):
    """Return a short actionable next-step suggestion for a missing/weak item."""
    low = required.lower()
    if etype == "CERT":
        if "pe license" in low or "pe " in low:
            return "PE license path: FE exam + 4yr experience + PE exam (~12mo)."
        if "api 57" in low or "api 51" in low or "api 1169" in low or "api " in low:
            return "API certification: ~3mo training + exam; register via API Events."
        if "nebosh" in low:
            return "NEBOSH IGC/Diploma: 2-8wk coursework + exam via accredited provider."
        if "csp" in low:
            return "CSP: requires BCSP eligibility + exam; ~6mo prep."
        if "osha 30" in low or "osha 10" in low:
            return "OSHA 10/30 online; 1-2 days, ~$100-$200."
        if "pmp" in low:
            return "PMP: 36mo PM experience + 35 contact hours + exam; ~6mo prep."
        if "nace" in low or "cip" in low:
            return "NACE/AMPP CIP: Levels 1-3; ~2wk coursework + exam per level."
        if "gwo" in low:
            return "GWO Basic Safety: 5-day hands-on training at approved provider."
        return "Identify cert exam path; typical cycle 3-6mo including study time."
    if etype == "SKILL":
        if "hazop" in low or "pha" in low or "process hazard" in low:
            return "Shadow 2-3 live HAZOPs, then attend 5-day facilitator course."
        if "scada" in low or "plc" in low or "dcs" in low or "instrumentation" in low:
            return "Vendor training (Rockwell / Siemens / Emerson) + 6mo on-the-job."
        if any(k in low for k in ["subsea", "riser", "flow assurance", "fpso", "offshore"]):
            return "Specialized offshore domain — typically 2+ yrs adjacent hands-on."
        if any(k in low for k in ["hysys", "aspen", "pipesim", "olga", "etap"]):
            return "Simulation tool: vendor self-study + ~3mo of applied projects."
        if "python" in low or "sql" in low or "matlab" in low:
            return "Technical interview sufficient; tools are broadly transferable."
        if any(k in low for k in ["safety", "risk assessment", "risk management"]):
            return "Assess via case-study interview; often overlaps with HSE certs."
        return "Map to adjacent skill candidate has; validate in technical interview."
    if etype == "DEGREE":
        return "Degree gaps are hard to close; consider equivalent-experience waiver."
    if etype == "TOOL":
        if any(k in low for k in ["hysys", "aspen", "pipesim", "olga", "etap",
                                   "ansys", "ansys fluent", "abaqus"]):
            return "Simulation tool: vendor self-study + ~3mo of applied projects."
        if any(k in low for k in ["python", "r programming", "matlab", "simulink",
                                   "sql", "c++", "vba", "labview"]):
            return "Technical interview sufficient; tools are broadly transferable."
        if any(k in low for k in ["autocad", "smartplant", "navisworks", "revit",
                                   "solidworks", "microstation"]):
            return "CAD/drafting tool: vendor training + 6mo on-the-job ramp."
        if any(k in low for k in ["sap", "maximo", "oracle", "jd edwards", "primavera"]):
            return "ERP/CMMS platform: 3-6mo to productivity; not hiring-blocking."
        if any(k in low for k in ["scada", "hmi", "dcs", "plc", "deltav", "experion",
                                   "tia portal", "rslogix", "centum", "ignition",
                                   "wonderware", "factorytalk"]):
            return "Automation platform: vendor training + 6mo on-the-job."
        return "Vendor training + applied project work; typically 1-3mo to productivity."
    if etype == "INDUSTRY":
        return "Sector shift is coachable — target 6-12mo exposure via project rotation."
    if etype == "LOCATION":
        return "Geographic / basin exposure — relocation or rotational secondment."
    if etype == "PROJECT":
        return "Named megaproject exposure — valuable signal; not replaceable in-role."
    if etype == "SOFT_SKILL":
        return "Interpersonal development — 360 feedback + executive coaching."
    return "Assess fit during interview."


# -----------------------------------------------------------------------------
# v7 Tier-2 helpers — additive signals on top of the 100-point base
# composite_fit_score. See ENTITY_SCHEMA_V7.md §"Fit-score formula guard".
#
# Design guard: these functions return empty / neutral outputs when the
# underlying entity type is absent (e.g., the v6 NER model never emits
# TOOL/INDUSTRY/LOCATION/PROJECT/SOFT_SKILL, so the tier2_bonus collapses
# to 0 and the composite fit score is untouched — this is what preserves
# the Carlos Mendez joint_hire R9-interview-stable byte-identity.
# -----------------------------------------------------------------------------


def _compute_tool_coverage(gap_by_type):
    """Hungarian coverage for TOOL type.

    Returns coverage in [0, 1] + raw counts. When the JD has no TOOL
    requirements, coverage is None (signals "skip" to the bonus formula).
    """
    bucket = gap_by_type.get("TOOL")
    if bucket is None:
        return {"coverage": None, "n_required": 0, "status": "not_evaluated"}
    total = len(bucket["matched"]) + len(bucket["weak"]) + len(bucket["missing"])
    if total == 0:
        return {"coverage": None, "n_required": 0, "status": "no_jd_tools"}
    matched = len(bucket["matched"]) + 0.5 * len(bucket["weak"])
    return {
        "coverage": round(matched / total, 3),
        "n_matched": len(bucket["matched"]),
        "n_weak": len(bucket["weak"]),
        "n_missing": len(bucket["missing"]),
        "n_required": total,
    }


def _compute_industry_match(cand_industries, job_industries):
    """Binary sector-match with an SBERT fuzzy fallback.

    Returns match_score in [0, 1]:
      1.0  → at least one industry string appears verbatim on both sides
      p    → highest cross-pair SBERT cosine similarity if no exact overlap
      0.0  → complete mismatch
    """
    if not job_industries:
        return {"match_score": None, "status": "no_jd_industry"}
    cand_set = {i.lower().strip() for i in (cand_industries or [])}
    job_set = {j.lower().strip() for j in job_industries}
    overlap = cand_set & job_set
    if overlap:
        return {
            "match_score": 1.0,
            "status": "aligned",
            "overlapping": sorted(overlap),
            "candidate_industries": sorted(cand_set),
            "jd_industries": sorted(job_set),
        }
    if cand_set:
        job_emb = _sbert_matcher.encode(sorted(job_set))
        cand_emb = _sbert_matcher.encode(sorted(cand_set))
        sims = cosine_similarity(job_emb, cand_emb)
        max_sim = float(np.max(sims))
        if max_sim >= 0.70:
            return {
                "match_score": round(max_sim, 3),
                "status": "partial",
                "candidate_industries": sorted(cand_set),
                "jd_industries": sorted(job_set),
            }
    return {
        "match_score": 0.0,
        "status": "mismatch",
        "candidate_industries": sorted(cand_set),
        "jd_industries": sorted(job_set),
    }


def _compute_location_match(cand_locations, job_locations):
    """Binary basin/region overlap. No SBERT fallback — locations are
    typically proper-noun spans where fuzzy matching is noisy."""
    if not job_locations:
        return {"match_score": None, "status": "no_jd_location"}
    cand_set = {loc.lower().strip() for loc in (cand_locations or [])}
    job_set = {loc.lower().strip() for loc in job_locations}
    overlap = cand_set & job_set
    if overlap:
        return {
            "match_score": 1.0,
            "status": "aligned",
            "overlapping": sorted(overlap),
            "candidate_locations": sorted(cand_set),
            "jd_locations": sorted(job_set),
        }
    if not cand_set:
        return {
            "match_score": 0.0,
            "status": "candidate_no_location_mentioned",
            "jd_locations": sorted(job_set),
        }
    return {
        "match_score": 0.0,
        "status": "mismatch",
        "candidate_locations": sorted(cand_set),
        "jd_locations": sorted(job_set),
    }


def _compute_soft_skill_overlap(cand_soft, job_soft):
    """Simple set-overlap ratio for SOFT_SKILL. Tier 3 so we don't run the
    full Hungarian — the signal is interpretive, not gating."""
    if not job_soft:
        return {"match_score": None, "status": "no_jd_soft_skills"}
    cand_set = {s.lower().strip() for s in (cand_soft or [])}
    job_set = {s.lower().strip() for s in job_soft}
    overlap = cand_set & job_set
    if not job_set:
        return {"match_score": None, "status": "no_jd_soft_skills"}
    coverage = len(overlap) / len(job_set)
    return {
        "match_score": round(coverage, 3),
        "status": "aligned" if coverage > 0 else "no_overlap",
        "n_overlap": len(overlap),
        "n_jd_required": len(job_set),
        "overlapping": sorted(overlap),
    }


def _compute_project_signal(cand_projects):
    """Presence signal for PROJECT mentions. Tier 3, interpretive only."""
    projects = [p for p in (cand_projects or []) if p.strip()]
    return {
        "match_score": min(1.0, len(projects) / 2.0) if projects else 0.0,
        "n_projects": len(projects),
        "projects": projects[:5],
    }


def _compute_tier2_bonus(tool_coverage, industry_match, location_match,
                          soft_skill_overlap, project_signal):
    """Compose a +15-capped additive bonus from five Tier-2/3 signals.

    Component sub-caps (sum = 15):
      TOOL:        5.0  (Tier 1-adjacent, highest weight)
      INDUSTRY:    4.0
      SOFT_SKILL:  3.0
      LOCATION:    2.0
      PROJECT:     1.0

    A None match_score means "not applicable" (JD didn't specify) and
    contributes 0 points. Cap at 15 is an absolute ceiling.
    """
    def _pts(signal, max_pts):
        s = signal.get("match_score") if signal else None
        return (max_pts * float(s)) if s is not None else 0.0

    tool_pts = _pts(tool_coverage, 5.0)
    industry_pts = _pts(industry_match, 4.0)
    soft_pts = _pts(soft_skill_overlap, 3.0)
    location_pts = _pts(location_match, 2.0)
    project_pts = _pts(project_signal, 1.0)

    raw = tool_pts + industry_pts + soft_pts + location_pts + project_pts
    total = round(min(15.0, raw), 1)
    return {
        "tier2_bonus_total": total,
        "components": {
            "tool_pts": round(tool_pts, 2),
            "industry_pts": round(industry_pts, 2),
            "soft_skill_pts": round(soft_pts, 2),
            "location_pts": round(location_pts, 2),
            "project_pts": round(project_pts, 2),
        },
        "raw_sum_before_cap": round(raw, 2),
        "cap": 15.0,
    }


@tool("Skill Gap Analyzer")
def analyze_skill_gap(candidate_text: str, job_text: str) -> str:
    """
    Diff a candidate's NER-extracted entities against a job description's
    requirements and produce an enterprise-grade fit analysis.

    Pipeline:
      1. NER ensemble (DistilBERT + GLiNER) extracts SKILL / CERT / DEGREE /
         YEARS_EXP from both texts; JD-side spans are cleaned of phrasing
         noise and title leakage.
      2. Each job requirement is tagged CRITICAL / PREFERRED /
         NICE_TO_HAVE / STANDARD based on modifier words in its JD context.
      3. Hungarian (1-to-1 optimal) assignment of requirements to candidate
         entities via SBERT cosine similarity — prevents double-counting.
         Cross-type fallback (SKILL↔CERT) kicks in only when the same-type
         match is below the STRONG threshold.
      4. Seniority alignment detected from title-level signals on both sides.
      5. Composite 0-100 fit score with weighted components:
         60% critical-weighted coverage, 15% overall coverage, 15% years,
         10% seniority. Maps to STRONG_HIRE / HIRE / INTERVIEW /
         CONDITIONAL / DO_NOT_ADVANCE.
      6. Actionable next-step suggestion per missing/weak item
         (certification path, training duration, adjacent-skill hint).

    Input: candidate_text (resume), job_text (job description).
    Output: JSON with matched/weak/missing items (tagged by criticality),
    coverage percentages, years delta, seniority alignment, composite fit
    score, per-gap action suggestions, top gaps, and a summary the agent
    can paraphrase.
    """
    if _ner_engine is None or _sbert_matcher is None:
        return json.dumps({"error": "NER engine or SBERT matcher not initialized"})

    cand_ents = _ner_engine.extract_entities(candidate_text)
    job_ents = _ner_engine.extract_entities(job_text)

    # Clean JD-side noise: prefix/suffix phrasing, boilerplate spans,
    # job-title leaks. The candidate side is already well-cleaned by
    # EnsembleNEREngine's post-processing.
    job_title = _extract_job_title(job_text)
    for etype in ["SKILL", "CERT", "DEGREE"]:
        job_ents[etype] = _clean_job_spans(job_ents.get(etype, []), job_title)

    # Lower-cased JD text used for criticality window search.
    job_text_lower = job_text.lower()

    # Clean JD-side TOOL spans too (v7 output) — same noise filter applies.
    if "TOOL" in job_ents:
        job_ents["TOOL"] = _clean_job_spans(job_ents.get("TOOL", []), job_title)

    # Hungarian 1-to-1 optimal matching per entity type, with cross-type
    # fallback for SKILL↔CERT when the same-type match is weak.
    skill_assign = _optimal_match(
        job_ents.get("SKILL", []), cand_ents.get("SKILL", []),
        fallback_pool=cand_ents.get("CERT", []), fallback_label="candidate_cert",
    )
    cert_assign = _optimal_match(
        job_ents.get("CERT", []), cand_ents.get("CERT", []),
        fallback_pool=cand_ents.get("SKILL", []), fallback_label="candidate_skill",
    )
    degree_assign = _optimal_match(
        job_ents.get("DEGREE", []), cand_ents.get("DEGREE", []),
    )
    # v7 TOOL Hungarian matching — mirrors the SKILL loop. Cross-type
    # fallback pulls from candidate SKILL in case the candidate's NER-tagged
    # tools leaked into SKILL (common on v6 extractions where "AutoCAD" /
    # "Python" are still tagged SKILL). Empty under v6 model (no TOOL output).
    tool_assign = _optimal_match(
        job_ents.get("TOOL", []), cand_ents.get("TOOL", []),
        fallback_pool=cand_ents.get("SKILL", []), fallback_label="candidate_skill",
    )

    # Attach criticality, taxonomy canonical form, skill cluster, and
    # skill-cluster similarity boost. The boost captures domain adjacency
    # that raw SBERT may underweight — e.g. 'SCADA integration' vs 'DCS
    # programming' score ~0.55 on SBERT but are functionally equivalent
    # inside the control_systems cluster. Boost: +0.10, capped at 1.0.
    taxo_hit_count = 0
    for assignments, etype in [
        (skill_assign, "SKILL"),
        (cert_assign, "CERT"),
        (degree_assign, "DEGREE"),
        (tool_assign, "TOOL"),
    ]:
        for a in assignments:
            a["criticality"] = _detect_criticality(a["required"], job_text_lower)
            canon = _canonicalize(a["required"], etype)
            if canon:
                a["canonical"] = canon
                taxo_hit_count += 1
            # Skill-cluster boost: record cluster; if matched skill shares
            # the cluster, lift the similarity up to +0.10 (capped at 1.0).
            req_cluster = _skill_cluster(a["required"])
            if req_cluster:
                a["cluster"] = req_cluster
            closest_cluster = _skill_cluster(a["closest"])
            if req_cluster and req_cluster == closest_cluster and a["similarity"] < 0.9:
                raw = a["similarity"]
                a["sbert_similarity"] = raw
                a["cluster_boost"] = 0.10
                a["similarity"] = round(min(1.0, raw + 0.10), 3)
            # Candidate-side evidence profile — only for items where the
            # model actually paired the requirement with something from the
            # resume (i.e., matched or weak). Missing items have no candidate
            # evidence to surface.
            if a["closest"] and a["similarity"] >= 0.50:
                profile = _extract_evidence_profile(a["closest"], candidate_text)
                if profile:
                    a["candidate_evidence"] = profile
            # Suggest action + interview question for non-strong matches.
            if a["similarity"] < 0.75:
                a["action_suggestion"] = _suggest_action(a["required"], etype)
                a["interview_question"] = _interview_question(a["required"], etype)
            # Audit one-liner — after all other fields are in place so the
            # explanation can cite them.
            a["match_explanation"] = _explain_match(a)

    gap_by_type = {
        "SKILL": _classify_assignments(skill_assign),
        "CERT": _classify_assignments(cert_assign),
        "DEGREE": _classify_assignments(degree_assign),
        # v7: TOOL joins gap_by_type but is NOT included in _compute_fit_score
        # (that loop still hardcodes SKILL/CERT/DEGREE), preserving the
        # Carlos Mendez static-mode byte-identity. TOOL feeds tier2_bonus.
        "TOOL": _classify_assignments(tool_assign),
    }

    years_gap = _compare_years(
        job_ents.get("YEARS_EXP", []),
        cand_ents.get("YEARS_EXP", []),
    )
    seniority = _seniority_alignment(job_text, candidate_text)

    # Unweighted coverage (for the components block and legacy consumers).
    coverage = {}
    all_req = 0
    all_matched = 0.0
    for etype, bucket in gap_by_type.items():
        total = len(bucket["matched"]) + len(bucket["weak"]) + len(bucket["missing"])
        matched_ct = len(bucket["matched"]) + 0.5 * len(bucket["weak"])
        pct = round((matched_ct / total * 100) if total else 100.0, 1)
        coverage[f"{etype.lower()}_coverage_pct"] = pct
        all_req += total
        all_matched += matched_ct
    coverage["overall_coverage_pct"] = round(
        (all_matched / all_req * 100) if all_req else 100.0, 1
    )

    # Composite 0-100 fit score with weighted components. Unchanged from
    # v6 — iterates SKILL/CERT/DEGREE only. TOOL + the 5 new v7 types ride
    # on the Tier-2 additive bonus (see below), preserving the Carlos
    # Mendez joint_hire R9-interview-stable byte-identity under v6 NER.
    fit = _compute_fit_score(gap_by_type, years_gap, seniority)

    # v7 Tier-2 / Tier-3 signals — additive, capped at +15. Under a v6 NER
    # model, every signal's match_score is None (no TOOL/INDUSTRY/LOCATION/
    # etc. in the ents dicts), so tier2_bonus_total collapses to 0.
    tool_coverage_signal = _compute_tool_coverage(gap_by_type)
    industry_match = _compute_industry_match(
        cand_ents.get("INDUSTRY", []),
        job_ents.get("INDUSTRY", []),
    )
    location_match = _compute_location_match(
        cand_ents.get("LOCATION", []),
        job_ents.get("LOCATION", []),
    )
    soft_skill_overlap = _compute_soft_skill_overlap(
        cand_ents.get("SOFT_SKILL", []),
        job_ents.get("SOFT_SKILL", []),
    )
    project_signal = _compute_project_signal(
        cand_ents.get("PROJECT", []),
    )
    tier2_bonus = _compute_tier2_bonus(
        tool_coverage_signal, industry_match, location_match,
        soft_skill_overlap, project_signal,
    )

    # Employer pedigree — tier-1 energy employers in candidate's history.
    pedigree = _employer_pedigree(cand_ents.get("EMPLOYER", []))

    # Resume-level impact signals — all %, $, and engineering-unit figures
    # found anywhere in the resume (not just near a specific skill). Lets
    # the agent cite candidate impact even when the quantifier isn't in
    # the same sentence as a matched skill.
    resume_impact_signals = _extract_all_quantifiers(candidate_text)

    # Confidence signal — extraction coverage + match strength + taxonomy hits.
    confidence = _compute_confidence(
        cand_ents, job_ents, gap_by_type, taxo_hit_count
    )

    # Counterfactual coaching paths — top-3 gap closures ranked by fit lift.
    coaching_paths = _counterfactual_paths(
        gap_by_type, years_gap, seniority, fit["composite_fit_score"],
    )

    # Top gaps — CRITICAL misses first, then PREFERRED, then the rest.
    crit_order = {"CRITICAL": 0, "PREFERRED": 1, "STANDARD": 2, "NICE_TO_HAVE": 3}
    all_missing = []
    for etype in ["SKILL", "CERT", "DEGREE"]:
        for m in gap_by_type[etype]["missing"]:
            all_missing.append((etype, m))
    all_missing.sort(key=lambda x: crit_order.get(x[1].get("criticality", "STANDARD"), 2))
    top_gaps = []
    for etype, m in all_missing[:5]:
        top_gaps.append({
            "requirement": m["required"],
            "type": etype.lower(),
            "criticality": m.get("criticality", "STANDARD"),
            "action": m.get("action_suggestion", ""),
        })

    # Natural-language summary the LLM can paraphrase directly.
    summary_parts = [
        f"Fit: {fit['composite_fit_score']}/100 → "
        f"{fit['recommendation_tier'].replace('_', ' ')}."
    ]
    if top_gaps:
        gap_strs = [f"{g['requirement']} [{g['criticality']}]"
                    for g in top_gaps[:3]]
        summary_parts.append(f"Top gaps: {', '.join(gap_strs)}.")
    else:
        summary_parts.append("No significant skill gaps identified.")
    if years_gap.get("meets_requirement") is True:
        summary_parts.append(
            f"Experience: {years_gap['candidate_has']} meets the "
            f"{years_gap['required']} requirement."
        )
    elif years_gap.get("meets_requirement") is False:
        summary_parts.append(
            f"Experience shortfall: candidate has {years_gap['candidate_has']}, "
            f"job requires {years_gap['required']}."
        )
    if seniority["alignment"] != "aligned":
        summary_parts.append(
            f"Seniority: {seniority['alignment'].replace('_', ' ')} "
            f"({seniority['candidate_level_label']} vs "
            f"{seniority['job_level_label']})."
        )
    if pedigree["tier_1_count"] > 0:
        names = ", ".join(h["employer"] for h in pedigree["tier_1_employers"][:3])
        summary_parts.append(
            f"Tier-1 energy pedigree: {pedigree['tier_1_count']} "
            f"({names})."
        )
    if resume_impact_signals:
        summary_parts.append(
            f"Quantified impact: {', '.join(resume_impact_signals[:3])}."
        )
    if coaching_paths:
        top_path = coaching_paths[0]
        if top_path["delta"] >= 3:
            summary_parts.append(
                f"Top coaching lift: closing '{top_path['if_candidate_closes_gap']}' "
                f"raises fit {top_path['current_fit']}→{top_path['projected_fit']} "
                f"(+{top_path['delta']})."
            )
    if confidence["level"] != "high":
        summary_parts.append(f"Confidence: {confidence['level']}.")

    result = {
        "fit_score": fit,
        "coverage": coverage,
        "years_experience_gap": years_gap,
        "seniority_alignment": seniority,
        "employer_pedigree": pedigree,
        "resume_impact_signals": resume_impact_signals,
        "confidence": confidence,
        "coaching_paths": coaching_paths,
        "gaps_by_type": gap_by_type,
        "top_gaps": top_gaps,
        "taxonomy_hits": taxo_hit_count,
        "summary": " ".join(summary_parts),
        # --- v7 additive signals (Tier-2 / Tier-3) ---
        # None of these fields feed back into composite_fit_score; they
        # are surfaced for memo rendering and downstream UI consumption.
        "tool_coverage": tool_coverage_signal,
        "industry_match": industry_match,
        "location_match": location_match,
        "soft_skill_overlap": soft_skill_overlap,
        "project_signal": project_signal,
        "tier2_bonus": tier2_bonus,
    }
    # Executive briefing — boardroom-ready 3-paragraph narrative that
    # synthesizes every signal above. Attached after the result is built
    # so the generator has access to all fields.
    result["executive_briefing"] = _generate_executive_briefing(result)
    return json.dumps(result, indent=2)


def _compare_years(job_years_list, candidate_years_list):
    """Extract leading integer from each YEARS_EXP span and compare."""
    def _first_num(strs):
        for s in strs:
            m = re.search(r'(\d+)', s)
            if m:
                return int(m.group(1))
        return None

    req_n = _first_num(job_years_list)
    cand_n = _first_num(candidate_years_list)

    if req_n is None and cand_n is None:
        return {"required": None, "candidate_has": None, "meets_requirement": None}
    if req_n is None:
        return {
            "required": None,
            "candidate_has": f"{cand_n} years",
            "meets_requirement": None,
        }
    if cand_n is None:
        return {
            "required": f"{req_n}+ years",
            "candidate_has": None,
            "meets_requirement": None,
        }
    return {
        "required": f"{req_n}+ years",
        "candidate_has": f"{cand_n} years",
        "meets_requirement": cand_n >= req_n,
        "delta_years": cand_n - req_n,
    }


@tool("O*NET Skill Lookup")
def lookup_onet_skills(skill_name: str) -> str:
    """
    Look up standardized skill mappings from the O*NET taxonomy.
    Maps energy-sector terminology to standardized occupational skills.
    Input: A skill name or term (e.g., "HAZOP", "P&ID review").
    Output: JSON with standardized skill info, related occupations, and importance.
    """
    # Simulated O*NET mapping for energy-sector skills
    onet_mappings = {
        "hazop": {
            "standardized_name": "Process Safety Analysis",
            "onet_code": "17-2041.00",
            "occupation": "Chemical Engineers",
            "importance": "Critical",
            "related_skills": ["Risk Assessment", "Process Control", "Safety Engineering"],
        },
        "p&id review": {
            "standardized_name": "Piping and Instrumentation Diagram Analysis",
            "onet_code": "17-2041.00",
            "occupation": "Chemical Engineers",
            "importance": "High",
            "related_skills": ["Engineering Drawing", "Process Design", "Instrumentation"],
        },
        "scada": {
            "standardized_name": "Supervisory Control and Data Acquisition",
            "onet_code": "17-2071.00",
            "occupation": "Electrical Engineers",
            "importance": "Critical",
            "related_skills": ["Control Systems", "PLC Programming", "Industrial Automation"],
        },
        "welding inspection": {
            "standardized_name": "Weld Quality Inspection",
            "onet_code": "17-3029.09",
            "occupation": "Engineering Technicians",
            "importance": "High",
            "related_skills": ["NDT", "Metallurgy", "Quality Control"],
        },
        "relay protection": {
            "standardized_name": "Protective Relay Engineering",
            "onet_code": "17-2071.00",
            "occupation": "Electrical Engineers",
            "importance": "High",
            "related_skills": ["Power Systems", "Fault Analysis", "Substation Design"],
        },
    }

    key = skill_name.lower().strip()
    if key in onet_mappings:
        return json.dumps(onet_mappings[key], indent=2)

    return json.dumps({
        "standardized_name": skill_name,
        "note": "No exact O*NET mapping found. Using raw skill name.",
        "suggestion": "Consider consulting O*NET OnLine for manual mapping.",
    }, indent=2)


def _generate_recommendation(score, tier):
    """Generate an actionable recommendation based on match score."""
    if score >= 0.80:
        return "Strong candidate. Proceed to interview immediately."
    elif score >= 0.65:
        return "Good fit. Schedule screening call to verify key qualifications."
    elif score >= 0.50:
        return "Partial match. Review specific skill gaps before proceeding."
    elif score >= 0.35:
        return "Weak match. Consider only if candidate pool is limited."
    else:
        return "Poor match. Do not advance for this role."


# =============================================================================
# Executive briefing — boardroom-ready 3-paragraph narrative from the
# full analyze_skill_gap output. Ties every signal into a single document
# a hiring manager can read in 30 seconds.
# =============================================================================

_VERDICT_OPENERS = {
    "STRONG_HIRE":
        "Strong match — fast-track to final-round interview.",
    "HIRE":
        "Solid candidate — advance to panel interview.",
    "INTERVIEW":
        "Viable candidate — merits a deeper look.",
    "CONDITIONAL":
        "Borderline candidate — advance only with targeted gap verification.",
    "DO_NOT_ADVANCE":
        "Not a fit for this role — declining is the right call.",
}


def _generate_executive_briefing(result, candidate_name="the candidate"):
    """
    Produce a 3-paragraph executive briefing from a full skill-gap result:
      1. Verdict + headline numbers (fit score, tier, years, seniority).
      2. Strengths: matched critical items with evidence.
      3. Concerns + coaching path: top gaps and what closing them would do.
    """
    fit = result.get("fit_score", {})
    score = fit.get("composite_fit_score", 0)
    tier = fit.get("recommendation_tier", "?")
    years = result.get("years_experience_gap", {}) or {}
    sen = result.get("seniority_alignment", {}) or {}
    pedigree = result.get("employer_pedigree", {}) or {}
    impact = result.get("resume_impact_signals", []) or []
    coaching = result.get("coaching_paths", []) or []
    gap_by_type = result.get("gaps_by_type", {}) or {}
    confidence = result.get("confidence", {}) or {}

    # ---- Paragraph 1 — verdict + context ----
    p1 = [_VERDICT_OPENERS.get(tier, "Assessment complete.")]
    p1.append(f"Composite fit score: {score}/100.")
    if years.get("meets_requirement") is True:
        p1.append(
            f"Experience ({years.get('candidate_has')}) meets the "
            f"{years.get('required')} threshold."
        )
    elif years.get("meets_requirement") is False:
        p1.append(
            f"Experience shortfall: {years.get('candidate_has')} vs "
            f"{years.get('required')} required."
        )
    if sen.get("alignment") == "aligned":
        p1.append(
            f"Seniority is aligned ({sen.get('candidate_level_label')})."
        )
    elif sen.get("alignment") == "candidate_under_leveled":
        p1.append(
            f"Seniority is below target "
            f"({sen.get('candidate_level_label')} vs "
            f"{sen.get('job_level_label')})."
        )
    elif sen.get("alignment") == "candidate_over_leveled":
        p1.append(
            f"Seniority is above target "
            f"({sen.get('candidate_level_label')} vs "
            f"{sen.get('job_level_label')}) — "
            f"confirm compensation and role-scope fit."
        )
    if pedigree.get("tier_1_count", 0) > 0:
        names = ", ".join(h["employer"]
                          for h in pedigree.get("tier_1_employers", [])[:3])
        p1.append(f"Tier-1 energy pedigree: {names}.")
    if impact:
        p1.append(f"Quantified impact in resume: {', '.join(impact[:3])}.")

    # ---- Paragraph 2 — strengths (matched CRITICAL items with evidence) ----
    p2_parts = []
    critical_matches = []
    for etype in ["SKILL", "CERT", "DEGREE"]:
        for m in gap_by_type.get(etype, {}).get("matched", []):
            if m.get("criticality") == "CRITICAL":
                critical_matches.append((etype, m))
    if critical_matches:
        p2_parts.append(
            f"Strengths: {candidate_name} has "
            f"{len(critical_matches)} critical requirement"
            f"{'s' if len(critical_matches) != 1 else ''} confirmed with evidence."
        )
        for etype, m in critical_matches[:3]:
            evidence = m.get("candidate_evidence", {}) or {}
            if evidence.get("sentence"):
                p2_parts.append(
                    f"  - {m.get('required')} "
                    f"(evidence: \"{evidence['sentence'][:120]}"
                    f"{'…' if len(evidence['sentence']) > 120 else ''}\")"
                )
            else:
                p2_parts.append(f"  - {m.get('required')}")
    else:
        # Fall back to any matched items
        any_matched = []
        for etype in ["SKILL", "CERT"]:
            for m in gap_by_type.get(etype, {}).get("matched", []):
                any_matched.append((etype, m))
        if any_matched:
            p2_parts.append(
                f"Strengths: {len(any_matched)} requirements matched, "
                f"but none tagged CRITICAL by the JD parser."
            )
        else:
            p2_parts.append(
                "Strengths: no direct requirement matches found — "
                "the candidate's background may be in an adjacent domain."
            )

    # ---- Paragraph 3 — concerns + coaching path ----
    p3_parts = []
    top_gaps = result.get("top_gaps", []) or []
    if top_gaps:
        crit_gaps = [g for g in top_gaps if g.get("criticality") == "CRITICAL"]
        label = f"{len(crit_gaps)} critical gap{'s' if len(crit_gaps) != 1 else ''}" \
                if crit_gaps else "no critical gaps (only weak spots)"
        p3_parts.append(f"Concerns: {label}.")
        for g in (crit_gaps or top_gaps)[:3]:
            action = g.get("action", "").rstrip(".")
            p3_parts.append(
                f"  - {g['requirement']} [{g['criticality']}]"
                + (f" — {action}." if action else ".")
            )
    else:
        p3_parts.append("Concerns: none significant.")
    if coaching:
        top = coaching[0]
        if top.get("delta", 0) >= 3:
            p3_parts.append(
                f"Biggest coaching lift: closing "
                f"'{top['if_candidate_closes_gap']}' raises fit "
                f"{top['current_fit']}→{top['projected_fit']} "
                f"(+{top['delta']}) → {top['projected_tier'].replace('_', ' ')}."
            )
    if confidence.get("level") and confidence["level"] != "high":
        p3_parts.append(
            f"Note: analysis confidence is {confidence['level']} — "
            f"verify borderline items in interview."
        )

    return {
        "verdict_paragraph": " ".join(p1),
        "strengths_paragraph": " ".join(p2_parts),
        "concerns_paragraph": " ".join(p3_parts),
    }


# =============================================================================
# Multi-JD triage — flips the direction: given one candidate + N open
# roles, rank the roles by how well the candidate fits each. Useful for
# internal-mobility recommendations and "which of our 9 openings is this
# candidate's best landing spot?"
# =============================================================================

@tool("Multi-Job Candidate Triage")
def triage_candidate_across_jobs(candidate_text: str, jobs_json: str) -> str:
    """
    Rank a list of job openings by how well ONE candidate fits each. Runs
    the full Skill Gap Analyzer against every JD and returns a ranked
    list with fit score, tier, top strength, and top gap per role.

    Use this when:
      - A single candidate is considering multiple internal openings.
      - You want to recommend the best landing role from a req board.
      - Evaluating career-pivot feasibility across several role families.

    Input:
      candidate_text: resume text (raw string)
      jobs_json: JSON array of objects, each with 'id', 'title', 'text'
    Output: JSON with ranked array of {job_id, title, fit_score, tier,
    top_match, top_gap, summary}, plus a top-level recommendation.
    """
    if _ner_engine is None or _sbert_matcher is None:
        return json.dumps({"error": "NER engine or SBERT matcher not initialized"})
    try:
        jobs = json.loads(jobs_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"jobs_json must be valid JSON array: {e}"})
    if not isinstance(jobs, list) or not jobs:
        return json.dumps({"error": "jobs_json must be a non-empty JSON array"})

    rankings = []
    for job in jobs:
        jid = job.get("id", "?")
        title = job.get("title", "(untitled)")
        text = job.get("text", "")
        if not text:
            continue
        # Run the full analyzer on this JD
        analysis_str = analyze_skill_gap.run(
            candidate_text=candidate_text, job_text=text
        )
        analysis = json.loads(analysis_str)
        if "error" in analysis:
            continue
        fit = analysis.get("fit_score", {})
        # Top match and top gap (most criticality-weighted)
        top_match, top_gap = None, None
        for etype in ["SKILL", "CERT", "DEGREE"]:
            for m in analysis.get("gaps_by_type", {}).get(etype, {}).get("matched", []):
                if m.get("criticality") == "CRITICAL":
                    top_match = {"requirement": m["required"],
                                 "type": etype.lower(),
                                 "similarity": m["similarity"]}
                    break
            if top_match:
                break
        tg = analysis.get("top_gaps", [])
        if tg:
            top_gap = {"requirement": tg[0]["requirement"],
                       "type": tg[0]["type"],
                       "criticality": tg[0]["criticality"],
                       "action": tg[0].get("action", "")}

        rankings.append({
            "job_id": jid,
            "title": title,
            "fit_score": fit.get("composite_fit_score", 0),
            "recommendation_tier": fit.get("recommendation_tier", "?"),
            "overall_coverage_pct": analysis.get("coverage", {})
                                            .get("overall_coverage_pct", 0),
            "years_aligned": analysis.get("years_experience_gap", {})
                                     .get("meets_requirement"),
            "seniority_alignment": analysis.get("seniority_alignment", {})
                                           .get("alignment"),
            "top_match": top_match,
            "top_gap": top_gap,
            "summary": analysis.get("summary", ""),
        })

    rankings.sort(key=lambda x: -x["fit_score"])

    # Top-level recommendation
    if rankings:
        best = rankings[0]
        rec_line = (
            f"Best fit: {best['title']} ({best['job_id']}) at "
            f"{best['fit_score']}/100 → {best['recommendation_tier']}."
        )
        # Note strong alternatives
        strong_alts = [r for r in rankings[1:]
                       if r["recommendation_tier"] in ("STRONG_HIRE", "HIRE")]
        if strong_alts:
            alt_line = (
                f"Other strong options: "
                + ", ".join(f"{a['title']} ({a['fit_score']})"
                             for a in strong_alts[:3]) + "."
            )
            rec_line += " " + alt_line
    else:
        rec_line = "No valid jobs could be evaluated."

    return json.dumps({
        "rankings": rankings,
        "recommendation": rec_line,
        "evaluated_jobs": len(rankings),
    }, indent=2)


# =============================================================================
# Multi-candidate ranking — given 1 JD + N candidates, rank candidates. This
# is the complement of triage_candidate_across_jobs and is what recruiters
# actually use day-to-day: for a given req, which of the 30 applicants
# should I interview first?
# =============================================================================

@tool("Multi-Candidate Shortlister")
def rank_candidates_for_job(job_text: str, candidates_json: str) -> str:
    """
    Rank a pool of candidate resumes against ONE job description. Runs
    the full Skill Gap Analyzer against every candidate and returns a
    ranked shortlist with fit score, tier, top strength, and top gap
    per candidate. This is the complement of Multi-Job Candidate
    Triage — use it when you have one open req and many applicants.

    Use this when:
      - Shortlisting applicants for a specific req.
      - Prioritizing a recruiter's outreach order.
      - Comparing internal-mobility candidates for one opening.

    Input:
      job_text: job description text (raw string)
      candidates_json: JSON array of objects, each with 'id', 'name', 'text'
    Output: JSON with ranked array of {candidate_id, name, fit_score, tier,
    years_aligned, top_match, top_gap, summary}, plus an interview-queue
    recommendation.
    """
    if _ner_engine is None or _sbert_matcher is None:
        return json.dumps({"error": "NER engine or SBERT matcher not initialized"})
    try:
        candidates = json.loads(candidates_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps(
            {"error": f"candidates_json must be valid JSON array: {e}"}
        )
    if not isinstance(candidates, list) or not candidates:
        return json.dumps(
            {"error": "candidates_json must be a non-empty JSON array"}
        )

    rankings = []
    for cand in candidates:
        cid = cand.get("id", "?")
        name = cand.get("name", "(unnamed)")
        text = cand.get("text", "")
        if not text:
            continue

        analysis_str = analyze_skill_gap.run(
            candidate_text=text, job_text=job_text
        )
        analysis = json.loads(analysis_str)
        if "error" in analysis:
            continue

        fit = analysis.get("fit_score", {})
        # Top strength: first CRITICAL matched skill or cert, else any match
        top_match = None
        for etype in ["SKILL", "CERT", "DEGREE"]:
            for m in analysis.get("gaps_by_type", {}).get(etype, {}).get("matched", []):
                if m.get("criticality") == "CRITICAL":
                    top_match = {"requirement": m["required"],
                                 "type": etype.lower(),
                                 "similarity": m["similarity"],
                                 "proficiency":
                                     (m.get("candidate_evidence") or {}).get(
                                         "proficiency")}
                    break
            if top_match:
                break

        # Top gap
        tg = analysis.get("top_gaps", [])
        top_gap = None
        if tg:
            top_gap = {
                "requirement": tg[0]["requirement"],
                "type": tg[0]["type"],
                "criticality": tg[0]["criticality"],
                "action": tg[0].get("action", ""),
            }

        rankings.append({
            "candidate_id": cid,
            "name": name,
            "fit_score": fit.get("composite_fit_score", 0),
            "recommendation_tier": fit.get("recommendation_tier", "?"),
            "overall_coverage_pct": analysis.get("coverage", {})
                                            .get("overall_coverage_pct", 0),
            "years_aligned": analysis.get("years_experience_gap", {})
                                     .get("meets_requirement"),
            "seniority_alignment": analysis.get("seniority_alignment", {})
                                           .get("alignment"),
            "tier_1_pedigree": analysis.get("employer_pedigree", {})
                                        .get("tier_1_count", 0),
            "resume_impact_signals": analysis.get("resume_impact_signals", []),
            "top_match": top_match,
            "top_gap": top_gap,
            "summary": analysis.get("summary", ""),
        })

    rankings.sort(key=lambda x: -x["fit_score"])

    # Build interview-queue recommendation
    strong = [r for r in rankings
              if r["recommendation_tier"] in ("STRONG_HIRE", "HIRE")]
    interview = [r for r in rankings if r["recommendation_tier"] == "INTERVIEW"]
    conditional = [r for r in rankings
                   if r["recommendation_tier"] == "CONDITIONAL"]
    decline = [r for r in rankings
               if r["recommendation_tier"] == "DO_NOT_ADVANCE"]

    rec_lines = []
    if strong:
        names = ", ".join(f"{r['name']} ({r['fit_score']})" for r in strong[:5])
        rec_lines.append(f"Shortlist / interview first: {names}.")
    if interview:
        names = ", ".join(f"{r['name']} ({r['fit_score']})"
                          for r in interview[:3])
        rec_lines.append(f"Also interview: {names}.")
    if conditional and not strong and not interview:
        names = ", ".join(f"{r['name']} ({r['fit_score']})"
                          for r in conditional[:3])
        rec_lines.append(f"Only borderline candidates available — "
                         f"consider reopening sourcing; tentatively interview: "
                         f"{names}.")
    if decline and not strong and not interview and not conditional:
        rec_lines.append("No candidates meet the bar — reopen sourcing.")
    rec_line = " ".join(rec_lines) or "No candidates evaluated."

    return json.dumps({
        "rankings": rankings,
        "recommendation": rec_line,
        "evaluated_candidates": len(rankings),
        "tier_counts": {
            "STRONG_HIRE": sum(1 for r in rankings
                               if r["recommendation_tier"] == "STRONG_HIRE"),
            "HIRE": sum(1 for r in rankings
                        if r["recommendation_tier"] == "HIRE"),
            "INTERVIEW": sum(1 for r in rankings
                             if r["recommendation_tier"] == "INTERVIEW"),
            "CONDITIONAL": sum(1 for r in rankings
                               if r["recommendation_tier"] == "CONDITIONAL"),
            "DO_NOT_ADVANCE": sum(1 for r in rankings
                                  if r["recommendation_tier"] == "DO_NOT_ADVANCE"),
        },
    }, indent=2)
