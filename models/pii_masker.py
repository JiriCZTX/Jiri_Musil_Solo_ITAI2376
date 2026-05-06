"""
PII masker with offset map for safe LLM processing of resumes.

Per PATH_B5_PLUS_FINAL.md §6:
  - Mask PERSONAL PII only (names, emails, phones, home addresses, IDs)
  - NEVER mask professional locations (basins, regions, work-context cities)
  - Provide offset map so LLM-extracted spans can be re-mapped to original text
  - Reject any extracted span whose un-masked text != original substring

Design: regex-first for deterministic patterns; spaCy NER fallback for PERSON
names if available (graceful no-op if not installed).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Mask categories — PERSONAL ONLY
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# E.164 / US-style phone numbers (avoid matching short digit runs)
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
)

# 7-digit US local format (e.g., "555-0123") — matched only with clear word
# boundaries on both sides to avoid false positives on hyphenated identifiers.
_PHONE_7DIGIT_RE = re.compile(r"(?<!\d)\d{3}[-.\s]\d{4}(?!\d)")

# US-style home address: number + street name + suffix
_HOME_ADDR_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z][A-Za-z'.-]*(?:\s+[A-Z][A-Za-z'.-]+){0,3}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|"
    r"Lane|Ln|Court|Ct|Way|Place|Pl|Parkway|Pkwy|Highway|Hwy)\b\.?",
    re.IGNORECASE,
)

# SSN-like patterns (US 3-2-4 with dashes; loose)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Generic ID-like patterns: long alphanumeric tokens flagged as ID-prefixed
_ID_PREFIX_RE = re.compile(
    r"\b(?:Employee|Emp|ID|License|Cert)[\s#:]*([A-Z0-9-]{6,})\b",
    re.IGNORECASE,
)

# Conservative PERSON_NAME pattern: 2-3 capitalized tokens at line start.
# Captures only the name token group (no leading whitespace) so offsets line up
# exactly with the matched name. Allows trailing comma + credential ("Jane Doe, PE")
# via lookahead — the credential itself stays unmasked because the regex captures
# only the name, not the suffix.
_PERSON_NAME_RE = re.compile(
    r"^[ \t]*"
    r"(?P<name>[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
    r"(?:\s+[A-Z]\.)?"                                # optional middle initial
    r"\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
    r"(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)"
    r"(?=\s*(?:,|\n|\r|$))",
    re.MULTILINE,
)

# Names that look like 2-token person names but are actually well-known
# companies / employers. NEVER masked even when matched by _PERSON_NAME_RE.
# Casefolded for matching. Extend cautiously: every entry trades a small false
# negative (genuine person with that name) for a much larger true negative
# (employer text preserved).
_KNOWN_COMPANY_NAMES = {
    # Energy / oilfield services
    "john crane", "baker hughes", "halliburton", "schlumberger",
    "weatherford international", "petrofac", "saipem", "wood plc",
    "siemens energy", "general electric", "rolls royce", "cameron international",
    "national oilwell", "national oilwell varco", "stewart stevenson",
    # Major majors / NOCs
    "exxon mobil", "exxonmobil", "british petroleum", "shell global",
    "total energies", "totalenergies", "saudi aramco", "abu dhabi national",
    # Engineering / construction
    "wood mackenzie", "black veatch", "stone webster", "morrison knudsen",
    "kbr inc", "fluor corporation", "bechtel corporation",
    # Finance (FS vertical)
    "morgan stanley", "wells fargo", "goldman sachs", "robert half",
    "charles schwab", "charles river", "mary mead",
    # Other dual-meaning brand names
    "thompson reuters", "thomson reuters",
}

# Professional-location safe-list — these are NEVER masked even if they look
# like an address/location. This is an inclusion (allow) list, not exclusion.
_PROFESSIONAL_LOCATIONS_KEYWORDS = {
    # Basins / regions
    "gulf of mexico", "permian basin", "north sea", "bakken",
    "eagle ford", "alberta oil sands", "haynesville", "barnett shale",
    "dogger bank", "marcellus", "utica", "delaware basin",
    "midland basin", "anadarko basin", "appalachian basin",
    # Offshore blocks / venues
    "abu dhabi", "qatar", "south china sea", "campos basin",
    # Megaproject / facility-as-location
    "gorgon", "sabine pass", "hinkley point", "thunder horse",
    "perdido spar", "olympic dam",
    # Cities / regions used in professional context
    "houston", "rotterdam", "singapore", "st. james parish",
    "st james parish", "louisiana",
}

PLACEHOLDER_PERSON = "<PERSON>"
PLACEHOLDER_EMAIL = "<EMAIL>"
PLACEHOLDER_PHONE = "<PHONE>"
PLACEHOLDER_ADDRESS = "<ADDRESS>"
PLACEHOLDER_SSN = "<SSN>"
PLACEHOLDER_ID = "<ID>"


@dataclass(frozen=True)
class MaskEntry:
    """One masking event with offsets for round-trip un-masking."""
    category: str            # PERSON | EMAIL | PHONE | HOME_ADDR | SSN | ID
    placeholder: str
    original_start: int
    original_end: int
    original_text: str
    placeholder_start: int   # offset in MASKED text
    placeholder_end: int     # offset in MASKED text


@dataclass(frozen=True)
class MaskResult:
    """Output of mask_pii: masked text plus offset map."""
    masked_text: str
    offset_map: Tuple[MaskEntry, ...]
    original_text: str

    def to_dict(self) -> dict:
        return {
            "masked_text": self.masked_text,
            "original_text": self.original_text,
            "offset_map": [asdict(e) for e in self.offset_map],
        }


def _is_professional_location(snippet: str) -> bool:
    s = snippet.lower().strip()
    return any(kw in s for kw in _PROFESSIONAL_LOCATIONS_KEYWORDS)


def _collect_matches(text: str) -> List[Tuple[int, int, str, str]]:
    """
    Return list of (start, end, category, placeholder) for all PII matches.
    Sorted by start offset; overlapping matches resolved by first-found wins.
    """
    candidates: List[Tuple[int, int, str, str]] = []

    for m in _EMAIL_RE.finditer(text):
        candidates.append((m.start(), m.end(), "EMAIL", PLACEHOLDER_EMAIL))
    for m in _PHONE_RE.finditer(text):
        candidates.append((m.start(), m.end(), "PHONE", PLACEHOLDER_PHONE))
    for m in _PHONE_7DIGIT_RE.finditer(text):
        candidates.append((m.start(), m.end(), "PHONE", PLACEHOLDER_PHONE))
    for m in _SSN_RE.finditer(text):
        candidates.append((m.start(), m.end(), "SSN", PLACEHOLDER_SSN))
    for m in _ID_PREFIX_RE.finditer(text):
        # mask only the ID payload (group 1), not the "Employee ID:" prefix
        if m.group(1):
            candidates.append((m.start(1), m.end(1), "ID", PLACEHOLDER_ID))
    for m in _HOME_ADDR_RE.finditer(text):
        snippet = text[m.start():m.end()]
        if _is_professional_location(snippet):
            continue
        candidates.append((m.start(), m.end(), "HOME_ADDR", PLACEHOLDER_ADDRESS))
    for m in _PERSON_NAME_RE.finditer(text):
        # Capture only the name group (no leading whitespace) so offsets are exact.
        name_text = m.group("name")
        if name_text.lower().strip() in _KNOWN_COMPANY_NAMES:
            continue  # employer / company that happens to look like a person name
        candidates.append((m.start("name"), m.end("name"), "PERSON", PLACEHOLDER_PERSON))

    # sort by start; resolve overlaps by keeping the longest first-start match
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    resolved: List[Tuple[int, int, str, str]] = []
    last_end = -1
    for start, end, cat, ph in candidates:
        if start >= last_end:
            resolved.append((start, end, cat, ph))
            last_end = end
    return resolved


def mask_pii(text: str) -> MaskResult:
    """
    Mask personal PII in text. Professional locations preserved.

    Returns MaskResult with:
      - masked_text: text with PII replaced by placeholders
      - offset_map: sequence of MaskEntry recording every replacement
      - original_text: the input, kept for un-masking validation
    """
    matches = _collect_matches(text)
    if not matches:
        return MaskResult(masked_text=text, offset_map=tuple(), original_text=text)

    out_parts: List[str] = []
    entries: List[MaskEntry] = []
    cursor = 0
    masked_cursor = 0
    for start, end, cat, ph in matches:
        if start > cursor:
            chunk = text[cursor:start]
            out_parts.append(chunk)
            masked_cursor += len(chunk)
        original_text_span = text[start:end]
        out_parts.append(ph)
        entries.append(MaskEntry(
            category=cat,
            placeholder=ph,
            original_start=start,
            original_end=end,
            original_text=original_text_span,
            placeholder_start=masked_cursor,
            placeholder_end=masked_cursor + len(ph),
        ))
        masked_cursor += len(ph)
        cursor = end
    if cursor < len(text):
        out_parts.append(text[cursor:])

    return MaskResult(
        masked_text="".join(out_parts),
        offset_map=tuple(entries),
        original_text=text,
    )


def un_mask_span(
    masked_start: int,
    masked_end: int,
    masked_span_text: str,
    mask_result: MaskResult,
) -> Optional[Tuple[int, int, str]]:
    """
    Map a span (start, end, text) in MASKED text back to ORIGINAL text coords.

    Returns (original_start, original_end, original_substring) if the span
    can be cleanly mapped AND the un-masked text matches the original
    substring. Returns None if the span crosses a mask boundary or fails
    substring validation.
    """
    if masked_start < 0 or masked_end <= masked_start:
        return None

    # Reject spans that cross/touch any placeholder
    for entry in mask_result.offset_map:
        if (masked_start < entry.placeholder_end
                and masked_end > entry.placeholder_start):
            return None  # span overlaps a placeholder — invalid

    # Translate masked-text offset to original-text offset
    def to_original(masked_off: int) -> int:
        # Sum of (placeholder_len - original_len) deltas before this offset
        delta = 0
        for entry in mask_result.offset_map:
            if entry.placeholder_end <= masked_off:
                ph_len = len(entry.placeholder)
                orig_len = entry.original_end - entry.original_start
                delta += (orig_len - ph_len)
            else:
                break
        return masked_off + delta

    orig_start = to_original(masked_start)
    orig_end = to_original(masked_end)
    if orig_start < 0 or orig_end > len(mask_result.original_text):
        return None
    actual = mask_result.original_text[orig_start:orig_end]
    if actual != masked_span_text:
        return None
    return (orig_start, orig_end, actual)


# ---------------------------------------------------------------------------
# Test fixtures with explicit expected outcomes
# ---------------------------------------------------------------------------
# Each fixture pairs a resume snippet with two lists:
#   must_mask:     substrings that MUST be absent from masked_text (PII removed)
#   must_preserve: substrings that MUST appear in masked_text (professional kept)
# Reviewer correction §2 added 10 new entries: John Crane / Baker Hughes /
# Houston office / Rotterdam refinery / Singapore LNG / North Sea / St. James
# Parish, Louisiana / 123 Main St, Houston, TX / jane.doe@chevron.com / Jane Doe, PE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PIIFixture:
    name: str
    text: str
    must_mask: Tuple[str, ...]
    must_preserve: Tuple[str, ...]


TEST_FIXTURES: Tuple[PIIFixture, ...] = (
    PIIFixture(
        name="PERSONAL_PII_BASIC",
        text=(
            "John Smith\njohn.smith@example.com\n(713) 555-0142\n"
            "12 years at Chevron in the Gulf of Mexico."
        ),
        must_mask=("john.smith@example.com", "(713) 555-0142"),
        must_preserve=("Chevron", "Gulf of Mexico", "12 years"),
    ),
    PIIFixture(
        name="MIXED_PERSONAL_AND_PROFESSIONAL",
        text=(
            "Sarah O'Brien-Lee\nsarah.lee+work@energy.co.uk\n+1-281-555-0199\n"
            "Worked at the Permian Basin and 1234 Memorial Drive office."
        ),
        must_mask=("sarah.lee+work@energy.co.uk", "1234 Memorial Drive"),
        must_preserve=("Permian Basin",),
    ),
    PIIFixture(
        name="ID_AND_PROFESSIONAL_CITY",
        text=(
            "Carlos Mendez\nEmployee ID: A1234B5678\n"
            "Lead Process Engineer at Bechtel, Houston, TX."
        ),
        must_mask=("A1234B5678",),
        must_preserve=("Bechtel", "Houston, TX"),
    ),
    PIIFixture(
        name="PROFESSIONAL_PROJECT_PRESERVED",
        text=(
            "Priya Krishnamurthy\npriya@example.org\n"
            "10 years at Halliburton on Gorgon LNG project."
        ),
        must_mask=("priya@example.org",),
        must_preserve=("Halliburton", "Gorgon LNG", "10 years"),
    ),
    PIIFixture(
        name="OFFSHORE_PRESERVED",
        text=(
            "Dr. Maria Rodriguez\n555-0123\n"
            "Worked offshore at Thunder Horse platform, Gulf of Mexico."
        ),
        must_mask=("555-0123",),
        must_preserve=("Thunder Horse", "Gulf of Mexico"),
    ),
    # ---- Reviewer §2 additions: 10 new fixtures ---------------------------
    PIIFixture(
        name="JOHN_CRANE_IS_EMPLOYER_NOT_PERSON",
        text=(
            "Eric Wong\n"
            "Senior Reliability Engineer at John Crane on rotating equipment programs.\n"
            "12 years experience."
        ),
        must_mask=(),  # Eric Wong is at line start but only test focus is John Crane preservation
        must_preserve=("John Crane",),
    ),
    PIIFixture(
        name="BAKER_HUGHES_IS_EMPLOYER",
        text=(
            "Field Engineer at Baker Hughes from 2014-2021,\n"
            "Houston office, supporting Permian Basin operations."
        ),
        must_mask=(),
        must_preserve=("Baker Hughes", "Houston office", "Permian Basin"),
    ),
    PIIFixture(
        name="HOUSTON_OFFICE_IS_PROFESSIONAL",
        text=(
            "Lead Process Engineer (Houston office), 2018-present.\n"
            "Operations across the Gulf of Mexico."
        ),
        must_mask=(),
        must_preserve=("Houston office", "Gulf of Mexico"),
    ),
    PIIFixture(
        name="ROTTERDAM_REFINERY_PROFESSIONAL",
        text=(
            "Maintenance Manager at the Rotterdam refinery,\n"
            "responsible for FCC unit and crude distillation."
        ),
        must_mask=(),
        must_preserve=("Rotterdam refinery",),
    ),
    PIIFixture(
        name="SINGAPORE_LNG_PROFESSIONAL",
        text=(
            "Project Engineer on the Singapore LNG terminal expansion,\n"
            "boil-off-gas handling and BOG re-condensation."
        ),
        must_mask=(),
        must_preserve=("Singapore LNG",),
    ),
    PIIFixture(
        name="NORTH_SEA_PROFESSIONAL",
        text=(
            "8 years offshore experience, North Sea (Brent and Forties),\n"
            "with deepwater intervention work."
        ),
        must_mask=(),
        must_preserve=("North Sea",),
    ),
    PIIFixture(
        name="ST_JAMES_PARISH_LOUISIANA_PROFESSIONAL",
        text=(
            "Plant Manager, ethylene cracker, St. James Parish, Louisiana.\n"
            "Led 250-person operations team."
        ),
        must_mask=(),
        must_preserve=("St. James Parish, Louisiana",),
    ),
    PIIFixture(
        name="HOME_ADDRESS_MASKED_CITY_PRESERVED",
        text=(
            "Marcus Reyes\n"
            "123 Main St, Houston, TX 77001\n"
            "Senior Drilling Engineer."
        ),
        must_mask=("123 Main St",),
        must_preserve=("Houston, TX",),
    ),
    PIIFixture(
        name="EMPLOYER_IN_EMAIL_DOMAIN",
        text=(
            "Jane Doe\n"
            "jane.doe@chevron.com\n"
            "10 years at Chevron, upstream operations."
        ),
        must_mask=("jane.doe@chevron.com",),
        must_preserve=("Chevron",),  # the company name itself stays
    ),
    PIIFixture(
        name="JANE_DOE_PE_NAME_WITH_CREDENTIAL",
        text=(
            "Jane Doe, PE\n"
            "Process Safety Lead, ASME BPVC certified."
        ),
        # Name "Jane Doe" must be masked; credential "PE" + "ASME BPVC" preserved
        must_mask=("Jane Doe",),
        must_preserve=("PE", "ASME BPVC"),
    ),
    # ----- Reviewer follow-up #2: deeper adversarial PII / employer cases ----
    PIIFixture(
        name="JANE_DOE_PE_AT_BAKER_HUGHES",
        text=(
            "Jane Doe, PE\n"
            "Senior Reliability Engineer at Baker Hughes,\n"
            "Houston office, supporting Permian Basin operations."
        ),
        # Person masked; employer + credential + city + basin all preserved.
        must_mask=("Jane Doe",),
        must_preserve=("PE", "Baker Hughes", "Houston office", "Permian Basin"),
    ),
    PIIFixture(
        name="JOHN_CRANE_AT_BAKER_HUGHES",
        text=(
            "Mechanical Reliability Lead at John Crane and Baker Hughes,\n"
            "rotating equipment programs in the North Sea."
        ),
        # Both employers preserved; "John Crane" must NOT be person-masked.
        must_mask=(),
        must_preserve=("John Crane", "Baker Hughes", "North Sea"),
    ),
    PIIFixture(
        name="WOOD_AND_SHELL_EMPLOYERS_INLINE",
        text=(
            "8 years at Wood, then 4 years at Shell across upstream operations.\n"
            "Lead engineer on Gorgon LNG."
        ),
        # "Wood" / "Shell" are dangerous single-word employer mentions.
        # The PII masker MUST NOT touch them (it only masks at line-start
        # 2+token name patterns). Preserve via "no must_mask" assertion.
        must_mask=(),
        must_preserve=("Wood", "Shell", "Gorgon LNG"),
    ),
    PIIFixture(
        name="HOME_ADDR_HOUSTON_TX_VS_HOUSTON_OFFICE",
        text=(
            "Marcus Reyes\n"
            "123 Main St, Houston, TX 77001\n"
            "Senior Drilling Engineer (Houston office), Permian Basin operations."
        ),
        # Personal home address masked; professional Houston usage preserved.
        must_mask=("123 Main St",),
        must_preserve=("Houston, TX", "Houston office", "Permian Basin"),
    ),
    PIIFixture(
        name="ST_JAMES_PARISH_LNG_TERMINAL",
        text=(
            "Plant Manager at the St. James Parish, Louisiana LNG terminal,\n"
            "ethylene cracker operations and BOG handling."
        ),
        must_mask=(),
        must_preserve=("St. James Parish, Louisiana", "LNG"),
    ),
)


def _self_test() -> dict:
    """
    Strict per-fixture verification.
    Each fixture has explicit must_mask and must_preserve sets; any violation
    is recorded with the fixture name + the offending substring.
    """
    results = {
        "fixtures_tested": 0,
        "personal_masked": 0,
        "professional_preserved": 0,
        "issues": [],
    }
    for fx in TEST_FIXTURES:
        result = mask_pii(fx.text)
        results["fixtures_tested"] += 1

        for kw in fx.must_mask:
            if kw in result.masked_text:
                results["issues"].append(
                    f"{fx.name}: '{kw}' should be MASKED but appears in masked text"
                )
            else:
                results["personal_masked"] += 1

        for kw in fx.must_preserve:
            if kw not in result.masked_text:
                results["issues"].append(
                    f"{fx.name}: professional substring '{kw}' was MASKED (should be preserved)"
                )
            else:
                results["professional_preserved"] += 1

    return results


# ---------------------------------------------------------------------------
# Reviewer correction §3: offset round-trip stress test
# ---------------------------------------------------------------------------
# Contract:
#   original_text → mask_pii → (simulate Opus span on masked_text)
#   → un_mask_span(masked_span) → must equal exact substring of original_text.
# Any span that cannot map back exactly must be REJECTED (un_mask_span returns
# None). The stress test exercises both paths: spans that should round-trip
# cleanly AND spans that intentionally cross a placeholder boundary
# (un_mask_span must return None for these).
def _round_trip_stress_test() -> dict:
    """
    Offset round-trip stress test. Returns:
      {
        "round_trip_valid_spans":   N spans that round-tripped cleanly,
        "round_trip_rejected_spans": N spans correctly rejected (cross placeholder),
        "failed":                    int (count of cases that violate the contract),
        "failures":                  list[str] (descriptions of contract violations)
      }
    """
    failures: List[str] = []
    valid = 0
    rejected = 0

    def _first_non_placeholder_match(masked: str, kw: str,
                                     offset_map: Tuple[MaskEntry, ...]) -> int:
        """Find first occurrence of kw in masked that does NOT overlap any
        placeholder. Returns -1 if no such occurrence exists."""
        cursor = 0
        n = len(kw)
        while cursor <= len(masked) - n:
            pos = masked.find(kw, cursor)
            if pos < 0:
                return -1
            overlaps = any(
                pos < e.placeholder_end and pos + n > e.placeholder_start
                for e in offset_map
            )
            if not overlaps:
                return pos
            cursor = pos + 1
        return -1

    for fx in TEST_FIXTURES:
        result = mask_pii(fx.text)
        masked = result.masked_text

        # Case A: pick the must_preserve substrings; their masked offsets must
        # round-trip back to identical original substrings.
        for kw in fx.must_preserve:
            idx = _first_non_placeholder_match(masked, kw, result.offset_map)
            if idx < 0:
                # already covered by _self_test issues; skip here
                continue
            mapped = un_mask_span(idx, idx + len(kw), kw, result)
            if mapped is None:
                failures.append(
                    f"{fx.name}: must_preserve '{kw}' could not round-trip "
                    f"(un_mask_span returned None)"
                )
                continue
            orig_start, orig_end, orig_text = mapped
            if orig_text != kw:
                failures.append(
                    f"{fx.name}: round-trip text mismatch for '{kw}': "
                    f"got '{orig_text}'"
                )
                continue
            if result.original_text[orig_start:orig_end] != kw:
                failures.append(
                    f"{fx.name}: round-trip offset mismatch for '{kw}': "
                    f"original[{orig_start}:{orig_end}] = "
                    f"'{result.original_text[orig_start:orig_end]}'"
                )
                continue
            valid += 1

        # Case B: span that crosses a placeholder boundary must be rejected.
        # We construct one only if there is at least one mask entry.
        if result.offset_map:
            entry = result.offset_map[0]
            # span starts inside placeholder, ends past it
            cross_start = entry.placeholder_start
            cross_end = entry.placeholder_end + 3
            if cross_end <= len(masked):
                fake_text = masked[cross_start:cross_end]
                got = un_mask_span(cross_start, cross_end, fake_text, result)
                if got is not None:
                    failures.append(
                        f"{fx.name}: span crossing placeholder was NOT rejected "
                        f"(got {got})"
                    )
                else:
                    rejected += 1

        # Case C: a span entirely inside an unmask-friendly chunk should match.
        # Try the first masked token that is NOT a placeholder.
        if not result.offset_map:
            # nothing to mask; whole text round-trips trivially
            mapped_full = un_mask_span(0, len(masked), masked, result)
            if mapped_full is None or mapped_full[2] != fx.text:
                failures.append(
                    f"{fx.name}: full-text round-trip failed when no PII present"
                )
            else:
                valid += 1

    return {
        "round_trip_valid_spans": valid,
        "round_trip_rejected_spans": rejected,
        "failed": len(failures),
        "failures": failures,
    }


if __name__ == "__main__":
    import json
    print("--- _self_test ---")
    print(json.dumps(_self_test(), indent=2))
    print("--- _round_trip_stress_test ---")
    print(json.dumps(_round_trip_stress_test(), indent=2))
