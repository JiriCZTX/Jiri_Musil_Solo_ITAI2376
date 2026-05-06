"""
IndustryBase — the abstract industry profile.

Every vertical (energy, finance, healthcare, tech, ...) provides a
concrete instance of this dataclass with its domain-specific constants.
The tool layer consults the active profile instead of hardcoding
constants — enabling zero-edit multi-industry deployment.

Design principles
-----------------
1.  Declarative, not behavioral — profiles are data, not logic. All the
    reasoning logic stays in tools/ and brain.py.
2.  Everything has a default — profiles only need to override what's
    industry-specific, so a new vertical is <200 lines of configuration.
3.  Runtime swappable — set_industry(profile) re-points the global
    active profile; all subsequent tool calls pick it up.
4.  Maps cleanly onto ESCO / O*NET classifications — so you can seed a
    new industry profile from those taxonomies automatically.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class DeptEconomics:
    """Per-role replacement economics."""
    base_salary: int
    fill_days: int
    replacement_multiplier: float
    safety_critical: bool = False
    hiring_difficulty: str = "moderate"  # easy / moderate / hard


@dataclass
class Intervention:
    """A retention lever that HR can activate."""
    description: str
    feature_affected: Optional[str]  # which Bi-LSTM feature it moves
    effect_per_unit: float
    cost_per_head_per_unit: int
    lead_time_days: int
    reduces_knowledge_loss: bool = False
    # Causal annotation — CAUSAL / MIXED / CORRELATIONAL / NONE.
    # Distinguishes direct causal pathways from confounded learned
    # correlations in the Bi-LSTM. The simulator's counterfactual
    # probability is only trustworthy for CAUSAL/MIXED levers;
    # CORRELATIONAL levers may have direction-wrong counterfactuals
    # (see WRITEUP §16.2 paradox). Default MIXED is the safe middle.
    causal_status: str = "MIXED"
    causal_rationale: str = ""


@dataclass
class CohortThresholds:
    """Where the industry draws its cohort boundaries."""
    retirement_critical_age: int = 58
    retirement_emerging_age: int = 55
    new_hire_tenure_years: float = 2.0
    low_engagement_score: float = 2.5
    low_satisfaction_score: float = 2.5
    comp_gap_ratio: float = 0.90
    tenured_ip_years: float = 8.0
    high_performer_score: float = 4.0


@dataclass
class IndustryBase:
    """
    The full profile for one vertical.

    Concrete instances (e.g., ENERGY in industries/energy.py) fill in
    all seven slots below. Downstream code reads the active profile via
    `get_industry()`.
    """
    name: str                                            # "energy" / "finance" / ...
    display_name: str                                    # "Energy Industry"
    description: str                                     # 1-line blurb
    dept_economics: Dict[str, DeptEconomics]             # dept -> economics
    interventions: Dict[str, Intervention]               # name -> intervention
    cohort_thresholds: CohortThresholds = field(
        default_factory=CohortThresholds
    )
    dept_adjacency: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_labels: Dict[str, str] = field(default_factory=dict)
    tier_1_employers: List[str] = field(default_factory=list)
    skill_clusters: Dict[str, List[str]] = field(default_factory=dict)
    skill_synonyms: Dict[str, List[str]] = field(default_factory=dict)
    interview_templates: Dict[str, str] = field(default_factory=dict)
    critical_certifications: List[str] = field(default_factory=list)
    taxonomy_source_refs: List[str] = field(default_factory=list)
    onet_crosswalk_codes: Dict[str, str] = field(default_factory=dict)
    # Synthetic workforce-data generator inputs (consumed by
    # data/generate_workforce_data.py when this profile is active):
    dept_sizes: Dict[str, int] = field(default_factory=dict)
    roles_by_dept: Dict[str, List[str]] = field(default_factory=dict)
    # Which subdirectory under data/ holds this vertical's workforce CSVs.
    # Energy lives at the root (data/employees.csv ...) for backcompat —
    # `data_subdir=""`. New verticals live under `data/<subdir>/*.csv`.
    data_subdir: str = ""

    def summarize(self) -> Dict[str, Any]:
        """Quick stats block — useful for the dashboard + tests."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "n_departments": len(self.dept_economics),
            "n_interventions": len(self.interventions),
            "n_adjacency_edges": sum(len(v) for v in self.dept_adjacency.values()),
            "n_feature_labels": len(self.feature_labels),
            "n_tier_1_employers": len(self.tier_1_employers),
            "n_skill_clusters": len(self.skill_clusters),
            "n_skill_synonyms": len(self.skill_synonyms),
            "n_interview_templates": len(self.interview_templates),
            "n_critical_certifications": len(self.critical_certifications),
            "taxonomy_source_refs": self.taxonomy_source_refs,
        }


# =============================================================================
# Global active profile + accessors
# =============================================================================

_ACTIVE_PROFILE: Optional[IndustryBase] = None


def set_industry(profile: IndustryBase) -> None:
    """Install a new active industry profile. Subsequent tool calls
    consult this profile for all domain-specific constants."""
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = profile


def get_industry() -> IndustryBase:
    """Return the active profile. Errors if none is installed — which
    should never happen because industries/__init__.py installs ENERGY
    as the default."""
    if _ACTIVE_PROFILE is None:
        raise RuntimeError(
            "No active industry profile installed. Call "
            "industries.set_industry(profile) before using tools."
        )
    return _ACTIVE_PROFILE
