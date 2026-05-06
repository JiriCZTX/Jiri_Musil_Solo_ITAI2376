"""
TEMPLATE — copy this file to `industries/<industry>.py` and fill it in
to onboard a new vertical. Then add the import + `set_industry()` line
in `industries/__init__.py`.

To seed values from ESCO or O*NET automatically, query their APIs with
the industry's top-level category and map:
  - ESCO skills → skill_synonyms / skill_clusters
  - O*NET SOC codes → onet_crosswalk_codes (one per department)
  - BLS OOH wage tables → dept_economics.base_salary
  - BLS time-to-fill survey → dept_economics.fill_days
  - Industry-standard certification lists → critical_certifications
  - Industry conference sponsor lists → tier_1_employers

Time-to-build a new vertical profile: 1-3 days of research + <200 lines
of configuration. No code changes to tool layer, brain, or dashboard.
"""
from .base import IndustryBase, DeptEconomics, Intervention, CohortThresholds


# TODO: rename to the vertical (e.g., FINANCE, HEALTHCARE, TECH, MANUFACTURING)
TEMPLATE = IndustryBase(
    name="template",
    display_name="Template Industry",
    description="Drop-in template for a new vertical.",

    # ---- Per-department economics --------------------------------
    # Pull base_salary from BLS OOH wage tables; fill_days from
    # industry time-to-hire surveys; replacement_multiplier from
    # SHRM Benchmarking (typical: 1.2-1.8× salary for white-collar,
    # 0.5-1.0× for operational roles).
    dept_economics={
        # "DeptName": DeptEconomics(
        #     base_salary=XXX, fill_days=XX, replacement_multiplier=X.X,
        #     safety_critical=bool, hiring_difficulty="easy/moderate/hard",
        # ),
    },

    # ---- Retention interventions ---------------------------------
    # Reuse the 6 canonical levers from energy unless the industry
    # has vertical-specific ones (e.g., healthcare clinician signing
    # bonuses, tech equity refresh grants).
    interventions={
        # "name": Intervention(...),
    },

    # ---- Cohort thresholds ---------------------------------------
    # Usually fine to keep defaults. Override only when the industry's
    # retirement age, tenure norms, or engagement norms differ
    # materially (e.g., tech: tenured_ip_years=4.0 not 8.0).
    cohort_thresholds=CohortThresholds(
        # retirement_critical_age=58,
        # tenured_ip_years=8.0,
        # ...
    ),

    # ---- Cross-dept adjacency ------------------------------------
    # Which departments share transferable skills with which?
    # Score 0.0 (no transfer) → 1.0 (interchangeable).
    dept_adjacency={
        # "DeptA": {"DeptB": 0.75, "DeptC": 0.45},
    },

    # ---- Feature labels ------------------------------------------
    # For driver attribution display. Usually identical to the
    # 8 Bi-LSTM features — override only if the model was retrained
    # with vertical-specific inputs.
    feature_labels={
        # "comp_ratio": "Compensation ratio",
        # ...
    },

    # ---- Tier-1 employers ----------------------------------------
    # Which employer names, when they appear on a resume, signal
    # strong pedigree? Lowercased for matching.
    tier_1_employers=[
        # "jpmorgan", "goldman sachs", ...  (for finance)
        # "mayo clinic", "cleveland clinic", ...  (for healthcare)
    ],

    # ---- Critical certifications ---------------------------------
    # Which certs / licenses are load-bearing in this industry?
    critical_certifications=[
        # "Series 7", "CFA", ...  (for finance)
        # "MD", "BLS", "ACLS", ...  (for healthcare)
    ],

    # ---- Taxonomy sources ----------------------------------------
    # Which authoritative sources fed the training data / canonicalizer?
    taxonomy_source_refs=[
        # "ESCO finance skills", "CFA Institute Body of Knowledge", ...
    ],

    # ---- O*NET crosswalk -----------------------------------------
    # Map each department to its canonical SOC code. Unlocks O*NET
    # auto-enrichment of role descriptions and skills.
    onet_crosswalk_codes={
        # "DeptName": "XX-XXXX",
    },
)
