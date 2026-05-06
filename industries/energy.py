"""
Energy Industry profile — the vertical this project validates.

Covers upstream oil & gas, midstream, power generation, utilities,
renewables (solar / wind / geothermal / BESS), nuclear, refining,
and hydrogen / DAC / CCUS. Taxonomy references:
  - O*NET 22 SOC codes (onetcenter.org/database.html)
  - ESCO (esco.ec.europa.eu)
  - API ICP (American Petroleum Institute)
  - NACE / AMPP
  - CEWD / DOL Energy Competency Model
  - IRENA / SPE / EPRI / IEA / BLS / IADC / ASME / ISA
"""
from .base import IndustryBase, DeptEconomics, Intervention, CohortThresholds


ENERGY = IndustryBase(
    name="energy",
    display_name="Energy Industry",
    description=(
        "Oil & gas (upstream / midstream / downstream) + power generation "
        "+ utilities + renewables + nuclear + hydrogen / CCUS. Validated "
        "against 16 authoritative taxonomy sources."
    ),
    dept_economics={
        "Operations": DeptEconomics(
            base_salary=95_000, fill_days=75, replacement_multiplier=1.5,
            safety_critical=True, hiring_difficulty="moderate",
        ),
        "Engineering": DeptEconomics(
            base_salary=130_000, fill_days=120, replacement_multiplier=1.8,
            safety_critical=True, hiring_difficulty="hard",
        ),
        "Maintenance": DeptEconomics(
            base_salary=85_000, fill_days=60, replacement_multiplier=1.4,
            safety_critical=True, hiring_difficulty="hard",
        ),
        "HSE": DeptEconomics(
            base_salary=110_000, fill_days=100, replacement_multiplier=1.6,
            safety_critical=True, hiring_difficulty="hard",
        ),
        "Projects": DeptEconomics(
            base_salary=125_000, fill_days=90, replacement_multiplier=1.7,
            safety_critical=False, hiring_difficulty="moderate",
        ),
        "Commercial": DeptEconomics(
            base_salary=105_000, fill_days=70, replacement_multiplier=1.3,
            safety_critical=False, hiring_difficulty="moderate",
        ),
        "IT": DeptEconomics(
            base_salary=110_000, fill_days=60, replacement_multiplier=1.4,
            safety_critical=False, hiring_difficulty="easy",
        ),
        "HR": DeptEconomics(
            base_salary=90_000, fill_days=55, replacement_multiplier=1.3,
            safety_critical=False, hiring_difficulty="easy",
        ),
    },
    interventions={
        "comp_adjustment": Intervention(
            description="Off-cycle compensation adjustment to close market gap.",
            feature_affected="comp_ratio", effect_per_unit=0.08,
            cost_per_head_per_unit=10_000, lead_time_days=30,
            causal_status="MIXED",
            causal_rationale=(
                "Direct causal pathway exists (higher pay → stay), but the "
                "Bi-LSTM's comp_ratio→attrition association is confounded by "
                "outside_options (higher earners attract more recruiters). "
                "Treat simulated magnitude as an approximation, not a ground "
                "truth; direction is correct."
            ),
        ),
        "satisfaction_program": Intervention(
            description="Manager coaching + quarterly 1:1 + career pathing review.",
            feature_affected="satisfaction", effect_per_unit=0.10,
            cost_per_head_per_unit=2_400, lead_time_days=90,
            causal_status="CORRELATIONAL",
            causal_rationale=(
                "Satisfaction→attrition in training data is confounded by "
                "selection (satisfied high-performers are headhunted more). "
                "The simulator may show a SPURIOUS +Δ attrition under the "
                "counterfactual — the §16.2 paradox. Real causal effect is "
                "likely negative but unobservable via this associational model. "
                "Treat the simulator output as an UPPER BOUND on harm and rely "
                "on published HR-program effect sizes for the lower bound."
            ),
        ),
        "engagement_initiative": Intervention(
            description="Team-level engagement sprint (recognition, autonomy, purpose).",
            feature_affected="engagement", effect_per_unit=0.07,
            cost_per_head_per_unit=1_500, lead_time_days=60,
            causal_status="CORRELATIONAL",
            causal_rationale=(
                "Same confounding as satisfaction_program — engaged "
                "high-performers draw outside recruiters. Simulator effect "
                "direction is unreliable; cite published engagement-program "
                "effect sizes instead."
            ),
        ),
        "retention_bonus": Intervention(
            description="Stay-bonus with 12-month cliff vesting.",
            feature_affected="comp_ratio", effect_per_unit=0.05,
            cost_per_head_per_unit=8_000, lead_time_days=14,
            causal_status="MIXED",
            causal_rationale=(
                "Direct causal mechanism (golden handcuffs during vesting), "
                "but reuses the comp_ratio confounded association. Magnitude "
                "approximate, direction correct."
            ),
        ),
        "flex_work": Intervention(
            description="Hybrid / compressed-schedule / field-rotation options.",
            feature_affected="satisfaction", effect_per_unit=0.04,
            cost_per_head_per_unit=500, lead_time_days=45,
            causal_status="CORRELATIONAL",
            causal_rationale=(
                "Applied through the satisfaction channel — same confounding "
                "as satisfaction_program. Real-world flex-work studies (SHRM, "
                "Gartner) show ~15% retention lift; trust those, not the "
                "simulator."
            ),
        ),
        "knowledge_capture": Intervention(
            description="Senior-to-junior pairing + documented SOPs before retirement.",
            feature_affected=None, effect_per_unit=0.0,
            cost_per_head_per_unit=3_500, lead_time_days=60,
            reduces_knowledge_loss=True,
            causal_status="CAUSAL",
            causal_rationale=(
                "This intervention reduces the COST of a departure (by "
                "capturing institutional knowledge) rather than the probability "
                "of one. The cost-reduction pathway is direct and unconfounded."
            ),
        ),
    },
    cohort_thresholds=CohortThresholds(
        retirement_critical_age=58, retirement_emerging_age=55,
        new_hire_tenure_years=2.0,
        low_engagement_score=2.5, low_satisfaction_score=2.5,
        comp_gap_ratio=0.90, tenured_ip_years=8.0, high_performer_score=4.0,
    ),
    dept_adjacency={
        "Operations":   {"Maintenance": 0.85, "HSE": 0.55, "Engineering": 0.40},
        "Engineering":  {"Projects": 0.80, "IT": 0.50, "Operations": 0.40, "Maintenance": 0.45},
        "Maintenance":  {"Operations": 0.85, "Engineering": 0.45, "HSE": 0.40},
        "HSE":          {"Operations": 0.55, "Engineering": 0.45, "Maintenance": 0.40},
        "Projects":     {"Engineering": 0.80, "Commercial": 0.55, "Operations": 0.35},
        "Commercial":   {"Projects": 0.55, "HR": 0.30},
        "IT":           {"Engineering": 0.50, "Commercial": 0.30},
        "HR":           {"Commercial": 0.30},
    },
    feature_labels={
        "tenure_years":    "Tenure",
        "comp_ratio":      "Compensation ratio",
        "satisfaction":    "Job satisfaction",
        "performance":     "Performance",
        "engagement":      "Engagement",
        "age":             "Age / retirement pressure",
        "bls_energy_index": "External labor market (BLS energy index)",
        "headcount":       "Department size trend",
    },
    tier_1_employers=[
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
        "vestas", "orsted", "first solar", "enphase", "sunpower",
        # Midstream
        "enterprise products", "williams companies", "williams", "kinder morgan",
        "enbridge", "energy transfer",
    ],
    critical_certifications=[
        "API 570", "API 510", "API 1169", "API 653", "API 571",
        "NACE CIP", "AMPP CIP", "NEBOSH", "CSP", "OSHA 30",
        "PE license", "PMP", "GWO", "ISA CCST",
    ],
    taxonomy_source_refs=[
        "O*NET 22 SOC codes", "ESCO skills+competences", "API ICP",
        "NACE/AMPP", "CEWD/DOL Energy Competency Model",
        "IRENA Renewable Skills", "SPE Competency Model",
        "EPRI Workforce Planning", "IEA World Energy Employment",
        "BLS Occupational Outlook Handbook", "IADC Drilling Contractors",
        "ASME BPVC+B31", "ISA Body of Knowledge",
        "USAJOBS", "Greenhouse public boards", "Lever public postings",
    ],
    onet_crosswalk_codes={
        "Operations": "51-8093",       # Petroleum, Pump System, and Refinery Operators
        "Engineering": "17-2041",       # Chemical Engineers (canonical)
        "Maintenance": "49-9041",       # Industrial Machinery Mechanics
        "HSE": "29-9011",               # Occupational Health and Safety Specialists
        "Projects": "17-2199",          # Engineers, All Other
        "Commercial": "13-1081",        # Logisticians
        "IT": "15-1212",                # Information Security Analysts
        "HR": "13-1071",                # Human Resources Specialists
    },
    # ---- Synthetic workforce-data generator inputs ----
    # These were previously hardcoded in data/generate_workforce_data.py;
    # moved here so the generator reads from the active industry profile.
    dept_sizes={
        "Operations": 80, "Engineering": 60, "Maintenance": 50,
        "HSE": 20, "Projects": 35, "Commercial": 20, "IT": 20, "HR": 15,
    },
    roles_by_dept={
        "Operations": ["Plant Operator", "Control Room Technician",
                        "Shift Supervisor", "Operations Manager",
                        "Field Operator"],
        "Engineering": ["Process Engineer", "Electrical Engineer",
                         "Mechanical Engineer", "Instrumentation Engineer",
                         "Project Engineer"],
        "Maintenance": ["Maintenance Technician", "Reliability Engineer",
                         "Planner/Scheduler", "Welding Inspector",
                         "NDT Technician"],
        "HSE": ["HSE Coordinator", "Safety Engineer",
                "Environmental Specialist", "Fire Protection Engineer"],
        "Projects": ["Project Manager", "Construction Manager",
                      "Commissioning Engineer", "Cost Engineer",
                      "Scheduler"],
        "Commercial": ["Commercial Analyst", "Contracts Manager",
                        "Procurement Specialist"],
        "IT": ["Systems Engineer", "Data Analyst", "SCADA Specialist",
                "Cybersecurity Analyst"],
        "HR": ["HR Business Partner", "Talent Acquisition Specialist",
                "Compensation Analyst"],
    },
    # Energy's data lives at data/ root for backcompat — subdirectory
    # is the empty string so loaders read data/employees.csv directly.
    data_subdir="",
)
