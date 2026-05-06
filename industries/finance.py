"""
Financial Services Industry profile — second vertical.

Covers investment banking, equity research, risk & compliance, wealth
management, retail banking, fintech, trading, and operations. Economics
calibrated from:
  - BLS Occupational Employment Statistics (OES) — SOC 13-2052
    (personal financial advisors), 13-2031 (budget analysts),
    13-2011 (accountants), 11-3031 (financial managers), etc.
  - Robert Half Salary Guide 2025 — FS compensation bands
  - SHRM Benchmarking — time-to-fill for regulated roles
  - FINRA / NFA / SEC — registration and license catalogs
  - CFA Institute — Body of Knowledge + ethics framework
  - CFP Board — planner certification pathway
  - GARP / PRMIA — risk-manager certifications

Onboarded 2026-04-20 to validate the industry plugin pattern beyond the
project's first (energy) vertical. This file is 100% configuration —
no tool layer edits were needed to support it.
"""
from .base import IndustryBase, DeptEconomics, Intervention, CohortThresholds


FINANCE = IndustryBase(
    name="finance",
    display_name="Financial Services",
    description=(
        "Investment banking, equity research, risk & compliance, wealth "
        "management, retail banking, fintech, trading, and operations. "
        "Calibrated against BLS OES, Robert Half Salary Guide, SHRM, "
        "FINRA/SEC/CFP/CFA credential catalogs."
    ),

    # ---- Per-department economics ----
    # Base salary = median of Robert Half 2025 + BLS OES cross-ref.
    # fill_days reflects regulated-role hiring overhead (background checks,
    # licensing confirmation) — typically 30-50% longer than non-regulated.
    # replacement_multiplier captures book-of-business transfer risk in
    # client-facing roles (Wealth / IB) and regulatory-exam risk in
    # compliance-heavy roles.
    dept_economics={
        "InvestmentBanking": DeptEconomics(
            base_salary=185_000, fill_days=110, replacement_multiplier=2.2,
            safety_critical=False, hiring_difficulty="hard",
        ),
        "EquityResearch": DeptEconomics(
            base_salary=155_000, fill_days=95, replacement_multiplier=1.9,
            safety_critical=False, hiring_difficulty="hard",
        ),
        "Risk": DeptEconomics(
            base_salary=140_000, fill_days=85, replacement_multiplier=1.8,
            safety_critical=True, hiring_difficulty="hard",
        ),
        "Compliance": DeptEconomics(
            base_salary=130_000, fill_days=100, replacement_multiplier=1.9,
            safety_critical=True, hiring_difficulty="hard",
        ),
        "WealthManagement": DeptEconomics(
            base_salary=135_000, fill_days=90, replacement_multiplier=2.1,
            safety_critical=False, hiring_difficulty="moderate",
        ),
        "RetailBanking": DeptEconomics(
            base_salary=75_000, fill_days=55, replacement_multiplier=1.3,
            safety_critical=False, hiring_difficulty="easy",
        ),
        "Fintech": DeptEconomics(
            base_salary=160_000, fill_days=70, replacement_multiplier=1.6,
            safety_critical=False, hiring_difficulty="hard",
        ),
        "Operations": DeptEconomics(
            base_salary=95_000, fill_days=60, replacement_multiplier=1.4,
            safety_critical=False, hiring_difficulty="moderate",
        ),
    },

    # ---- Retention interventions ----
    # Re-uses 4 core levers from Energy + 2 FS-specific ones (equity
    # refresh, sabbatical). Causal annotations follow the same framework
    # (CAUSAL / MIXED / CORRELATIONAL / NONE) with FS-specific rationale.
    interventions={
        "comp_adjustment": Intervention(
            description="Off-cycle salary increase to close market gap; common after bonus cycle.",
            feature_affected="comp_ratio", effect_per_unit=0.09,
            cost_per_head_per_unit=15_000, lead_time_days=30,
            causal_status="MIXED",
            causal_rationale=(
                "Direct causal pathway (higher pay → stay), confounded by "
                "outside_options which is especially strong in FS where "
                "headhunters benchmark compensation actively."
            ),
        ),
        "equity_refresh": Intervention(
            description="Additional RSU / deferred cash grant vesting 3-4 years.",
            feature_affected="comp_ratio", effect_per_unit=0.07,
            cost_per_head_per_unit=25_000, lead_time_days=45,
            causal_status="MIXED",
            causal_rationale=(
                "Long-vesting equity creates golden-handcuff retention — "
                "direct causal mechanism. Reuses the comp_ratio confounded "
                "association, so simulator magnitude is approximate."
            ),
        ),
        "satisfaction_program": Intervention(
            description="Manager training + mental-health benefits + career pathing.",
            feature_affected="satisfaction", effect_per_unit=0.10,
            cost_per_head_per_unit=3_500, lead_time_days=90,
            causal_status="CORRELATIONAL",
            causal_rationale=(
                "Same selection confounding as in Energy — satisfied "
                "high-performers are the ones headhunted most. Simulator "
                "effect direction may be wrong; trust published FS retention "
                "benchmarks (Gartner, McKinsey) instead."
            ),
        ),
        "retention_bonus": Intervention(
            description="Stay-bonus with 12-24 month cliff vesting; paid cash.",
            feature_affected="comp_ratio", effect_per_unit=0.06,
            cost_per_head_per_unit=12_000, lead_time_days=14,
            causal_status="MIXED",
            causal_rationale=(
                "Direct mechanism during the vesting window. Magnitude "
                "approximate, direction correct."
            ),
        ),
        "sabbatical_program": Intervention(
            description="4-6 week paid sabbatical after 7+ years tenure.",
            feature_affected="satisfaction", effect_per_unit=0.05,
            cost_per_head_per_unit=4_000, lead_time_days=180,
            causal_status="CORRELATIONAL",
            causal_rationale=(
                "Applied through satisfaction channel — same confounding as "
                "satisfaction_program. Real-world ROI evidence exists "
                "(Deloitte, PwC) but simulator counterfactual is unreliable."
            ),
        ),
        "knowledge_capture": Intervention(
            description="Senior-to-junior shadowing + client-relationship transfer plans.",
            feature_affected=None, effect_per_unit=0.0,
            cost_per_head_per_unit=5_000, lead_time_days=60,
            reduces_knowledge_loss=True,
            causal_status="CAUSAL",
            causal_rationale=(
                "Reduces the COST of a departure (client-book transfer, "
                "institutional knowledge preservation) rather than the "
                "probability. In FS, losing a senior banker without book "
                "transfer can cost 3-5× their salary in client attrition."
            ),
        ),
    },

    # ---- Cohort thresholds ----
    # FS tenured_ip_years is lower than Energy (6.0 vs 8.0) because client
    # relationships + product knowledge mature faster in this vertical.
    # High-performer bar is slightly higher given the signal-to-noise
    # ratio of P&L-denominated performance reviews.
    cohort_thresholds=CohortThresholds(
        retirement_critical_age=62,   # regulated industry retires later
        retirement_emerging_age=58,
        new_hire_tenure_years=2.0,
        low_engagement_score=2.5,
        low_satisfaction_score=2.5,
        comp_gap_ratio=0.92,           # tighter — FS benchmarks aggressively
        tenured_ip_years=6.0,          # faster IP maturation
        high_performer_score=4.2,       # higher bar
    ),

    # ---- Cross-department adjacency ----
    # Which FS depts share transferable skills?
    #   - Risk ↔ Compliance: 0.80 (overlapping regulatory mental model)
    #   - EquityResearch ↔ InvestmentBanking: 0.60 (sector expertise)
    #   - WealthManagement ↔ RetailBanking: 0.55 (client-relationship skills)
    #   - Fintech ↔ Operations: 0.50 (systems + process mindset)
    dept_adjacency={
        "InvestmentBanking": {"EquityResearch": 0.60, "WealthManagement": 0.35, "Risk": 0.30},
        "EquityResearch":    {"InvestmentBanking": 0.60, "Risk": 0.40},
        "Risk":              {"Compliance": 0.80, "EquityResearch": 0.40, "Operations": 0.35},
        "Compliance":        {"Risk": 0.80, "Operations": 0.40},
        "WealthManagement":  {"RetailBanking": 0.55, "InvestmentBanking": 0.35},
        "RetailBanking":     {"WealthManagement": 0.55, "Operations": 0.45},
        "Fintech":           {"Operations": 0.50, "Risk": 0.30},
        "Operations":        {"Compliance": 0.40, "Fintech": 0.50, "RetailBanking": 0.45},
    },

    # ---- Feature labels ----
    # Identical to energy — the 8 Bi-LSTM features are domain-agnostic.
    # The one exception is the external labor market index: Energy uses
    # BLS energy index, Finance would ideally use BLS financial-activities
    # index (CES code ``55``). Keeping the same column name preserves the
    # saved Bi-LSTM model; label just gets re-captioned for the UI.
    feature_labels={
        "tenure_years":    "Tenure",
        "comp_ratio":      "Compensation ratio",
        "satisfaction":    "Job satisfaction",
        "performance":     "Performance",
        "engagement":      "Engagement",
        "age":             "Age / retirement pressure",
        "bls_energy_index": "External labor market (BLS financial-activities index)",
        "headcount":       "Department size trend",
    },

    # ---- Tier-1 employers ----
    # FS pedigree signal: bulge-bracket + big law-firm-adjacent +
    # top consultancies + top fintech. Lowercased for matching.
    tier_1_employers=[
        # Bulge bracket + major investment banks
        "goldman sachs", "goldman", "morgan stanley", "jpmorgan",
        "jp morgan", "jpmorgan chase", "bank of america", "citigroup",
        "citi", "barclays", "deutsche bank", "credit suisse", "ubs",
        "wells fargo", "hsbc", "bnp paribas", "societe generale",
        # Boutique + elite advisory
        "evercore", "lazard", "moelis", "centerview partners",
        "pjt partners", "rothschild", "perella weinberg", "guggenheim",
        # Asset management + alternatives
        "blackrock", "vanguard", "state street", "fidelity investments",
        "fidelity", "pimco", "t. rowe price", "invesco", "franklin templeton",
        "blackstone", "kkr", "carlyle", "apollo", "brookfield",
        "bridgewater", "two sigma", "de shaw", "citadel",
        "renaissance technologies", "millennium management",
        # Private equity + venture
        "tpg", "bain capital", "thoma bravo", "silver lake",
        "sequoia capital", "andreessen horowitz", "a16z",
        # Consulting with FS practice
        "mckinsey", "bain & company", "bain", "bcg",
        "boston consulting group", "deloitte", "pwc",
        "pricewaterhousecoopers", "ey", "ernst & young", "kpmg",
        "accenture",
        # Retail + wealth
        "edward jones", "raymond james", "charles schwab",
        "merrill lynch", "northern trust",
        # Fintech + payments + crypto
        "stripe", "plaid", "robinhood", "coinbase", "chime",
        "affirm", "block", "square", "visa", "mastercard",
        "american express",
        # Exchanges + market infrastructure
        "nasdaq", "nyse", "cme group", "ice", "intercontinental exchange",
        "lseg", "london stock exchange",
        # Agencies / regulators (when candidate worked there)
        "federal reserve", "sec", "ffr", "finra", "cfpb", "occ",
    ],

    # ---- Critical certifications ----
    # FS credentials HR screens for actively. Many are role-gating:
    # Series 7 for broker-dealer reps, CFA for research analysts,
    # CFP for financial planners, FRM for risk managers, CPA for
    # accounting leadership.
    critical_certifications=[
        # Securities licenses
        "Series 7", "Series 24", "Series 63", "Series 65", "Series 66",
        "Series 79", "Series 86", "Series 87", "Series 3", "Series 31",
        # Investment analysis + portfolio management
        "CFA", "CFA Level I", "CFA Level II", "CFA Level III",
        "CAIA", "CMT",
        # Financial planning
        "CFP", "ChFC", "CLU", "RICP", "PFS",
        # Risk management
        "FRM", "PRM", "ERP",
        # Accounting / audit
        "CPA", "CMA", "CIA", "CFE", "CISA",
        # Compliance + AML
        "CAMS", "CRCM", "NCCO",
        # Insurance
        "Series 6", "Series 51", "Series 52", "Series 53",
        # Project + program management (common leadership adjacency)
        "PMP", "PRINCE2",
    ],

    # ---- Taxonomy sources ----
    taxonomy_source_refs=[
        "BLS OES (13-2052 / 13-2031 / 13-2011 / 11-3031)",
        "Robert Half Salary Guide 2025",
        "CFA Institute Body of Knowledge",
        "CFP Board Certification Pathway",
        "GARP FRM Curriculum",
        "PRMIA PRM Handbook",
        "FINRA Exam Registration Matrix",
        "SEC + NFA + OCC regulatory taxonomies",
        "ESCO 2421 Management & Organization Analysts",
        "SHRM Benchmarking (time-to-fill, regulated roles)",
        "McKinsey FS Practice Reports",
        "Deloitte Human Capital Trends (FS cut)",
    ],

    # ---- O*NET crosswalk ----
    # Each FS dept maps to its canonical SOC — unlocks O*NET auto-enrichment
    # of role descriptions, skills, and tasks when the v7 NER pipeline is
    # expanded to the FS vertical.
    onet_crosswalk_codes={
        "InvestmentBanking": "13-2051",   # Financial and Investment Analysts
        "EquityResearch":    "13-2051",
        "Risk":              "13-2054",   # Financial Risk Specialists
        "Compliance":        "13-1041",   # Compliance Officers
        "WealthManagement":  "13-2052",   # Personal Financial Advisors
        "RetailBanking":     "13-2071",   # Credit Counselors
        "Fintech":           "15-1251",   # Computer Programmers (fintech lean)
        "Operations":        "13-2099",   # Financial Specialists, All Other
    },
    # ---- Synthetic workforce-data generator inputs ----
    # Total = 305 employees; roughly matches Energy's 300 for comparable
    # Bi-LSTM signal strength. Distribution reflects typical FS firm shape
    # (operations heavy, IB narrow at the top).
    dept_sizes={
        "InvestmentBanking": 35, "EquityResearch": 25, "Risk": 40,
        "Compliance": 35, "WealthManagement": 50, "RetailBanking": 60,
        "Fintech": 30, "Operations": 30,
    },
    roles_by_dept={
        "InvestmentBanking": ["Analyst", "Associate", "Vice President",
                               "Director", "Managing Director"],
        "EquityResearch":    ["Equity Research Analyst", "Senior Analyst",
                               "Research Associate", "Sector Head"],
        "Risk":              ["Credit Risk Analyst", "Market Risk Manager",
                               "Operational Risk Specialist",
                               "Model Validation Analyst"],
        "Compliance":        ["Compliance Officer", "AML Analyst",
                               "Regulatory Affairs Manager",
                               "Chief Compliance Officer"],
        "WealthManagement":  ["Financial Advisor", "Wealth Manager",
                               "Private Banker", "Portfolio Manager",
                               "Client Service Associate"],
        "RetailBanking":     ["Personal Banker", "Branch Manager",
                               "Loan Officer", "Teller", "Relationship Manager"],
        "Fintech":           ["Quantitative Developer", "Data Engineer",
                               "Platform Engineer", "Trading Systems Engineer",
                               "Payments Engineer"],
        "Operations":        ["Operations Analyst", "Settlement Specialist",
                               "Middle Office Analyst", "Reconciliation Analyst"],
    },
    # Finance's data lives at data/finance/*.csv — keeps Energy's data/
    # root-level files untouched for byte-identity preservation.
    data_subdir="finance",
)
