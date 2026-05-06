"""
v7 NER data generator — O*NET-grounded, 10-type BIO tagging.

Entity types (v7 schema, see ENTITY_SCHEMA_V7.md):
  Tier 1: SKILL, TOOL, CERT, DEGREE, EMPLOYER, YEARS_EXP
  Tier 2: INDUSTRY, LOCATION
  Tier 3: PROJECT, SOFT_SKILL

Per-sample generation seeds from one of 22 energy-sector SOC codes:
  technology_skills          → TOOL spans (HYSYS, Aspen Plus, AutoCAD, ...)
  knowledge / DWAs / taxonomy → SKILL pool (abstract techniques)
  abilities / work_styles     → SOFT_SKILL pool
  professional_associations   → EMPLOYER supplement
  SOC_TO_INDUSTRIES           → INDUSTRY spans
  SOC_TO_LOCATIONS            → LOCATION spans
  SOC_TO_PROJECTS             → PROJECT spans

Output: data/processed/ner_training_v7.json (synthetic-only, ~12k samples).
The v6 file data/processed/ner_training_expanded.json is NOT overwritten.
The v6 generator data/generate_ner_data.py is preserved as the fallback.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.taxonomy.energy_taxonomy_v2 import (
    CERTIFICATIONS as V2_CERTS,
    DEGREES as V2_DEGREES,
    EMPLOYERS as V2_EMPLOYERS,
)
from data.scrapers.onet_scraper import ONetScraper


# v7 YEARS expressions — each phrase stands alone (no dangling "working in"
# or trailing preposition), so template composition can't produce broken
# grammar like "22 years working in of hands-on ...".
V7_YEARS_EXPRESSIONS = [
    "{n} years of experience",
    "{n}+ years of experience",
    "{n} years of hands-on experience",
    "over {n} years",
    "more than {n} years",
    "{n}-year track record",
    "{n} years in the industry",
    "{n} years of industry experience",
    "nearly {n} years of experience",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
RAW_ONET_DIR = BASE / "raw" / "onet"
PROC_DIR = BASE / "processed"
DEFAULT_OUTPUT = PROC_DIR / "ner_training_v7.json"


# ---------------------------------------------------------------------------
# Canonical TOOL list. v7 treats these as TOOL, NEVER SKILL. Seeded from
# O*NET technology_skills across the 22 energy SOCs + explicit vendor
# products common in energy resumes.
# ---------------------------------------------------------------------------

TOOLS_CANONICAL: List[str] = [
    # Process simulators
    "HYSYS", "Aspen Plus", "Aspen HYSYS", "Pipesim", "PHAST", "OLGA",
    "Petrel", "Eclipse reservoir simulator", "CMG STARS", "CMG IMEX",
    "ProMax", "UniSim", "PRO/II", "gPROMS",
    # CAD / drafting
    "AutoCAD", "SolidWorks", "SmartPlant 3D", "Navisworks", "MicroStation",
    "AutoCAD Civil 3D", "Autodesk Inventor", "Bentley OpenPlant",
    "PDMS", "Revit",
    # Electrical / power
    "ETAP", "SKM PowerTools", "DIgSILENT PowerFactory", "EMTP",
    "CYME", "PSCAD", "Synergi Electric",
    # Mechanical FEA
    "ANSYS", "ANSYS Fluent", "Abaqus", "NASTRAN",
    # Reservoir / geo
    "Kingdom Suite", "IHS Petra", "IHS Harmony",
    "Landmark OpenWorks", "Schlumberger Techlog", "Kappa Workstation",
    # Automation / control
    "Rockwell Studio 5000", "Rockwell RSLogix 5000", "Siemens TIA Portal",
    "Honeywell Experion", "Yokogawa CENTUM", "DeltaV",
    "Allen-Bradley ControlLogix",
    # PLM / PM / ERP
    "Primavera P6", "Microsoft Project", "SAP", "SAP PM", "Maximo",
    "JD Edwards", "SAP S/4HANA", "Oracle EBS", "Oracle Primavera",
    # Data / analytics / historian
    "OSIsoft PI", "PI System", "Power BI", "Tableau", "Spotfire",
    # Programming / scripting
    "Python", "R programming", "MATLAB", "Simulink", "LabVIEW",
    "C++", "SQL", "VBA",
    # Wind-specific
    "WindPRO", "WAsP", "WindFarmer", "Bladed",
    # Solar-specific
    "PVsyst", "SAM solar simulation", "Helioscope",
    # Corrosion
    "OLI Studio",
    # GIS
    "ArcGIS", "QGIS",
    # SCADA / HMI
    "Wonderware", "Ignition SCADA", "FactoryTalk View",
    # ABB automation (v7 expansion 2026-04-19)
    "ABB 800xA", "System 800xA", "Freelance DCS", "AC 800M",
    "Symphony Plus", "RELION", "REX 640", "ABB Ability",
    # Emerson automation (v7 expansion)
    "Ovation", "Safety Manager SC", "AMS Device Manager",
    "AMS Machinery Manager", "PlantWeb", "Plantweb Optics",
    "Forge APM", "Fisher", "Rosemount",
    # Schneider Electric automation (v7 expansion)
    "Modicon M580", "Modicon M340", "Foxboro DCS", "Foxboro Evo",
    "Avantis APM", "Triconex", "ClearSCADA", "Citect SCADA",
    "EcoStruxure",
    # Rockwell Automation (v7 expansion)
    "PlantPAx", "CompactLogix", "FactoryTalk Historian",
    "Emulate3D", "Kepware",
    # Honeywell Process Solutions (v7 expansion)
    "Experion PKS", "TDC3000", "UniSim Design", "UniSim Operations",
    "Honeywell Forge", "OneWireless", "QCS",
    # Yokogawa (v7 expansion)
    "CENTUM VP", "CENTUM CS 3000", "ProSafe-RS", "STARDOM FCN",
    "FAST/TOOLS", "Exaquantum", "OpreX",
    # Additional plant-info / process-historian / asset-management products
    # (v7 pre-launch top-up to 49 vendor entries)
    "Bently Nevada System 1", "AVEVA System Platform",
    "OSIsoft AF", "Aspen Process Explorer",
]


# ---------------------------------------------------------------------------
# SOC → INDUSTRY mapping. At least one, up to three industries per SOC.
# ---------------------------------------------------------------------------

SOC_TO_INDUSTRIES: Dict[str, List[str]] = {
    "17-2041.00": ["refining", "petrochemicals", "LNG"],
    "17-2071.00": ["T&D utility", "power distribution", "renewable energy"],
    "17-2111.00": ["upstream oil and gas", "refining", "petrochemicals"],
    "17-2141.00": ["upstream oil and gas", "midstream pipelines", "LNG"],
    "17-2151.00": ["upstream oil and gas", "mining", "offshore drilling"],
    "17-2161.00": ["nuclear power"],
    "17-2171.00": ["upstream oil and gas", "offshore drilling"],
    "17-2199.03": ["energy efficiency", "CCUS", "power generation"],
    "17-2199.10": ["offshore wind", "onshore wind"],
    "17-2199.11": ["solar PV", "battery storage"],
    "11-3051.00": ["refining", "petrochemicals", "power generation"],
    "11-9041.00": ["upstream oil and gas", "power generation", "LNG"],
    "51-8011.00": ["nuclear power"],
    "51-8012.00": ["T&D utility", "power distribution"],
    "51-8013.00": ["power generation"],
    "51-8021.00": ["refining", "petrochemicals"],
    "47-5013.00": ["upstream oil and gas", "offshore drilling"],
    "47-5071.00": ["upstream oil and gas", "offshore drilling"],
    "49-9041.00": ["refining", "power generation", "petrochemicals"],
    "49-9043.00": ["refining", "power generation"],
    "49-9081.00": ["offshore wind", "onshore wind"],
    "47-2231.00": ["solar PV"],
    # v7 expansion — automation / instrumentation SOCs
    "17-2112.00": ["industrial automation", "process manufacturing", "petrochemicals"],
    "17-3023.00": ["T&D utility", "power distribution", "power generation"],
    "17-3024.00": ["industrial automation", "process manufacturing"],
    "49-9012.00": ["midstream pipelines", "refining", "petrochemicals"],
    "49-9069.00": ["refining", "petrochemicals", "power generation"],
}


# ---------------------------------------------------------------------------
# SOC → LOCATION mapping. Professional venues where this SOC typically works.
# ---------------------------------------------------------------------------

SOC_TO_LOCATIONS: Dict[str, List[str]] = {
    "17-2041.00": ["Gulf Coast", "Houston", "Baton Rouge"],
    "17-2071.00": ["ERCOT grid", "PJM grid", "MISO grid"],
    "17-2111.00": ["Gulf of Mexico", "Permian Basin", "North Sea"],
    "17-2141.00": ["Gulf of Mexico", "Houston", "Abu Dhabi"],
    "17-2151.00": ["Permian Basin", "Bakken", "Alberta oil sands"],
    "17-2161.00": ["Illinois", "South Carolina", "Georgia"],
    "17-2171.00": [
        "Permian Basin", "Bakken", "Eagle Ford", "Gulf of Mexico",
        "Haynesville", "North Sea", "Barnett Shale", "offshore Brazil",
        "Abu Dhabi",
    ],
    "17-2199.03": ["Houston", "California", "Texas"],
    "17-2199.10": ["West Texas", "North Sea", "Dogger Bank"],
    "17-2199.11": ["Arizona", "West Texas", "California"],
    "11-3051.00": ["Gulf Coast", "Houston", "Baton Rouge"],
    "11-9041.00": ["Houston", "Calgary", "Abu Dhabi"],
    "51-8011.00": ["Georgia", "South Carolina", "Illinois"],
    "51-8012.00": ["ERCOT grid", "CAISO grid"],
    "51-8013.00": ["ERCOT grid", "PJM grid"],
    "51-8021.00": ["Gulf Coast", "Midwest"],
    "47-5013.00": ["Permian Basin", "Bakken", "Eagle Ford", "Gulf of Mexico"],
    "47-5071.00": ["Permian Basin", "Bakken", "Eagle Ford", "Gulf of Mexico"],
    "49-9041.00": ["Gulf Coast", "Midwest"],
    "49-9043.00": ["Gulf Coast"],
    "49-9081.00": ["West Texas", "Iowa", "North Sea"],
    "47-2231.00": ["Arizona", "West Texas", "California"],
    # v7 expansion — automation / instrumentation SOCs
    "17-2112.00": ["Gulf Coast", "Houston", "Midwest"],
    "17-3023.00": ["ERCOT grid", "PJM grid", "MISO grid"],
    "17-3024.00": ["Gulf Coast", "Houston", "Midwest"],
    "49-9012.00": ["Gulf Coast", "Permian Basin", "Houston"],
    "49-9069.00": ["Gulf Coast", "Midwest"],
}


# ---------------------------------------------------------------------------
# SOC → PROJECT mapping. Named facilities / megaprojects. SOCs without
# well-known named projects (operators, roustabouts, mechanics) fall back
# to the global ENERGY_PROJECTS_VOCAB list.
# ---------------------------------------------------------------------------

SOC_TO_PROJECTS: Dict[str, List[str]] = {
    "17-2041.00": [
        "Sabine Pass LNG", "Cameron LNG", "Freeport LNG",
        "Golden Pass LNG", "Port Arthur LNG",
    ],
    "17-2071.00": ["Vogtle Unit 3", "Coastal Virginia Offshore Wind"],
    "17-2111.00": [
        "Thunder Horse platform", "Mad Dog platform", "Appomattox",
    ],
    "17-2141.00": [
        "Gorgon LNG", "Wheatstone LNG", "Train 4 LNG Qatar",
    ],
    "17-2151.00": ["Kearl Oil Sands", "Olympic Dam"],
    "17-2161.00": [
        "Vogtle Unit 3", "Hinkley Point C", "Olkiluoto 3", "Flamanville 3",
    ],
    "17-2171.00": [
        "Thunder Horse platform", "Perdido Spar", "Mad Dog platform",
        "Stones FPSO", "Anchor project", "Appomattox", "Tiber prospect",
    ],
    "17-2199.03": ["Quest CCS", "Gorgon CO2 Injection"],
    "17-2199.10": [
        "Dogger Bank Wind Farm", "Vineyard Wind", "Hornsea Two",
        "Revolution Wind", "South Fork Wind",
    ],
    "17-2199.11": ["Gemini Solar Project", "Copper Mountain Solar"],
    "11-3051.00": ["Sabine Pass LNG", "Port Arthur LNG"],
    "11-9041.00": ["Kashagan field", "Train 7 LNG Qatar"],
    "51-8011.00": ["Vogtle Unit 3", "Vogtle Unit 4"],
    "51-8012.00": [],
    "51-8013.00": [],
    "51-8021.00": [],
    "47-5013.00": [],
    "47-5071.00": [],
    "49-9041.00": [],
    "49-9043.00": [],
    "49-9081.00": ["Dogger Bank Wind Farm"],
    "47-2231.00": ["Gemini Solar Project", "Copper Mountain Solar"],
    # v7 expansion — automation / instrumentation SOCs (no named megaprojects)
    "17-2112.00": [],
    "17-3023.00": [],
    "17-3024.00": [],
    "49-9012.00": [],
    "49-9069.00": [],
}


# ---------------------------------------------------------------------------
# SOC → ROLE TITLE mapping. Used in preambles.
# ---------------------------------------------------------------------------

SOC_TO_ROLES: Dict[str, List[str]] = {
    "17-2041.00": [
        "Senior Process Engineer", "Lead Chemical Engineer",
        "Process Engineering Manager",
    ],
    "17-2071.00": [
        "Senior Electrical Engineer", "Lead Power Systems Engineer",
        "Substation Design Engineer",
    ],
    "17-2111.00": [
        "Senior HSE Engineer", "Lead Health and Safety Specialist",
        "Process Safety Engineer",
    ],
    "17-2141.00": [
        "Senior Mechanical Engineer", "Lead Machinery Engineer",
        "Rotating Equipment Specialist",
    ],
    "17-2151.00": [
        "Senior Mining Engineer", "Lead Geological Engineer",
        "Exploration Geologist",
    ],
    "17-2161.00": [
        "Senior Nuclear Engineer", "Lead Reactor Engineer",
        "Reactor Design Engineer",
    ],
    "17-2171.00": [
        "Senior Petroleum Engineer", "Reservoir Engineering Manager",
        "Drilling Engineer", "Production Engineer",
    ],
    "17-2199.03": [
        "Senior Energy Engineer", "Energy Systems Specialist",
        "Decarbonization Lead",
    ],
    "17-2199.10": [
        "Senior Wind Energy Engineer", "Wind Turbine Design Engineer",
        "Offshore Wind Package Manager",
    ],
    "17-2199.11": [
        "Senior Solar Energy Systems Engineer", "Solar PV Design Engineer",
        "PV Systems Integration Engineer",
    ],
    "11-3051.00": [
        "Plant Manager", "Industrial Production Manager",
        "Operations Director",
    ],
    "11-9041.00": [
        "VP of Engineering", "Engineering Manager",
        "Director of Engineering",
    ],
    "51-8011.00": [
        "Senior Reactor Operator", "Shift Technical Advisor",
        "Reactor Operator",
    ],
    "51-8012.00": [
        "Power Dispatcher", "System Operator", "Grid Operator",
    ],
    "51-8013.00": [
        "Power Plant Operator", "Control Room Operator", "Plant Operator",
    ],
    "51-8021.00": [
        "Stationary Engineer", "Boiler Operator", "Utility Operator",
    ],
    "47-5013.00": [
        "Service Unit Supervisor", "Field Service Operator",
        "Oilfield Service Foreman",
    ],
    "47-5071.00": ["Roustabout", "Rig Hand", "Oil Field Worker"],
    "49-9041.00": [
        "Industrial Maintenance Technician", "Millwright",
        "Rotating Equipment Mechanic",
    ],
    "49-9043.00": ["Maintenance Technician", "Industrial Mechanic"],
    "49-9081.00": [
        "Wind Turbine Service Technician", "Senior Wind Technician",
        "WTG Field Technician",
    ],
    "47-2231.00": [
        "Solar PV Installer", "Senior Solar Technician",
        "PV Field Foreman",
    ],
    # v7 expansion — automation / instrumentation SOCs
    "17-2112.00": [
        "Senior Industrial Engineer", "Lead Process Improvement Engineer",
        "Operations Excellence Manager",
    ],
    "17-3023.00": [
        "Electrical Engineering Technician", "Senior Electrical Technician",
        "Substation Test Technician",
    ],
    "17-3024.00": [
        "Mechatronics Engineer", "Electro-Mechanical Technologist",
        "Automation Systems Specialist",
    ],
    "49-9012.00": [
        "Control Systems Technician", "Valve Installer",
        "Field Instrumentation Technician",
    ],
    "49-9069.00": [
        "Instrumentation Technician", "Precision Instrument Repairer",
        "Senior Calibration Technician",
    ],
}


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Pure interpersonal / leadership soft skills (NOT technical).
SOFT_SKILLS_VOCAB: List[str] = [
    "cross-functional leadership", "stakeholder management",
    "stakeholder alignment",  # hand-labeled R006/R017 variant
    "mentoring", "change management", "conflict resolution",
    "team building", "executive communication",
    "decision-making under pressure", "strategic thinking",
    "coaching", "team leadership", "vendor management",
    "negotiation", "cross-cultural communication",
    "consensus building", "presentation skills",
    "emotional intelligence", "conflict management",
    "influence without authority", "active listening",
]


# Abstract SKILL seeds (technical capabilities / methodologies / techniques,
# NOT TOOLs and NOT SOFT_SKILLs).
SKILL_SEEDS: List[str] = [
    "process safety management", "HAZOP facilitation", "SCADA integration",
    "reservoir simulation", "well completion design", "P&ID review",
    "turnaround execution", "FEED studies", "fatigue analysis",
    "pressure vessel inspection", "piping stress analysis",
    "fire protection engineering", "environmental impact assessment",
    "cathodic protection", "pipeline integrity management",
    "inline inspection", "leak detection", "corrosion engineering",
    "flow assurance", "production optimization",
    "drilling optimization", "artificial lift design",
    "LNG liquefaction", "gas dehydration", "amine treating",
    "sulfur recovery operations",
    "relay protection", "arc flash analysis", "load flow studies",
    "substation design", "distribution system planning",
    "switchgear maintenance", "transformer testing",
    "wind resource assessment", "wind turbine commissioning",
    "solar farm design", "solar inverter commissioning",
    "BESS commissioning", "grid interconnection studies",
    "reactor design", "radiation protection",
    "functional safety assessment",
    "emergency response planning", "process hazard analysis",
    "root cause failure analysis", "reliability centered maintenance",
    "predictive maintenance", "vibration analysis",
    "machine learning", "predictive modeling",
    "advanced process control", "model predictive control",
    "DCS programming", "PLC programming", "HMI development",
    "control loop tuning", "fieldbus configuration",
    "commissioning and startup", "electrical commissioning",
    "direct air capture operations", "carbon capture",
    "lifecycle assessment", "GHG inventory accounting",
    "ammonia synthesis", "methanol synthesis",
    "PEM electrolyzer operations",
]


# Cross-role fallback vocabularies.
ENERGY_INDUSTRIES_VOCAB: List[str] = [
    "upstream oil and gas", "offshore drilling", "LNG",
    "refining", "petrochemicals", "midstream pipelines",
    "CCUS", "hydrogen", "ammonia",
    "solar PV", "onshore wind", "offshore wind",
    "nuclear power", "T&D utility", "power distribution",
    "power generation", "battery storage", "geothermal",
    "energy efficiency", "renewable energy",
    "fuel cells",
    # v7 expansion — automation/instrumentation contexts
    "industrial automation", "process manufacturing",
]


# ---------------------------------------------------------------------------
# v7 automation standards — sourced from primary peer-reviewed standards
# bodies (ISA / IEC / NERC / NAMUR / IEEE / NIST). Treated as CERT entities
# per the v7 schema convention (already established for "API 570", which is
# also a standard rather than a credential per se). Standards routinely
# appear in resumes as proof of competency / project compliance ("designed
# safety system per IEC 61508", "implemented HAZOP per ISA 84").
# Added 2026-04-19 to broaden CERT recall on automation-vendor resumes.
# ---------------------------------------------------------------------------

CERTS_V7_AUTOMATION_STANDARDS: List[str] = [
    # ISA — process automation, alarm management, MES/ERP, OT cyber, HMI
    "ISA 5.1", "ISA 18.2", "ISA 84", "ISA 95", "ISA 99", "ISA 101",
    # IEC — functional safety, substation automation, PLC programming,
    #       ICS cyber, short-circuit, telecontrol
    "IEC 61508", "IEC 61511", "IEC 61850", "IEC 61131-3",
    "IEC 62443", "IEC 60909", "IEC 60870",
    # NERC CIP — North American grid cybersecurity (CIP-002 through CIP-014)
    "NERC CIP-002", "NERC CIP-003", "NERC CIP-004",
    "NERC CIP-005", "NERC CIP-006", "NERC CIP-007",
    "NERC CIP-008", "NERC CIP-009", "NERC CIP-010",
    "NERC CIP-011", "NERC CIP-012", "NERC CIP-013", "NERC CIP-014",
    # NAMUR — European process-automation user-association recommendations
    "NAMUR NE 43", "NAMUR NE 107", "NAMUR NE 163",
    # IEEE — PTP synchronization, relay protection, harmonics, DER interconnect
    "IEEE 1588", "IEEE C37.90", "IEEE C37.118",
    "IEEE 519", "IEEE 1547",
    # NIST — Industrial Control Systems Security
    "NIST SP 800-82",
]


ENERGY_LOCATIONS_VOCAB: List[str] = [
    "Gulf of Mexico", "Permian Basin", "Bakken", "Eagle Ford",
    "Haynesville", "Barnett Shale", "Alberta oil sands",
    "North Sea", "Dogger Bank", "Hornsea",
    "offshore Brazil", "Abu Dhabi", "West Africa",
    "Arabian Gulf",  # hand-labeled R014 (Saudi Aramco)
    "West Texas", "Gulf Coast", "Houston",
    "Arizona", "California",
]

ENERGY_PROJECTS_VOCAB: List[str] = [
    "Sabine Pass LNG", "Cameron LNG", "Gorgon LNG",
    "Train 7 LNG Qatar", "Hinkley Point C", "Vogtle Unit 3",
    "Thunder Horse platform", "Perdido Spar", "Mad Dog platform",
    "Dogger Bank Wind Farm", "Vineyard Wind", "Hornsea Two",
    "Gemini Solar Project", "Quest CCS",
]


# ---------------------------------------------------------------------------
# TIER 0 aux-taxonomy loader (added 2026-04-19)
#
# Wires 5 peer-curated energy-domain taxonomies already on disk into the v7
# *synthetic* generator's pools. Each was scraped by a dedicated module in
# data/scrapers/ but never consumed by the v7 generator.
#
# Sources (all on disk):
#   data/raw/cewd/flat.json   — Center for Energy Workforce Development
#                               5-tier competency model + named credentials
#   data/raw/spe/flat.json    — Society of Petroleum Engineers
#                               drilling / production / reservoir / completions
#                               / well-integrity disciplines
#   data/raw/iadc/flat.json   — Int'l Assoc of Drilling Contractors
#                               WellCAP / WellSharp + well-control techniques
#   data/raw/irena/flat.json  — Int'l Renewable Energy Agency
#                               solar / wind / hydropower / geothermal /
#                               battery storage / cross-cutting integration
#   data/raw/asme/flat.json   — American Society of Mechanical Engineers
#                               BPVC sections / B31 piping / rotating equipment
#                               / materials & welding / NDE
#
# IMPORTANT: aux pools are only used by the synthetic generator. They are
# NOT wired into _reannotate_v6_sample so the held-out eval set, hand-labeled
# resumes, Mehyaar IT CVs, and Dataturks IT resumes don't get auto-tagged
# with potentially over-broad spans. Re-annotation stays conservative
# (TOOLS_CANONICAL + CERTS_V7_AUTOMATION_STANDARDS only).
# ---------------------------------------------------------------------------

# Drop-list — entries too generic to label confidently. CEWD Tier-2
# academic competencies and business fundamentals would false-positive
# in any English text and are skipped at ingest.
_AUX_DROPLIST = {
    "reading comprehension", "writing", "mathematics",
    "science and engineering technology",
    "information literacy", "fundamental it skills",
    "business fundamentals",
    "working with tools and technology",
}

# Exact-match allowlist for SOFT_SKILL routing — manually curated from
# CEWD Tier 1-3 personal-effectiveness / workplace-competency entries.
# Exact-match (not substring) to avoid catching technical SKILL spans like
# "casing integrity testing" via a substring like "integrity".
_AUX_SOFT_SKILL_EXACT = {
    "interpersonal skills", "integrity", "professionalism", "initiative",
    "adaptability and flexibility", "dependability", "lifelong learning",
    "willingness to learn", "honesty",
    "communication", "active listening",
    "critical and analytical thinking",
    "teamwork", "customer focus",
    "planning and organizing", "creative thinking",
    "problem solving and decision making",
    "checking and monitoring", "scheduling and coordinating",
    "strategic planning", "change management", "continuous improvement",
    "performance management", "team leadership",
    "organizational development", "leadership",
    "stakeholder alignment", "vendor management",
    "talent development", "succession planning",
    "ethics", "professional ethics",
}

# Spans routed to CERT — named credentials, standards, codes.
_AUX_CERT_TOKEN_HINTS = ("WellCAP", "WellSharp", "BPVC", "B31.")
_AUX_CERT_PHRASE_HINTS = (
    "certification", "certified", "certificate",
    "license", "licensed", "licensure",
)
_AUX_CERT_PREFIX_RX = re.compile(
    r"^(API|ASME|ISA|IEC|IEEE|NERC|NACE|NAMUR|NIST|OSHA|GWO|NEBOSH|"
    r"AWS|IADC|ASNT|AWWA)\s+"
)


def _classify_aux_span(span: str) -> Optional[str]:
    """Route an aux-taxonomy span to SKILL / CERT / SOFT_SKILL.

    Returns None to drop entries that are too generic for confident labeling.
    """
    s = span.strip()
    sl = s.lower()
    if sl in _AUX_DROPLIST:
        return None
    if sl in _AUX_SOFT_SKILL_EXACT:
        return "SOFT_SKILL"
    if any(t in s for t in _AUX_CERT_TOKEN_HINTS):
        return "CERT"
    if any(p in sl for p in _AUX_CERT_PHRASE_HINTS):
        return "CERT"
    if _AUX_CERT_PREFIX_RX.match(s):
        return "CERT"
    return "SKILL"


def _load_aux_taxonomy_pools() -> Tuple[List[str], List[str], List[str]]:
    """Load (skills, certs, soft_skills) from the 5 on-disk taxonomies.

    Returns deduped, length-filtered, sorted lists. Missing source files
    are skipped silently with a single-line stderr-style note.
    """
    aux_raw = BASE / "raw"
    skills: set = set()
    certs: set = set()
    soft: set = set()

    def _add(span: str) -> None:
        s = (span or "").strip()
        if not (3 <= len(s) <= 80):
            return
        cls = _classify_aux_span(s)
        if cls is None:
            return  # too-generic entry dropped
        if cls == "SOFT_SKILL":
            soft.add(s)
        elif cls == "CERT":
            certs.add(s)
        else:
            skills.add(s)

    counters: Dict[str, int] = {}

    # CEWD — dict with explicit 'skills' + 'certifications' keys
    cewd_path = aux_raw / "cewd" / "flat.json"
    if cewd_path.exists():
        try:
            d = json.loads(cewd_path.read_text())
            for s in d.get("skills", []) or []:
                if isinstance(s, str):
                    _add(s)
            for c in d.get("certifications", []) or []:
                if isinstance(c, str):
                    cs = c.strip()
                    if 3 <= len(cs) <= 80:
                        certs.add(cs)
            counters["cewd"] = (
                len(d.get("skills", []) or [])
                + len(d.get("certifications", []) or [])
            )
        except Exception as e:
            print(f"[v7gen] aux: cewd skipped ({e})")

    # SPE / IADC / IRENA / ASME — list of {skill, ...}
    for src_name, rel in [
        ("spe",   "spe/flat.json"),
        ("iadc",  "iadc/flat.json"),
        ("irena", "irena/flat.json"),
        ("asme",  "asme/flat.json"),
    ]:
        path = aux_raw / rel
        if not path.exists():
            continue
        try:
            d = json.loads(path.read_text())
            if not isinstance(d, list):
                continue
            n = 0
            for item in d:
                if not isinstance(item, dict):
                    continue
                span = (item.get("skill") or item.get("name") or "").strip()
                if span:
                    _add(span)
                    n += 1
            counters[src_name] = n
        except Exception as e:
            print(f"[v7gen] aux: {src_name} skipped ({e})")

    print(
        f"[v7gen] aux taxonomy loader: source counts {counters}, "
        f"out skills={len(skills)} certs={len(certs)} soft={len(soft)}"
    )
    return sorted(skills), sorted(certs), sorted(soft)


(
    AUX_SKILLS_FROM_TAXONOMIES,
    AUX_CERTS_FROM_TAXONOMIES,
    AUX_SOFT_SKILLS_FROM_TAXONOMIES,
) = _load_aux_taxonomy_pools()


# Canonical energy-sector employer list. This is the ONLY pool used by the
# generator — v2 taxonomy EMPLOYERS is discarded because USAJOBS contributed
# hundreds of non-energy federal agencies, BPO / outsourcing firms, and
# other noise that polluted the pool.
EMPLOYER_ALLOWLIST: List[str] = [
    # IOCs + NOCs
    "Chevron", "ExxonMobil", "Shell", "BP", "ConocoPhillips",
    "TotalEnergies", "Saudi Aramco", "ADNOC", "Petrobras", "Equinor",
    "Repsol", "Eni", "Pemex", "Petronas", "Sinopec", "CNPC",
    # Oilfield services / EPC
    "Baker Hughes", "Halliburton", "Schlumberger", "TechnipFMC",
    "Saipem", "Wood PLC", "KBR", "Bechtel", "Fluor Corporation",
    "Jacobs Engineering", "McDermott International",
    "Worley", "AECOM Energy", "Aker Solutions", "NOV",
    # Midstream / integrated
    "Williams Companies", "Enterprise Products Partners", "Kinder Morgan",
    "Enbridge", "Cheniere Energy", "Targa Resources",
    "Plains All American",
    # Utilities / power
    "NextEra Energy", "Duke Energy", "Southern Company",
    "Dominion Energy", "Exelon", "Sempra Energy", "AES Corporation",
    "Entergy", "American Electric Power", "PG&E",
    "Xcel Energy", "DTE Energy", "FirstEnergy", "CenterPoint Energy",
    # OEMs / tech / automation
    "Siemens Energy", "GE Vernova", "Emerson Electric",
    "Honeywell UOP", "ABB Energy Industries", "Schneider Electric",
    "Rockwell Automation", "Yokogawa",
    # Renewables / storage / EV
    "Vestas", "Orsted", "First Solar", "Enphase Energy",
    "Sunrun", "SunPower", "Form Energy", "ChargePoint", "Tesla",
    # Nuclear
    "NuScale Power", "Westinghouse Electric", "Framatome", "TerraPower",
    "Kairos Power",
    # Decarbonization / clean fuels
    "Climeworks", "Heirloom Carbon", "Monolith",
    "Fervo Energy", "Twelve",
    # Construction / craft labor
    "Zachry Group", "Kiewit", "Bilfinger",
    # Industry bodies / safety (as EMPLOYER when candidate worked there)
    "Bureau of Safety and Environmental Enforcement",
    "Nuclear Regulatory Commission",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GENERIC_TOOL_TAIL = {
    "software", "system", "systems", "program", "programs",
    "suite", "platform", "application", "applications",
}


def _clean_tool_name(raw: str) -> Optional[str]:
    """Filter raw O*NET technology_skills entries. Drop category-level
    junk; keep vendor product names. Returns None if the entry is noise."""
    raw = (raw or "").strip()
    if len(raw) < 2 or len(raw) > 60:
        return None
    low = raw.lower()
    if any(bad in low for bad in (" see ", "various", "other ", "n/a")):
        return None
    last = raw.split()[-1].lower()
    if last in _GENERIC_TOOL_TAIL:
        return None
    return raw


def _build_employer_pool() -> List[str]:
    """Use the curated EMPLOYER_ALLOWLIST exclusively. See the comment on
    EMPLOYER_ALLOWLIST for why v2 taxonomy employers are not mixed in."""
    return sorted(set(EMPLOYER_ALLOWLIST))


def _clean_degrees(raw_degrees: List[str]) -> List[str]:
    """Keep energy-relevant majors: engineering, science, technology, MBA.

    Drops noisy/irrelevant entries: quoted strings, BCA/HR/marketing MBAs,
    and anything with line breaks or outside length bounds.
    """
    keep_kw = (
        "chemical engineering", "mechanical engineering",
        "electrical engineering", "petroleum engineering",
        "civil engineering", "nuclear engineering", "industrial engineering",
        "environmental", "materials", "energy systems",
        "computer science", "information technology",
        "power engineering", "process technology",
        "instrumentation",
    )
    deny_kw = (
        "marketing", "human resources", "tourism", "airport",
        "customer care", "hotel", "hospitality", " engg", " bpo",
        "enginerring",  # typo variants
        "information technology and management",  # off-topic MBA variants
        '"',  # quoted degree strings
    )
    out = set()
    for d in raw_degrees:
        d = d.strip()
        if not d:
            continue
        dl = d.lower()
        if "\n" in d or len(d) > 60:
            continue
        # Drop entries with lowercase start (data-quality noise from the v2
        # taxonomy's hf_normalized source, e.g., "bachelor of ... in ...").
        if not d[0].isupper():
            continue
        if any(bad in dl for bad in deny_kw):
            continue
        # Accept only if it contains an energy-relevant keyword OR is MBA
        if dl == "mba" or dl == "executive mba":
            out.add(d)
            continue
        if any(k in dl for k in keep_kw):
            out.add(d)
    # Ensure a few short canonical degrees are always present.
    canonical = [
        "B.S. in Chemical Engineering", "B.S. in Electrical Engineering",
        "B.S. in Mechanical Engineering", "B.S. in Petroleum Engineering",
        "B.S. in Civil Engineering", "B.S. in Nuclear Engineering",
        "B.S. in Environmental Science", "B.S. in Materials Science",
        "M.S. in Chemical Engineering", "M.S. in Petroleum Engineering",
        "M.S. in Electrical Engineering", "M.S. in Mechanical Engineering",
        "M.S. in Energy Systems", "M.S. in Materials Science",
        "M.S. in Environmental Science", "M.S. in Nuclear Engineering",
        "Ph.D. in Petroleum Engineering", "Ph.D. in Chemical Engineering",
        "Ph.D. in Materials Science", "Ph.D. in Nuclear Engineering",
        "MBA", "Executive MBA",
        "Associate in Process Technology",
        "Associate in Instrumentation Technology",
    ]
    out.update(canonical)
    return sorted(out)


def _tokenize_and_tag(text: str, entities: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Punctuation-tolerant longest-span-first BIO tagger.

    Tokens are whitespace-split. Comparison strips surrounding punctuation
    so 'license.' still matches 'license'. Longer spans are tagged first
    to avoid sub-span collisions. First occurrence only per entity string.
    """
    tokens = text.split()
    labels = ["O"] * len(tokens)

    def _norm(tok: str) -> str:
        return tok.strip(" .,;:()[]\"'").lower()

    normed = [_norm(t) for t in tokens]

    for span_text, etype in sorted(
        entities.items(), key=lambda x: -len(x[0].split())
    ):
        span_tokens = [_norm(t) for t in span_text.split()]
        n = len(span_tokens)
        if n == 0:
            continue
        for i in range(len(tokens) - n + 1):
            if normed[i:i + n] == span_tokens:
                if any(labels[i + j] != "O" for j in range(n)):
                    continue
                labels[i] = f"B-{etype}"
                for j in range(1, n):
                    labels[i + j] = f"I-{etype}"
                break
    return tokens, labels


# ---------------------------------------------------------------------------
# O*NET corpus loader — per-SOC bucket of entity candidates
# ---------------------------------------------------------------------------


class OnetCorpus:
    """Loads all per-SOC O*NET JSON files into entity-candidate buckets."""

    def __init__(self, onet_dir: Path = RAW_ONET_DIR):
        self.per_soc: Dict[str, Dict] = {}
        self.all_tools: Counter = Counter()
        self.all_knowledge: Counter = Counter()
        self.all_dwas: Counter = Counter()
        self.all_associations: Counter = Counter()
        self._load(onet_dir)

    def _load(self, onet_dir: Path) -> None:
        for jf in sorted(onet_dir.glob("*.json")):
            if jf.name.startswith("_"):
                continue
            try:
                d = json.loads(jf.read_text())
            except Exception as exc:
                print(f"[v7gen] skip {jf}: {exc}")
                continue
            soc = d.get("soc_code", "")
            if not soc:
                continue

            tools_raw = ONetScraper.flatten_technology_skills(
                d.get("technology_skills", {}) or {}
            )
            tools = []
            for t in tools_raw:
                cn = _clean_tool_name(t.get("name", ""))
                if cn:
                    tools.append(cn)

            def _names(section_key: str) -> List[str]:
                section = d.get(section_key, {}) or {}
                return [
                    el.get("name", "").strip()
                    for el in (section.get("element", []) or [])
                    if isinstance(el, dict) and el.get("name", "").strip()
                ]

            knowledge = _names("knowledge")
            abilities = _names("abilities")
            work_styles = _names("work_styles")
            generic_skills = _names("skills")
            dwas = [
                a["title"] for a in
                ONetScraper.flatten_detailed_work_activities(
                    d.get("detailed_work_activities", {}) or {}
                )
            ]
            assocs = [
                a.get("name", "").strip()
                for a in (
                    (d.get("professional_associations", {}) or {})
                    .get("association", []) or []
                )
                if isinstance(a, dict) and a.get("name", "").strip()
            ]

            self.per_soc[soc] = {
                "title": d.get("title", ""),
                "tools": tools,
                "knowledge": knowledge,
                "abilities": abilities,
                "work_styles": work_styles,
                "generic_skills": generic_skills,
                "dwas": dwas,
                "associations": assocs,
            }
            for t in tools:
                self.all_tools[t] += 1
            for k in knowledge:
                self.all_knowledge[k] += 1
            for a in dwas:
                self.all_dwas[a] += 1
            for a in assocs:
                self.all_associations[a] += 1

    def soc_codes(self) -> List[str]:
        return sorted(self.per_soc.keys())


# ---------------------------------------------------------------------------
# Block renderers — each returns (text, entities_map)
# ---------------------------------------------------------------------------


PREAMBLE_TEMPLATES = [
    "{years} as a {role} at {employer}.",
    "Currently {role} at {employer} with {years}.",
    "{years} working at {employer} as {role}.",
    "{years} in {industry} operations. Currently at {employer} as {role}.",
    "Employed at {employer} with {years} spanning {industry}.",
    "{role} at {employer}. {years} focused on {industry}.",
    "{years} across {industry}. {role} at {employer}.",
    "{years} in {industry}. Currently at {employer} as {role}.",
]


def render_preamble(rng, soc, industries, employers, roles, years_templates):
    n = rng.randint(2, 25)
    years_expr = rng.choice(years_templates).format(n=n)
    role = rng.choice(roles)
    employer = rng.choice(employers)
    include_industry = industries and rng.random() < 0.70
    industry = rng.choice(industries) if include_industry else None

    viable = [t for t in PREAMBLE_TEMPLATES if ("{industry}" in t) == bool(industry)]
    tmpl = rng.choice(viable)
    kwargs = {
        "years": years_expr, "role": role, "employer": employer,
    }
    if industry:
        kwargs["industry"] = industry
    text = tmpl.format(**kwargs)
    ents: Dict[str, str] = {
        years_expr: "YEARS_EXP",
        employer: "EMPLOYER",
    }
    if industry:
        ents[industry] = "INDUSTRY"
    return text, ents


SKILL_BLOCK_TEMPLATES = [
    "Expertise in {s1}, {s2}, and {s3}.",
    "Skilled in {s1}, {s2}, and {s3}.",
    "Core skills: {s1}, {s2}, {s3}.",
    "Deep knowledge of {s1} and {s2}.",
    "Strong in {s1}, {s2}, and {s3}.",
    "Technical capabilities include {s1}, {s2}, and {s3}.",
]


def render_skill_block(rng, skill_pool):
    k = rng.choice([2, 3, 3, 3])
    if len(skill_pool) < k:
        return "", {}
    picks = rng.sample(skill_pool, k)
    if k == 2:
        text = rng.choice([
            "Deep knowledge of {s1} and {s2}.",
            "Expertise in {s1} and {s2}.",
        ]).format(s1=picks[0], s2=picks[1])
    else:
        tmpl = rng.choice([t for t in SKILL_BLOCK_TEMPLATES if "{s3}" in t])
        text = tmpl.format(s1=picks[0], s2=picks[1], s3=picks[2])
    ents = {p: "SKILL" for p in picks}
    return text, ents


TOOL_BLOCK_TEMPLATES = [
    "Proficient in {t1}, {t2}, and {t3}.",
    "Tools: {t1}, {t2}, {t3}.",
    "Hands-on with {t1} and {t2}.",
    "Software stack: {t1}, {t2}, and {t3}.",
    "Experienced using {t1}, {t2}, and {t3}.",
    "Day-to-day tools include {t1} and {t2}.",
]


def render_tool_block(rng, tool_pool):
    if not tool_pool:
        return "", {}
    k = rng.choice([2, 2, 3, 3, 3])
    k = min(k, len(tool_pool))
    if k < 2:
        return "", {}
    picks = rng.sample(tool_pool, k)
    if k == 2:
        tmpl = rng.choice([t for t in TOOL_BLOCK_TEMPLATES if "{t3}" not in t])
        text = tmpl.format(t1=picks[0], t2=picks[1])
    else:
        tmpl = rng.choice([t for t in TOOL_BLOCK_TEMPLATES if "{t3}" in t])
        text = tmpl.format(t1=picks[0], t2=picks[1], t3=picks[2])
    ents = {p: "TOOL" for p in picks}
    return text, ents


def render_cert_block(rng, cert_pool):
    if not cert_pool:
        return "", {}
    k = rng.choice([2, 2, 3, 3, 3])
    k = min(k, len(cert_pool))
    if k < 2:
        return "", {}
    picks = rng.sample(cert_pool, k)
    if k == 2:
        text = rng.choice([
            "Certifications: {c1} and {c2}.",
            "Holds {c1} and {c2}.",
            "Credentials: {c1}, {c2}.",
        ]).format(c1=picks[0], c2=picks[1])
    else:
        text = rng.choice([
            "Certifications: {c1}, {c2}, and {c3}.",
            "Holds {c1}, {c2}, {c3}.",
            "Credentials: {c1}, {c2}, and {c3}.",
        ]).format(c1=picks[0], c2=picks[1], c3=picks[2])
    ents = {p: "CERT" for p in picks}
    return text, ents


def render_degree_block(rng, degree_pool):
    if not degree_pool:
        return "", {}
    k = rng.choice([1, 1, 2])
    k = min(k, len(degree_pool))
    picks = rng.sample(degree_pool, k)
    universities = [
        "Texas A&M University", "University of Houston",
        "University of Texas at Austin", "Stanford University",
        "MIT", "Rice University", "Colorado School of Mines",
        "Georgia Tech", "Imperial College London",
        "ETH Zurich", "Penn State University",
        "Louisiana State University",
    ]
    uni = rng.choice(universities)
    if k == 1:
        text = rng.choice([
            "Education: {d1} from {u}.",
            "{d1} from {u}.",
            "Holds a {d1} from {u}.",
        ]).format(d1=picks[0], u=uni)
    else:
        text = rng.choice([
            "Education: {d1} and {d2} from {u}.",
            "{d1}, {d2} from {u}.",
        ]).format(d1=picks[0], d2=picks[1], u=uni)
    ents = {p: "DEGREE" for p in picks}
    return text, ents


def render_industry_block(rng, industries):
    if not industries:
        return "", {}
    k = rng.choice([1, 2, 2, 2])
    k = min(k, len(industries))
    picks = rng.sample(industries, k)
    if k == 1:
        text = rng.choice([
            "Sector focus: {i1}.",
            "Industry experience in {i1}.",
            "Deep {i1} experience.",
            "Primary sector: {i1}.",
        ]).format(i1=picks[0])
    else:
        text = rng.choice([
            "Sector experience spans {i1} and {i2}.",
            "Worked across {i1} and {i2}.",
            "Experience in {i1} and {i2} operations.",
        ]).format(i1=picks[0], i2=picks[1])
    ents = {p: "INDUSTRY" for p in picks}
    return text, ents


def render_location_block(rng, locations):
    if not locations:
        return "", {}
    k = rng.choice([1, 1, 2])
    k = min(k, len(locations))
    picks = rng.sample(locations, k)
    if k == 1:
        text = rng.choice([
            "Field experience in {l1}.",
            "Projects based in {l1}.",
            "Assignments in {l1}.",
            "Worked out of {l1}.",
        ]).format(l1=picks[0])
    else:
        text = rng.choice([
            "Field experience across {l1} and {l2}.",
            "Assignments in {l1} and {l2}.",
            "Projects in {l1} and {l2}.",
        ]).format(l1=picks[0], l2=picks[1])
    ents = {p: "LOCATION" for p in picks}
    return text, ents


def render_project_block(rng, projects):
    if not projects:
        return "", {}
    pick = rng.choice(projects)
    text = rng.choice([
        "Led commissioning of {p}.",
        "Key contributor on {p}.",
        "Delivered scope on {p}.",
        "Worked on {p}.",
        "Technical lead for {p}.",
    ]).format(p=pick)
    return text, {pick: "PROJECT"}


def render_soft_skill_block(rng, soft_skills):
    if not soft_skills:
        return "", {}
    k = rng.choice([2, 2, 3])
    k = min(k, len(soft_skills))
    picks = rng.sample(soft_skills, k)
    if k == 2:
        text = rng.choice([
            "Known for {s1} and {s2}.",
            "Strengths in {s1} and {s2}.",
            "Demonstrated {s1} and {s2}.",
        ]).format(s1=picks[0], s2=picks[1])
    else:
        text = rng.choice([
            "Known for {s1}, {s2}, and {s3}.",
            "Strengths: {s1}, {s2}, and {s3}.",
        ]).format(s1=picks[0], s2=picks[1], s3=picks[2])
    ents = {p: "SOFT_SKILL" for p in picks}
    return text, ents


# ---------------------------------------------------------------------------
# V7 Generator
# ---------------------------------------------------------------------------


class V7Generator:
    """Per-SOC sample generator. Assembles block-based synthetic resumes
    with 10-type BIO tags."""

    def __init__(self, onet_corpus: OnetCorpus, rng: random.Random):
        self.onet = onet_corpus
        self.rng = rng
        self.employers = _build_employer_pool()
        self.degrees = _clean_degrees(list(V2_DEGREES))
        v2_certs_clean = [
            c for c in V2_CERTS if 3 <= len(c) <= 60 and "\n" not in c
        ]
        # v7 expansion: union with automation-standards vocab so synthetic
        # samples for automation SOCs can surface ISA/IEC/NERC codes;
        # plus TIER 0 aux taxonomies (CEWD/SPE/IADC/IRENA/ASME credentials).
        self.certs = sorted(
            set(v2_certs_clean)
            | set(CERTS_V7_AUTOMATION_STANDARDS)
            | set(AUX_CERTS_FROM_TAXONOMIES)
        )
        self.years_expressions = V7_YEARS_EXPRESSIONS
        # Per-SOC pools are computed at generate-time to mix O*NET + seeds.

    def _skill_pool(self, soc: str) -> List[str]:
        """SOC-specific SKILL pool. SKILL_SEEDS is primary; augmented with
        a small filtered slice of O*NET knowledge (only domain-specific)."""
        bucket = self.onet.per_soc.get(soc, {})
        # Drop generic O*NET knowledge domains — they overlap with common
        # English and cause mislabeling. Keep only clearly technical ones.
        too_generic = {
            "Mathematics", "Physics", "Chemistry", "Biology",
            "English Language", "Economics and Accounting",
            "Administration and Management", "Customer and Personal Service",
            "Clerical", "Education and Training",
            "Psychology", "Sociology and Anthropology",
            "Foreign Language", "Fine Arts",
            "History and Archeology", "Philosophy and Theology",
            "Communications and Media", "Telecommunications",
            "Transportation", "Production and Processing",
            "Food Production", "Computers and Electronics",
            "Public Safety and Security", "Law and Government",
            "Therapy and Counseling", "Medicine and Dentistry",
            "Building and Construction", "Sales and Marketing",
            "Personnel and Human Resources", "Geography",
            "Design", "Mechanical", "Administrative", "Engineering",
            "Engineering and Technology",
        }
        knowledge = [
            k for k in bucket.get("knowledge", [])
            if k and 5 <= len(k) <= 40 and k not in too_generic
        ]
        # O*NET DWAs are full imperative sentences ("Develop safety
        # standards, policies, or procedures."). Use them only if they
        # are short noun-phrase-like — skip any with internal punctuation
        # (comma or period before the final one) or starting with an
        # imperative verb we can't repurpose cleanly.
        dwas = []
        for raw in bucket.get("dwas", []) or []:
            s = raw.rstrip(".").strip()
            if not s or len(s) < 5 or len(s) > 60:
                continue
            # Skip sentences with internal commas or quotes — they pollute
            # BIO tags (entire sentence becomes one SKILL span, and trailing
            # ' .' gets glued to the last I-SKILL token).
            if "," in s or '"' in s or "." in s:
                continue
            # Skip imperative-verb forms that don't read as a noun phrase.
            first = s.split()[0].lower()
            imperative_verbs = {
                "develop", "determine", "research", "evaluate", "design",
                "estimate", "prepare", "conduct", "monitor", "direct",
                "apply", "test", "inspect", "train", "maintain",
                "resolve", "document", "operate", "record", "select",
                "calculate", "analyze", "investigate", "coordinate",
                "assign", "schedule", "implement",
                "diagnose", "measure", "advise", "assemble", "assist",
                "collect", "communicate", "compile", "confer", "create",
                "educate", "gather", "identify", "install", "interpret",
                "lead", "manage", "negotiate", "obtain", "plan",
                "provide", "read", "recommend", "review", "set",
                "supervise", "take", "translate", "update", "verify",
                "write", "troubleshoot", "clean", "discuss", "ensure",
                "explain", "file", "help", "issue", "join", "keep",
                "pay", "perform", "remove", "report", "request",
                "sell", "sort", "submit", "teach", "use",
                "adjust", "lubricate", "exchange", "watch",
                "disassemble", "notify", "observe", "reassemble",
                "repair", "enter", "climb", "position", "drive",
                "cut", "replace", "devise", "fabricate", "mark",
                "move", "load", "unload", "align", "check",
                "dig", "deliver", "distribute", "drive", "shut",
                "start", "stop", "handle", "pour", "paint",
                "pump", "tend", "weld", "wire", "hold",
                "oversee", "guide", "build", "prevent", "restore",
                "respond", "survey", "control",
                "approve", "hire", "testify", "mix", "ignite",
                "confirm", "prescribe", "examine",
            }
            if first in imperative_verbs:
                continue
            dwas.append(s)
        # TIER 0 aux taxonomies (CEWD/SPE/IADC/IRENA/ASME) extend the SKILL
        # candidate pool with ~700 peer-curated energy-domain skills.
        return SKILL_SEEDS + AUX_SKILLS_FROM_TAXONOMIES + knowledge + dwas

    def _tool_pool(self, soc: str) -> List[str]:
        """Combine SOC-specific O*NET tools + canonical TOOL list."""
        bucket = self.onet.per_soc.get(soc, {})
        soc_tools = bucket.get("tools", [])
        # Filter SOC tools through the canonical list to prefer recognizable
        # names. Also include any canonical tool (randomly) so samples from
        # role-specific SOCs still see Python/MATLAB/SAP.
        pool = list(set(soc_tools) | set(TOOLS_CANONICAL))
        return pool

    def _soft_skill_pool(self, soc: str) -> List[str]:
        """Pure interpersonal skills. Use SOFT_SKILLS_VOCAB; optionally
        seed with O*NET work_styles (when appropriate)."""
        # work_styles are trait-adjacent: "Attention to Detail",
        # "Dependability", "Innovation". Not quite standard soft skills but
        # acceptable. Keep SOFT_SKILLS_VOCAB as primary.
        # TIER 0: CEWD personal-effectiveness tier extends the pool with
        # additional peer-curated soft-skill candidates.
        return SOFT_SKILLS_VOCAB + AUX_SOFT_SKILLS_FROM_TAXONOMIES

    def _industry_pool(self, soc: str) -> List[str]:
        soc_industries = SOC_TO_INDUSTRIES.get(soc, [])
        # Mix SOC-specific with global fallback (weighted toward specific).
        if soc_industries and self.rng.random() < 0.75:
            return soc_industries
        return soc_industries + self.rng.sample(
            ENERGY_INDUSTRIES_VOCAB,
            min(4, len(ENERGY_INDUSTRIES_VOCAB)),
        )

    def _location_pool(self, soc: str) -> List[str]:
        soc_locs = SOC_TO_LOCATIONS.get(soc, [])
        if soc_locs and self.rng.random() < 0.75:
            return soc_locs
        return soc_locs + self.rng.sample(
            ENERGY_LOCATIONS_VOCAB,
            min(3, len(ENERGY_LOCATIONS_VOCAB)),
        )

    def _project_pool(self, soc: str) -> List[str]:
        soc_projects = SOC_TO_PROJECTS.get(soc, [])
        if soc_projects:
            return soc_projects
        return ENERGY_PROJECTS_VOCAB

    def generate_sample(self, soc: Optional[str] = None) -> Dict:
        if soc is None:
            soc = self.rng.choice(list(self.onet.per_soc.keys()))

        industries = self._industry_pool(soc)
        locations = self._location_pool(soc)
        projects = self._project_pool(soc)
        roles = SOC_TO_ROLES.get(soc, ["Senior Engineer", "Lead Engineer"])
        skills = self._skill_pool(soc)
        tools = self._tool_pool(soc)
        soft = self._soft_skill_pool(soc)

        parts: List[str] = []
        all_ents: Dict[str, str] = {}

        def _add(text: str, ents: Dict[str, str]) -> None:
            if not text:
                return
            parts.append(text)
            for span, etype in ents.items():
                # Skip if the span would collide with an already-assigned
                # entity of a different type (longest-first tagging handles
                # nesting; we avoid cross-type clashes here.)
                if span in all_ents and all_ents[span] != etype:
                    continue
                all_ents[span] = etype

        # Preamble always present.
        _add(*render_preamble(
            self.rng, soc, industries, self.employers, roles,
            self.years_expressions,
        ))

        # Skills — high probability
        if self.rng.random() < 0.95:
            _add(*render_skill_block(self.rng, skills))

        # Tools — very high probability (core new Tier-1 signal)
        if self.rng.random() < 0.85:
            _add(*render_tool_block(self.rng, tools))

        # Certs — high probability
        if self.rng.random() < 0.80:
            _add(*render_cert_block(self.rng, self.certs))

        # Degree — high probability
        if self.rng.random() < 0.80:
            _add(*render_degree_block(self.rng, self.degrees))

        # Industry block — medium probability (preamble already has one
        # sometimes; add an extra industry mention here).
        if self.rng.random() < 0.55:
            _add(*render_industry_block(self.rng, industries))

        # Location block — medium probability
        if self.rng.random() < 0.55:
            _add(*render_location_block(self.rng, locations))

        # Project block — low-medium probability (Tier 3, long-tail)
        if self.rng.random() < 0.40 and projects:
            _add(*render_project_block(self.rng, projects))

        # Soft-skill block — medium probability (Tier 3)
        if self.rng.random() < 0.55:
            _add(*render_soft_skill_block(self.rng, soft))

        text = " ".join(parts)
        tokens, labels = _tokenize_and_tag(text, all_ents)
        return {
            "tokens": tokens,
            "ner_tags": labels,
            "text": text,
            "soc_code": soc,
        }


# ---------------------------------------------------------------------------
# Hand-labeled re-annotation (Step B)
#
# Re-annotate the 40 v6 hand-labeled energy resumes with the 5 new v7 entity
# types. The 40 resumes live inside data/generate_ner_data._hand_labeled_samples
# (v6 is preserved as a fallback — per V7_PROGRESS.md constraints, that file
# is NOT edited). We import the v6 samples, reconstruct the {span: etype} map
# from their BIO tags, reclassify SKILL spans that are actually TOOLs (or a
# small set of SOFT_SKILLs), and auto-augment with 5 new-type candidates
# drawn from the same vocabularies the synthetic generator uses.
# ---------------------------------------------------------------------------


def _reconstruct_entities_from_bio(
    tokens: List[str], tags: List[str]
) -> Dict[str, str]:
    """Rebuild a {span_text: etype} dict from BIO-tagged tokens.

    The span text is whitespace-joined and stripped of surrounding
    punctuation so it matches the form expected by `_tokenize_and_tag`."""
    entities: Dict[str, str] = {}
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == "I-" + etype:
                i += 1
            span = " ".join(tokens[start:i]).strip(" .,;:()[]\"'")
            if span:
                entities[span] = etype
        else:
            i += 1
    return entities


def _word_boundary_present(text: str, candidate: str) -> bool:
    """Case-insensitive word-boundary membership test."""
    pattern = (
        r"(?<![A-Za-z0-9])" + re.escape(candidate) + r"(?![A-Za-z0-9])"
    )
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _reannotate_v6_sample(
    text: str, v6_entities: Dict[str, str]
) -> Dict[str, str]:
    """Given v6 5-type entities, return 10-type entities for the same text.

    Two-stage augmentation:
      1. Reclassify SKILL spans whose normalized form is in TOOLS_CANONICAL
         or SOFT_SKILLS_VOCAB. This corrects v6 entries like "AutoCAD" that
         were tagged SKILL but are actually TOOL.
      2. Auto-tag new spans for TOOL / INDUSTRY / LOCATION / PROJECT /
         SOFT_SKILL using the same vocabularies the synthetic generator
         uses. Only spans that appear in the text (word-boundary match)
         and don't collide with an existing entity are added.
    """
    tools_lower = {t.lower() for t in TOOLS_CANONICAL}
    soft_lower = {s.lower() for s in SOFT_SKILLS_VOCAB}
    ents: Dict[str, str] = dict(v6_entities)

    # Stage 1 — reclassify existing SKILL spans.
    for span, etype in list(ents.items()):
        if etype != "SKILL":
            continue
        sl = span.strip().lower()
        if sl in tools_lower:
            ents[span] = "TOOL"
        elif sl in soft_lower:
            ents[span] = "SOFT_SKILL"

    # Stage 2 — auto-tag new entity types from vocabulary.
    def _add_if_present(candidates: List[str], etype: str) -> None:
        for cand in candidates:
            if cand in ents:
                continue
            if _word_boundary_present(text, cand):
                ents[cand] = etype

    _add_if_present(TOOLS_CANONICAL, "TOOL")
    _add_if_present(CERTS_V7_AUTOMATION_STANDARDS, "CERT")
    _add_if_present(ENERGY_INDUSTRIES_VOCAB, "INDUSTRY")
    _add_if_present(ENERGY_LOCATIONS_VOCAB, "LOCATION")
    _add_if_present(ENERGY_PROJECTS_VOCAB, "PROJECT")
    _add_if_present(SOFT_SKILLS_VOCAB, "SOFT_SKILL")

    return ents


def _heldout_eval_texts() -> set:
    """Return the set of resume texts from the frozen held-out eval set.

    Used to EXCLUDE held-out resumes from the training data so the train
    / eval partition is clean (no leakage). If the eval file is missing,
    returns an empty set and training includes everything (matches v6
    behavior when the eval set hasn't been built yet).
    """
    eval_path = BASE / "processed" / "eval_held_out.json"
    if not eval_path.exists():
        return set()
    try:
        d = json.loads(eval_path.read_text())
        return {
            " ".join(s["tokens"])
            for s in d.get("eval_samples", [])
            if "tokens" in s
        }
    except Exception:
        return set()


def hand_labeled_samples_v7(
    repeat: int = 80,
    exclude_heldout: bool = True,
) -> List[Dict]:
    """Load the 40 v6 hand-labeled resumes, re-annotate with 10 types.

    When `exclude_heldout` is True (default), the 5 resumes in the frozen
    held-out eval set are filtered out — prevents training on samples we
    evaluate on. v6 stats show {hand: 2800} = 35 × 80, consistent with
    the 5 held-out exclusion.

    Each remaining unique resume is emitted `repeat` times.
    """
    # Lazy import — keeps the v7 module importable even if the v6 module
    # is renamed or moved.
    from data.generate_ner_data import _hand_labeled_samples as _v6_hl  # noqa: WPS433

    raw = _v6_hl()
    # v6 repeats each unique sample 80 times — deduplicate by text.
    seen: set = set()
    uniques: List[Dict] = []
    for s in raw:
        t = s["text"]
        if t in seen:
            continue
        seen.add(t)
        uniques.append(s)

    heldout = _heldout_eval_texts() if exclude_heldout else set()
    n_excluded = 0

    out: List[Dict] = []
    for s in uniques:
        if s["text"] in heldout:
            n_excluded += 1
            continue
        v6_entities = _reconstruct_entities_from_bio(s["tokens"], s["ner_tags"])
        v7_entities = _reannotate_v6_sample(s["text"], v6_entities)
        tokens, tags = _tokenize_and_tag(s["text"], v7_entities)
        # Use text hash as resume_id so the stratified splitter can keep
        # all 80 repeats together on the same side of the train/val line.
        resume_id = f"hand_{abs(hash(s['text'])) % (10 ** 10):010d}"
        sample = {
            "tokens": tokens,
            "ner_tags": tags,
            "text": s["text"],
            "source": "hand_labeled_v7",
            "resume_id": resume_id,
        }
        out.extend([sample] * repeat)
    if n_excluded:
        print(f"[v7gen] hand_labeled: excluded {n_excluded} held-out resume(s)")
    return out


# Noise spans commonly seen in Mehyaar annotations — generic English words
# that the original annotators marked as SKILL but which aren't real skills.
# Dropping them at ingest keeps the v7 SKILL gold clean.
_MEHYAAR_NOISE_SPANS = {
    "building", "processing", "knowledge", "experience", "skills", "skill",
    "ability", "abilities", "working", "work", "team", "teamwork",
    "quick", "excellent", "strong", "good", "fast", "deep", "high",
    "detail", "details", "quality", "quantity", "basic", "advanced",
    "solving", "proficient", "proficiency", "familiar", "familiarity",
    "understanding", "awareness", "exposure", "knowledge", "usage",
    "hands", "hands-on", "self", "selfmotivated", "eager", "creative",
    "organized", "responsible", "reliable", "motivated",
    "development", "developer", "engineer", "engineering",  # too generic
    "programming", "coding", "software",
    "insights", "object", "anomaly", "oxide",
    "various", "different", "multiple", "several",
    "etc", "i.e", "e.g",
}


def mehyaar_aux_samples_v7(
    repeat: int = 1,
    max_samples: int = 5000,
    max_tokens_per_sample: int = 256,
    exclude_heldout: bool = True,
) -> List[Dict]:
    """Load the Mehyaar 5,029-CV IT-skills NER dataset and re-annotate
    it with v7 10-type BIO tags.

    Source: https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes
    License: MIT. Manual annotations, SKILL-only in the original (char-
    offset format). The v7 pipeline:

      1. Drop noise spans (generic English words like "Knowledge",
         "Processing", "Building") — original annotations have a non-
         trivial false-positive rate on these.
      2. Reclassify SKILL→TOOL for known tools (Python, AutoCAD, SAP...)
         via TOOLS_CANONICAL.
      3. Auto-tag new entity types (INDUSTRY, LOCATION, PROJECT,
         SOFT_SKILL) where the v7 vocab finds a hit.
      4. Whitespace-tokenize; truncate to `max_tokens_per_sample` so
         training MAX_LEN=256 doesn't silently drop back-loaded content.

    These CVs are IT-domain (data scientists, developers) not energy —
    so they contribute out-of-domain diversity for SKILL/TOOL but
    shouldn't dominate. Use at repeat=1.

    Returns [] if the dataset directory isn't present.
    """
    mehyaar_dir = BASE / "raw" / "huggingface" / "mehyaar" / "ResumesJsonAnnotated"
    if not mehyaar_dir.exists():
        print(f"[v7gen] mehyaar: {mehyaar_dir} missing — skipping")
        return []

    files = sorted(mehyaar_dir.glob("*.json"))
    if max_samples:
        files = files[:max_samples]
    heldout = _heldout_eval_texts() if exclude_heldout else set()

    out: List[Dict] = []
    n_noise_dropped = 0
    n_empty_after_filter = 0
    n_heldout_excluded = 0

    for idx, cv_file in enumerate(files):
        try:
            d = json.loads(cv_file.read_text())
        except Exception:
            continue
        text = d.get("text", "")
        if not text or len(text) < 100:
            continue
        if exclude_heldout:
            text_norm = " ".join(text.split())
            if text_norm in heldout or text in heldout:
                n_heldout_excluded += 1
                continue

        # Extract annotations: [[start, end, "SKILL: <name>"], ...]
        v6_entities: Dict[str, str] = {}
        for ann in d.get("annotations", []):
            if not isinstance(ann, list) or len(ann) < 3:
                continue
            start, end, label = ann[0], ann[1], ann[2]
            if not isinstance(label, str):
                continue
            # Only accept SKILL: prefix (original schema).
            if ":" in label:
                etype, _, _label_text = label.partition(":")
                etype = etype.strip().upper()
            else:
                etype = label.strip().upper()
            if etype != "SKILL":
                continue
            try:
                span = text[int(start):int(end)].strip()
            except (ValueError, TypeError):
                continue
            if len(span) < 2 or len(span) > 60:
                continue
            low = span.lower().strip(" .,;:()[]\"'-")
            if low in _MEHYAAR_NOISE_SPANS:
                n_noise_dropped += 1
                continue
            # Skip 1-word spans that look like noise after lowercasing.
            v6_entities[span] = "SKILL"

        if not v6_entities:
            n_empty_after_filter += 1
            continue

        # v6 → v7: reclassify SKILL→TOOL, add new-type vocab spans.
        v7_entities = _reannotate_v6_sample(text, v6_entities)
        tokens, tags = _tokenize_and_tag(text, v7_entities)
        if not tokens:
            continue
        # Truncate to the training MAX_LEN so short-by-default samples
        # preserve their most-annotated front-loaded portion (skills blocks
        # in CVs typically appear early).
        if len(tokens) > max_tokens_per_sample:
            tokens = tokens[:max_tokens_per_sample]
            tags = tags[:max_tokens_per_sample]
        resume_id = f"mh_{idx:05d}"
        sample = {
            "tokens": tokens,
            "ner_tags": tags,
            "text": text[:3000],  # trim for storage — training uses tokens only
            "source": "mehyaar_v7",
            "resume_id": resume_id,
        }
        out.extend([sample] * repeat)

    if n_noise_dropped or n_empty_after_filter or n_heldout_excluded:
        print(
            f"[v7gen] mehyaar: {len(out)} samples, "
            f"{n_noise_dropped} noise spans dropped, "
            f"{n_empty_after_filter} CVs had no clean spans, "
            f"{n_heldout_excluded} held-out exclusions"
        )
    return out


def dataturks_aux_samples_v7(
    repeat: int = 1,
    exclude_heldout: bool = True,
) -> List[Dict]:
    """Load the Dataturks 220 resumes and re-annotate them with v7 10 types.

    Dataturks resumes are IT-domain (Infosys, Accenture, Oracle), not
    energy-specific. Their existing SKILL labels are noisy (job titles
    sometimes tagged SKILL), but CERT/DEGREE/EMPLOYER/YEARS_EXP labels
    are clean. The v7 re-annotation reclassifies tool-like SKILLs
    (AutoCAD/Python/SQL) to TOOL and adds new-type spans where the
    vocabulary matches. Used at `repeat=1` so they don't dominate the
    40 × 80 hand-labeled signal but still contribute ~220 samples of
    out-of-domain diversity for the legacy 5 types.

    Returns [] if data/raw/kaggle/dataturks_parsed.json is missing.
    """
    dt_path = BASE / "raw" / "kaggle" / "dataturks_parsed.json"
    if not dt_path.exists():
        print(f"[v7gen] dataturks: {dt_path} missing — skipping aux data")
        return []

    raw = json.loads(dt_path.read_text())
    heldout = _heldout_eval_texts() if exclude_heldout else set()
    n_excluded = 0

    out: List[Dict] = []
    for idx, r in enumerate(raw):
        text = r.get("text", "")
        if not text:
            continue
        # The held-out eval uses tokenized text, whitespace-joined. Match
        # either the raw text or its whitespace-normalized form.
        text_norm = " ".join(text.split())
        if text_norm in heldout or text in heldout:
            n_excluded += 1
            continue

        # Dataturks format: {text, entities: {span: etype}}. The entities
        # dict is already in the shape _reannotate_v6_sample expects.
        v6_entities = {
            span: etype for span, etype in (r.get("entities") or {}).items()
            if etype in ("SKILL", "CERT", "DEGREE", "EMPLOYER", "YEARS_EXP")
        }
        v7_entities = _reannotate_v6_sample(text, v6_entities)
        tokens, tags = _tokenize_and_tag(text, v7_entities)
        if not tokens:
            continue
        resume_id = f"dt_{idx:04d}"
        sample = {
            "tokens": tokens,
            "ner_tags": tags,
            "text": text,
            "source": "dataturks_v7",
            "resume_id": resume_id,
        }
        out.extend([sample] * repeat)
    if n_excluded:
        print(f"[v7gen] dataturks: excluded {n_excluded} held-out resume(s)")
    return out


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------


def _synthetic_samples(n_samples: int, seed: int) -> Tuple[List[Dict], List[str]]:
    """Generate the synthetic half of the v7 dataset."""
    rng = random.Random(seed)
    corpus = OnetCorpus(RAW_ONET_DIR)
    if not corpus.per_soc:
        raise RuntimeError(
            f"No O*NET files found under {RAW_ONET_DIR}. "
            "Run data/scrapers/onet_scraper.py first."
        )
    gen = V7Generator(corpus, rng)

    soc_codes = corpus.soc_codes()
    per_soc_count = n_samples // len(soc_codes)
    remainder = n_samples - per_soc_count * len(soc_codes)

    data: List[Dict] = []
    next_id = 0
    for soc in soc_codes:
        for _ in range(per_soc_count):
            ex = gen.generate_sample(soc)
            ex["source"] = "synthetic_v7"
            # resume_id makes every synthetic row unique — stratified split
            # doesn't need to deduplicate; each row is already standalone.
            ex["resume_id"] = f"syn_{next_id:06d}"
            next_id += 1
            data.append(ex)
    for _ in range(remainder):
        ex = gen.generate_sample(rng.choice(soc_codes))
        ex["source"] = "synthetic_v7"
        ex["resume_id"] = f"syn_{next_id:06d}"
        next_id += 1
        data.append(ex)

    rng.shuffle(data)
    return data, soc_codes


def _summarize(data: List[Dict]) -> Dict:
    """Return a stats dict for a list of v7 samples."""
    tag_counts: Counter = Counter()
    source_counts: Counter = Counter()
    for ex in data:
        source_counts[ex.get("source", "unknown")] += 1
        for tag in ex["ner_tags"]:
            if tag.startswith("B-"):
                tag_counts[tag[2:]] += 1
    return {
        "total_samples": len(data),
        "source_counts": dict(source_counts),
        "entity_span_counts": dict(tag_counts),
    }


def _write_dataset(data: List[Dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=1)
    stats = _summarize(data)
    stats_path = output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[v7gen] Wrote {len(data)} samples -> {output_path}")
    print(f"[v7gen] Stats -> {stats_path}")
    print(f"[v7gen] B-tag counts: {stats['entity_span_counts']}")


def generate_dataset(
    n_samples: int = 12_000,
    output_path: Optional[Path] = None,
    seed: int = 42,
    include_handlabeled: bool = False,
    handlabeled_repeat: int = 80,
    include_dataturks: bool = False,
    dataturks_repeat: int = 1,
    include_mehyaar: bool = False,
    mehyaar_repeat: int = 1,
    mehyaar_max: int = 5000,
) -> Dict:
    """Generate a v7 NER dataset.

    Composition (all optional except synthetic):
      - synthetic: `n_samples` O*NET-grounded generator rows
      - hand-labeled: 35 unique energy resumes × `handlabeled_repeat`
      - dataturks aux: ~200 IT-domain resumes (v5/v6-style 5-type gold),
        re-annotated via v7 vocab for cross-domain diversity
      - mehyaar aux: up to `mehyaar_max` IT CVs (SKILL-only gold from
        Mehyaar/Annotated_NER_PDF_Resumes, MIT), re-annotated for v7.
        Biggest volume boost; keep at repeat=1 to avoid dominating the
        energy-focused signal.

    Held-out exclusion is driven by `data/processed/eval_held_out.json`
    so the train / eval partition stays clean without hardcoded indices.
    """
    synthetic, _soc_codes = _synthetic_samples(n_samples, seed)
    data = list(synthetic)
    if include_handlabeled:
        hl = hand_labeled_samples_v7(repeat=handlabeled_repeat)
        data.extend(hl)
    if include_dataturks:
        dt = dataturks_aux_samples_v7(repeat=dataturks_repeat)
        data.extend(dt)
    if include_mehyaar:
        mh = mehyaar_aux_samples_v7(
            repeat=mehyaar_repeat, max_samples=mehyaar_max,
        )
        data.extend(mh)
    if include_handlabeled or include_dataturks or include_mehyaar:
        rng = random.Random(seed + 1)
        rng.shuffle(data)

    stats = _summarize(data)
    if output_path:
        _write_dataset(data, Path(output_path))
    return stats


if __name__ == "__main__":
    # 1) Synthetic-only — kept as a pure-O*NET traceability artifact.
    # n_samples=14,850 = 550/SOC × 27 SOCs (restores per-SOC density that
    # diluted when we expanded from 22 → 27 SOCs at the prior 12,000 budget).
    generate_dataset(
        n_samples=14_850,
        output_path=PROC_DIR / "ner_training_v7_synthetic.json",
        seed=42,
        include_handlabeled=False,
    )

    # 2) Hand-labeled only — 35 unique resumes × 80 repeats (5 held-out excluded).
    hl = hand_labeled_samples_v7(repeat=80)
    _write_dataset(hl, PROC_DIR / "ner_training_v7_handlabeled.json")

    # 3) Dataturks aux — ~200 unique × 1 repeat (20 held-out excluded).
    dt = dataturks_aux_samples_v7(repeat=1)
    if dt:
        _write_dataset(dt, PROC_DIR / "ner_training_v7_dataturks.json")

    # 4) Mehyaar aux — 1,000 IT CVs subsampled from the 5,029-CV corpus
    #    (Mehyaar/Annotated_NER_PDF_Resumes, MIT license). Full corpus has
    #    ~200k SKILL spans which would swamp the energy-specific signal
    #    (~30k). Subsampling to 1,000 CVs keeps the energy/IT token ratio
    #    at ~65/35 — enough SKILL diversity boost without diluting the
    #    energy-specific terminology learning.
    MEHYAAR_N = 1000
    mh = mehyaar_aux_samples_v7(repeat=1, max_samples=MEHYAAR_N)
    if mh:
        _write_dataset(mh, PROC_DIR / "ner_training_v7_mehyaar.json")

    # 5) Combined — the primary training file. Overwrites the Step A output
    #    on disk. v6's ner_training_expanded.json is NOT touched.
    generate_dataset(
        n_samples=14_850,
        output_path=DEFAULT_OUTPUT,
        seed=42,
        include_handlabeled=True,
        handlabeled_repeat=80,
        include_dataturks=True,
        dataturks_repeat=1,
        include_mehyaar=True,
        mehyaar_repeat=1,
        mehyaar_max=MEHYAAR_N,
    )
