"""
Comprehensive Energy Industry Taxonomy for NER Training Data Generation.

Sources:
  - O*NET OnLine (SOC 17-2041, 17-2071, 17-2141, 11-3051)
  - API Individual Certification Programs (api.org)
  - NACE International / AMPP certifications
  - ISA (International Society of Automation) certifications
  - NCCER (National Center for Construction Education & Research)
  - OSHA safety certifications
  - IEEE / NFPA standards
  - ESCO European Skills/Competences taxonomy
  - BLS Occupational Outlook Handbook — Energy sector

This taxonomy powers the synthetic NER training data generator.
~200 skills, ~50 certifications, ~30 degrees, ~60 employers, ~20 templates.
"""

# ═══════════════════════════════════════════════════════════════════════
# SKILLS (~200 entries across energy sub-sectors)
# ═══════════════════════════════════════════════════════════════════════

SKILLS_PROCESS = [
    "P&ID review", "HAZOP facilitation", "process safety management",
    "process simulation", "heat exchanger design", "pressure vessel inspection",
    "control valve sizing", "distillation column design", "fluid dynamics analysis",
    "mass and energy balance", "thermodynamic modeling", "reactor design",
    "flare system design", "relief valve sizing", "piping stress analysis",
    "process flow diagram development", "material selection", "corrosion engineering",
    "chemical injection systems", "desalination process design",
]

SKILLS_ELECTRICAL = [
    "relay protection", "arc flash analysis", "load flow studies",
    "power systems analysis", "electrical commissioning", "motor control design",
    "transformer testing", "switchgear maintenance", "cable sizing and routing",
    "grounding system design", "power factor correction", "harmonic analysis",
    "substation design", "transmission line design", "distribution system planning",
    "short circuit analysis", "coordination study", "battery storage systems",
    "EV charging infrastructure", "microgrid design", "protective relay programming",
]

SKILLS_INSTRUMENTATION = [
    "SCADA integration", "DCS programming", "PLC programming",
    "instrumentation calibration", "control loop tuning", "HMI development",
    "fieldbus configuration", "SIS design", "functional safety assessment",
    "cybersecurity for OT networks", "fiber optic sensing", "flow measurement",
    "level measurement", "pressure transmitter calibration", "RTD calibration",
    "control narrative development", "P&ID markup for instrumentation",
    "alarm management", "advanced process control", "model predictive control",
]

SKILLS_MECHANICAL = [
    "turbine maintenance", "vibration analysis", "NDT testing",
    "welding inspection", "rotating equipment alignment", "pump selection",
    "compressor maintenance", "heat recovery steam generator maintenance",
    "gas turbine performance monitoring", "reciprocating engine maintenance",
    "centrifugal pump troubleshooting", "bearing failure analysis",
    "mechanical seal replacement", "valve maintenance", "crane inspection",
    "rigging and lifting", "precision alignment", "predictive maintenance",
    "reliability centered maintenance", "root cause failure analysis",
]

SKILLS_PROJECT = [
    "project management", "budget forecasting", "vendor management",
    "shutdown planning", "turnaround execution", "brownfield engineering",
    "greenfield design", "FEED studies", "detailed engineering",
    "construction management", "commissioning and startup",
    "procurement management", "contract negotiation", "stakeholder alignment",
    "schedule development", "earned value management", "risk register management",
    "change order management", "scope definition", "project controls",
]

SKILLS_HSE = [
    "risk assessment", "MOC management", "SIL verification",
    "incident investigation", "safety culture development",
    "permit to work systems", "confined space entry procedures",
    "lockout tagout procedures", "emergency response planning",
    "environmental impact assessment", "air quality monitoring",
    "wastewater treatment oversight", "hazardous waste management",
    "behavioral based safety", "job safety analysis", "ergonomic assessment",
    "noise monitoring", "radiation protection", "fire protection engineering",
    "process hazard analysis",
]

SKILLS_OFFSHORE = [
    "offshore operations", "subsea engineering", "drilling optimization",
    "well completion design", "production optimization", "reservoir simulation",
    "artificial lift design", "well intervention", "marine operations",
    "dynamic positioning", "pipeline integrity", "riser analysis",
    "mooring system design", "topside process design", "FPSO operations",
    "jack-up rig operations", "deepwater drilling", "subsea tieback design",
    "flow assurance", "hydrate management",
]

SKILLS_RENEWABLES = [
    "renewable energy integration", "solar panel installation",
    "wind turbine commissioning", "solar farm design", "wind resource assessment",
    "energy storage design", "hydrogen production", "carbon capture",
    "geothermal system design", "biomass energy systems",
    "solar inverter commissioning", "wind turbine blade inspection",
    "grid interconnection studies", "power purchase agreement analysis",
    "renewable energy forecasting", "BESS commissioning",
    "green hydrogen electrolysis", "carbon sequestration monitoring",
    "lifecycle assessment", "sustainability reporting",
]

SKILLS_MIDSTREAM = [
    "LNG processing", "natural gas compression", "pipeline design",
    "pipeline integrity management", "gas processing plant operations",
    "NGL fractionation", "gas dehydration", "amine treating",
    "sulfur recovery", "metering and custody transfer",
    "pig launcher and receiver operations", "cathodic protection",
    "inline inspection", "leak detection systems", "SCADA for pipelines",
    "compressor station design", "tank farm operations",
    "truck loading and unloading", "rail terminal operations",
    "blending and batch operations",
]

SKILLS_SOFTWARE = [
    "Python", "MATLAB", "AutoCAD", "HYSYS", "Aspen Plus",
    "ETAP", "SKM PowerTools", "SAP PM", "Maximo", "SQL",
    "C++", "R programming", "LabVIEW", "Simulink", "SolidWorks",
    "CAESAR II", "PVElite", "COMSOL Multiphysics", "ANSYS",
    "Primavera P6", "Microsoft Project", "Power BI", "Tableau",
    "OSIsoft PI", "Aveva E3D", "SmartPlant 3D", "MicroStation",
    "PDMS", "Navisworks", "Revit",
]

SKILLS_DATA = [
    "data analytics", "machine learning", "predictive modeling",
    "digital twin development", "IoT sensor integration",
    "cloud computing for energy", "edge computing",
    "data visualization", "statistical process control",
    "regression analysis",
]

# Combined skills list
SKILLS = (
    SKILLS_PROCESS + SKILLS_ELECTRICAL + SKILLS_INSTRUMENTATION +
    SKILLS_MECHANICAL + SKILLS_PROJECT + SKILLS_HSE + SKILLS_OFFSHORE +
    SKILLS_RENEWABLES + SKILLS_MIDSTREAM + SKILLS_SOFTWARE + SKILLS_DATA
)

# ═══════════════════════════════════════════════════════════════════════
# CERTIFICATIONS (~50 entries)
# ═══════════════════════════════════════════════════════════════════════

CERTIFICATIONS = [
    # API (American Petroleum Institute)
    "API 510 certified", "API 570 certified", "API 650 certified",
    "API 653 certified", "API 580 certified", "API 1169 certified",
    "API 571 certified", "API 577 certified",
    # Professional Engineering
    "PE license", "FE certification", "P.Eng license",
    # Project Management
    "PMP certified", "PRINCE2 Practitioner", "PMI-RMP certified",
    # Safety & Environmental
    "OSHA 30", "OSHA 10", "NEBOSH IGC certified", "NEBOSH Diploma",
    "CSP certified", "CIH certified", "CHMM certified",
    "TÜV Functional Safety Engineer", "CFSE certified",
    # Quality & Reliability
    "Six Sigma Black Belt", "Six Sigma Green Belt",
    "Lean Manufacturing certified", "CRE certified",
    "ISO 9001 Lead Auditor", "ISO 14001 Lead Auditor",
    "ISO 45001 Lead Auditor",
    # Electrical & Instrumentation
    "ISA CCST", "ISA CAP certified", "NICET Level III",
    "CompTIA Security+", "LEED AP", "CEM certified",
    # Welding & Inspection
    "AWS CWI", "AWS CWE", "NACE CIP Level 1", "NACE CIP Level 2",
    "NACE CIP Level 3", "ASNT NDT Level II", "ASNT NDT Level III",
    "PCN Level 2",
    # Construction & Trades
    "NCCER certified", "Master Electrician license",
    "Journeyman Electrician license", "Crane Operator certification",
    # Renewable Energy
    "NABCEP PV Installation Professional",
    "GWO Basic Safety Training",
    "Certified Energy Manager",
]

# ═══════════════════════════════════════════════════════════════════════
# DEGREES (~30 entries)
# ═══════════════════════════════════════════════════════════════════════

DEGREES = [
    "B.S. in Mechanical Engineering", "M.S. in Mechanical Engineering",
    "B.S. in Electrical Engineering", "M.S. in Electrical Engineering",
    "B.S. in Chemical Engineering", "M.S. in Chemical Engineering",
    "Ph.D. in Chemical Engineering", "Ph.D. in Petroleum Engineering",
    "B.S. in Petroleum Engineering", "M.S. in Petroleum Engineering",
    "B.S. in Civil Engineering", "M.S. in Civil Engineering",
    "B.S. in Industrial Engineering", "M.S. in Industrial Engineering",
    "B.S. in Computer Science", "M.S. in Computer Science",
    "B.S. in Environmental Science", "M.S. in Environmental Science",
    "B.S. in Power Engineering", "M.S. in Energy Systems",
    "B.S. in Nuclear Engineering", "M.S. in Nuclear Engineering",
    "B.S. in Materials Science", "M.S. in Materials Science",
    "Associate in Process Technology", "Associate in Instrumentation Technology",
    "MBA", "Executive MBA",
    "B.S. in Physics", "M.S. in Applied Mathematics",
]

# ═══════════════════════════════════════════════════════════════════════
# EMPLOYERS (~60 entries — major energy companies globally)
# ═══════════════════════════════════════════════════════════════════════

EMPLOYERS = [
    # Supermajors & IOCs
    "ExxonMobil", "Chevron", "Shell", "BP", "TotalEnergies",
    "ConocoPhillips", "Eni", "Equinor", "Repsol",
    # Oilfield Services
    "Schlumberger", "Baker Hughes", "Halliburton", "Weatherford",
    "NOV", "TechnipFMC", "Saipem",
    # EPC Contractors
    "Bechtel", "Fluor Corporation", "KBR", "Worley", "Wood PLC",
    "McDermott International", "Jacobs Engineering", "AECOM",
    "Kiewit", "Zachry Group",
    # Power & Utilities
    "NextEra Energy", "Duke Energy", "Southern Company",
    "Dominion Energy", "AES Corporation", "Sempra Energy",
    "Vistra Corp", "NRG Energy", "Entergy", "Exelon",
    # Technology & Automation
    "ABB Energy Industries", "Siemens Energy", "GE Vernova",
    "Honeywell UOP", "Emerson Electric", "Yokogawa",
    "Schneider Electric", "Rockwell Automation", "Endress+Hauser",
    # Midstream & Pipeline
    "Enbridge", "Enterprise Products Partners", "Williams Companies",
    "Kinder Morgan", "ONEOK", "Plains All American Pipeline",
    # Renewables & Clean Energy
    "Vestas", "Ørsted", "First Solar", "Enphase Energy",
    "Sunrun", "Brookfield Renewable", "Pattern Energy",
    # National Oil Companies
    "Saudi Aramco", "ADNOC", "Petrobras", "Petronas",
]

# ═══════════════════════════════════════════════════════════════════════
# YEARS OF EXPERIENCE EXPRESSIONS
# ═══════════════════════════════════════════════════════════════════════

YEARS_EXPRESSIONS = [
    "{n} years of experience", "{n}+ years in the field",
    "{n} years of hands-on experience", "over {n} years",
    "{n} years working in", "{n}-year track record",
    "{n} years in the industry", "more than {n} years",
    "{n} years of progressive experience",
    "{n} years of combined experience",
]

# ═══════════════════════════════════════════════════════════════════════
# SENTENCE TEMPLATES (20 diverse patterns)
# ═══════════════════════════════════════════════════════════════════════

SENTENCE_TEMPLATES = [
    # Standard paragraph style
    "{years} in oil and gas operations. Currently at {employer} as Lead Engineer. Expertise in {skill}, {skill2}, and {skill3}. Certifications: {cert}. Education: {degree}.",
    "{years} in power systems and renewable energy. Working at {employer} as Senior Engineer. Skilled in {skill}, {skill2}, and {skill3}. Certifications: {cert}. Education: {degree}.",
    "{years} in industrial maintenance. Employed at {employer} as Supervisor. Core skills: {skill}, {skill2}, {skill3}. Certifications: {cert}. Education: {degree}.",
    "{years} in health, safety, and environment. Currently at {employer}. Expertise in {skill}, {skill2}, and {skill3}. Certifications: {cert}. Education: {degree}.",
    "{years} in control systems and automation. Working at {employer} as Specialist. Skills: {skill}, {skill2}, {skill3}. Certifications: {cert}. Education: {degree}.",
    "{years} in capital projects for energy. Currently at {employer} as Director. Expert in {skill}, {skill2}, and {skill3}. Certifications: {cert}. Education: {degree}.",
    # Two-employer pattern
    "Previously at {employer} for {years}. Proficient in {skill} and {skill2}. Holds {cert}. Completed {degree}.",
    "Led {skill} projects at {employer}. {years} of combined experience. Also experienced in {skill2} and {skill3}. {cert}. {degree} graduate.",
    # Skills-first pattern
    "Expertise in {skill} and {skill2}. {years} at {employer}. Certifications: {cert}. Education: {degree}.",
    "Core competencies include {skill}, {skill2}, and {skill3}. {years} at {employer}. {cert}. Education: {degree}.",
    "Skilled in {skill} and {skill2}. Familiar with {skill3}. {years}. Employed at {employer}. Certifications: {cert}. Education: {degree}.",
    # Employer-first pattern
    "Engineer at {employer} with {years}. Specializing in {skill} and {skill2}. Holds {cert} and a {degree}.",
    "Currently employed at {employer} with {years}. Primary focus: {skill}, {skill2}, {skill3}. Certified: {cert}. Education: {degree}.",
    # Mixed patterns
    "{degree} holder with {cert}. {years} at {employer} performing {skill} and {skill2}.",
    "Managed {skill} across multiple facilities for {employer}. {years}. Certified: {cert}. Education: {degree}.",
    # Renewable/modern energy
    "{years} in clean energy and sustainability. At {employer} leading {skill} and {skill2} initiatives. {cert}. {degree}.",
    "Offshore specialist with {years} at {employer}. Expert in {skill}, {skill2}. Holds {cert}. {degree}.",
    # Short/concise patterns
    "{skill} engineer. {years} at {employer}. {cert}. {degree}.",
    "Senior {skill} specialist at {employer}. {years}. Also skilled in {skill2} and {skill3}. {cert}. {degree}.",
    "{years} spanning upstream, midstream, and downstream operations at {employer}. Key skills: {skill}, {skill2}, {skill3}. {cert}. {degree}.",
]

# ═══════════════════════════════════════════════════════════════════════
# ROLE TITLES (for context in templates)
# ═══════════════════════════════════════════════════════════════════════

ROLE_TITLES = [
    "Senior Process Engineer", "Lead Electrical Engineer",
    "Control Systems Engineer", "Maintenance Supervisor",
    "HSE Manager", "Project Director", "SCADA Specialist",
    "Pipeline Integrity Engineer", "Reliability Engineer",
    "Commissioning Manager", "Instrument Technician",
    "Operations Manager", "Plant Manager", "Field Engineer",
    "Rotating Equipment Engineer", "Corrosion Engineer",
    "Drilling Engineer", "Completions Engineer",
    "Renewable Energy Engineer", "Grid Operations Analyst",
    "Process Safety Consultant", "Environmental Engineer",
    "Subsea Engineer", "Reservoir Engineer",
    "Energy Storage Engineer", "Hydrogen Systems Engineer",
]
