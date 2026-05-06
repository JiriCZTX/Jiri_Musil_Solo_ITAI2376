"""
Sample resume texts and job descriptions for demo and testing.

Two demo corpora:
  - SAMPLE_RESUMES / SAMPLE_JOB_DESCRIPTIONS (Energy, 28 + 9)
  - SAMPLE_RESUMES_FINANCE / SAMPLE_JOB_DESCRIPTIONS_FINANCE (Finance, 8 + 5)

Use `get_active_resumes()` / `get_active_jds()` to pull the corpus that
matches the currently-active industry profile. Module-level constants
SAMPLE_RESUMES / SAMPLE_JOB_DESCRIPTIONS continue to point at Energy
for backward compatibility with main.py / agents.py / brain.py call sites.
"""

SAMPLE_RESUMES = [
    {
        "id": "R001",
        "name": "Carlos Mendez",
        "text": """Carlos Mendez — Senior Process Engineer
        12 years of experience in oil and gas operations.
        Currently at Chevron as Lead Process Engineer.
        Expertise in P&ID review, HAZOP facilitation, and process safety management.
        Led turnaround execution for a 150,000 bpd refinery unit.
        Proficient in HYSYS, AutoCAD, and Python for process simulation.
        Certifications: PE license, API 570 certified, Six Sigma Black Belt.
        Education: M.S. in Chemical Engineering from Texas A&M University.
        Previously at Shell for 5 years in upstream operations."""
    },
    {
        "id": "R002",
        "name": "Sarah Chen",
        "text": """Sarah Chen — Electrical Engineer
        8 years of experience in power systems and renewable energy.
        Working at NextEra Energy as Senior Electrical Engineer.
        Skilled in relay protection, arc flash analysis, and load flow studies.
        Managed electrical commissioning for 3 solar farms (total 450 MW).
        Tools: ETAP, SKM PowerTools, MATLAB, Python.
        Certifications: PE license, LEED AP.
        Education: B.S. in Electrical Engineering from University of Houston.
        Prior experience at Siemens Energy in gas turbine commissioning."""
    },
    {
        "id": "R003",
        "name": "James Rodriguez",
        "text": """James Rodriguez — Maintenance Supervisor
        15 years of hands-on experience in industrial maintenance.
        Employed at Baker Hughes as Maintenance Planning Supervisor.
        Core skills: vibration analysis, NDT testing, welding inspection.
        Implemented predictive maintenance program reducing downtime by 30%.
        Familiar with SAP PM, Maximo, and PLC programming.
        Certifications: AWS CWI, NACE CIP Level 2.
        Education: Associate in Process Technology from San Jacinto College.
        Previously at Halliburton for 8 years."""
    },
    {
        "id": "R004",
        "name": "Aisha Patel",
        "text": """Aisha Patel — HSE Manager
        10 years of experience in health, safety, and environment.
        Currently HSE Director at TotalEnergies Americas.
        Expertise in risk assessment, MOC management, and SIL verification.
        Led safety culture transformation across 4 offshore platforms.
        Reduced recordable incident rate by 45% over 3 years.
        Certifications: NEBOSH certified, CSP certified, OSHA 30.
        Education: B.S. in Environmental Science from Rice University.
        M.S. in Energy Systems from University of Texas at Austin."""
    },
    {
        "id": "R005",
        "name": "Michael Thompson",
        "text": """Michael Thompson — SCADA Engineer
        6 years of experience in control systems and automation.
        Working at Emerson Electric as Senior SCADA Specialist.
        Skills: SCADA integration, DCS programming, PLC programming,
        instrumentation calibration, and cybersecurity for OT networks.
        Deployed SCADA upgrade across 12 natural gas compression stations.
        Certifications: ISA CCST, CompTIA Security+.
        Education: B.S. in Computer Science from University of Houston.
        Python, SQL, and C++ proficiency."""
    },
    {
        "id": "R006",
        "name": "Elena Vasquez",
        "text": """Elena Vasquez — Project Manager
        20 years of experience in capital projects for energy.
        Currently at Bechtel as Senior Project Director.
        Managed $2.1B LNG export terminal construction project.
        Expert in shutdown planning, brownfield engineering, and FEED studies.
        Skilled in budget forecasting, vendor management, and stakeholder alignment.
        Certifications: PMP certified, PE license, Lean Manufacturing certified.
        Education: B.S. in Civil Engineering, MBA from Rice University."""
    },
    # --- Offshore / Subsea ---
    {
        "id": "R007",
        "name": "David Okafor",
        "text": """David Okafor — Subsea Engineer
        14 years of experience in offshore operations and subsea engineering.
        Currently at TechnipFMC as Principal Subsea Systems Engineer.
        Expertise in subsea tieback design, flow assurance, and riser analysis.
        Designed subsea production system for deepwater field at 7,500 ft depth.
        Proficient in OLGA, Pipesim, and ANSYS for subsea analysis.
        Certifications: PE license, API 1169 certified.
        Education: M.S. in Petroleum Engineering from University of Houston.
        Previously at Saipem for 6 years in marine operations."""
    },
    {
        "id": "R008",
        "name": "Karen Bjornsen",
        "text": """Karen Bjornsen — Offshore Operations Manager
        18 years of experience in offshore oil and gas production.
        Employed at Equinor as Operations Director for Gulf of Mexico assets.
        Led FPSO operations, dynamic positioning, and topside process design reviews.
        Managed production optimization across 3 deepwater platforms.
        Skills include well intervention, mooring system design, and marine operations.
        Certifications: NEBOSH Diploma, OSHA 30, PMP certified.
        Education: B.S. in Mechanical Engineering from Texas A&M University.
        Previous role at BP in jack-up rig operations for 7 years."""
    },
    # --- Renewables (Wind) ---
    {
        "id": "R009",
        "name": "Lars Eriksson",
        "text": """Lars Eriksson — Wind Energy Engineer
        9 years of experience in wind energy development and commissioning.
        Working at Vestas as Senior Wind Turbine Engineer.
        Skilled in wind resource assessment, wind turbine commissioning,
        and wind turbine blade inspection.
        Led commissioning of 300 MW onshore wind farm in West Texas.
        Tools: WindPRO, WAsP, SCADA for pipelines, Python.
        Certifications: GWO Basic Safety Training, PE license.
        Education: M.S. in Mechanical Engineering from Technical University of Denmark.
        Prior experience at Ørsted in offshore wind development."""
    },
    # --- Renewables (Solar / Storage) ---
    {
        "id": "R010",
        "name": "Priya Sharma",
        "text": """Priya Sharma — Solar Energy Project Engineer
        5 years of experience in solar energy and battery storage systems.
        Currently at First Solar as Project Engineer.
        Expertise in solar farm design, solar inverter commissioning,
        and BESS commissioning for utility-scale projects.
        Managed grid interconnection studies for 200 MW solar-plus-storage facility.
        Proficient in PVsyst, ETAP, and AutoCAD.
        Certifications: NABCEP PV Installation Professional, FE certification.
        Education: B.S. in Electrical Engineering from Arizona State University."""
    },
    # --- Midstream / Pipeline ---
    {
        "id": "R011",
        "name": "Robert Tran",
        "text": """Robert Tran — Pipeline Integrity Engineer
        11 years of experience in midstream operations and pipeline integrity.
        Working at Enterprise Products Partners as Senior Integrity Engineer.
        Expert in pipeline integrity management, inline inspection,
        and cathodic protection systems.
        Managed integrity assessment program covering 2,500 miles of NGL pipeline.
        Skills: pig launcher and receiver operations, leak detection systems, SCADA for pipelines.
        Certifications: NACE CIP Level 3, API 570 certified, ASNT NDT Level II.
        Education: B.S. in Mechanical Engineering from University of Texas at Austin.
        Previously at Kinder Morgan for 5 years."""
    },
    {
        "id": "R012",
        "name": "Amanda Whitfield",
        "text": """Amanda Whitfield — Gas Processing Engineer
        7 years of experience in natural gas processing and LNG operations.
        Employed at Williams Companies as Process Engineer.
        Skilled in gas dehydration, amine treating, NGL fractionation,
        and sulfur recovery unit operations.
        Optimized LNG processing throughput by 15% at Transco Station 165.
        Tools: HYSYS, Aspen Plus, OSIsoft PI.
        Certifications: PE license, Six Sigma Green Belt.
        Education: B.S. in Chemical Engineering from Louisiana State University.
        Prior role at Enbridge in natural gas compression."""
    },
    # --- Nuclear ---
    {
        "id": "R013",
        "name": "Gregory Park",
        "text": """Gregory Park — Nuclear Systems Engineer
        16 years of experience in nuclear power plant operations and safety.
        Currently at Exelon as Lead Nuclear Engineer.
        Expertise in reactor design, radiation protection, and functional safety assessment.
        Led SIS design upgrade for emergency core cooling system.
        Proficient in RELAP5, MATLAB, and Simulink for nuclear modeling.
        Certifications: PE license, TÜV Functional Safety Engineer.
        Education: M.S. in Nuclear Engineering from MIT.
        Previous experience at GE Vernova in boiling water reactor design for 8 years."""
    },
    # --- Drilling ---
    {
        "id": "R014",
        "name": "Hassan Al-Rashidi",
        "text": """Hassan Al-Rashidi — Senior Drilling Engineer
        13 years of experience in drilling operations and well design.
        Working at Saudi Aramco as Drilling Superintendent.
        Expert in drilling optimization, well completion design, and deepwater drilling.
        Managed 40+ well drilling campaigns in the Arabian Gulf.
        Skilled in reservoir simulation and artificial lift design.
        Certifications: API 510 certified, NEBOSH IGC certified, OSHA 30.
        Education: B.S. in Petroleum Engineering from King Fahd University.
        Previously at Schlumberger for 5 years in directional drilling services."""
    },
    # --- Entry-Level ---
    {
        "id": "R015",
        "name": "Jessica Liu",
        "text": """Jessica Liu — Junior Instrumentation Engineer
        2 years of experience in control systems and instrumentation.
        Entry-level engineer at Honeywell UOP.
        Skills: control loop tuning, HMI development, fieldbus configuration.
        Assisted with DCS programming for a refinery modernization project.
        Proficient in LabVIEW, Python, and SQL.
        Certifications: FE certification.
        Education: B.S. in Electrical Engineering from University of Houston.
        Completed internship at Yokogawa in process control."""
    },
    {
        "id": "R016",
        "name": "Tyler Washington",
        "text": """Tyler Washington — Field Technician
        3 years of hands-on experience in industrial instrumentation.
        Currently at Endress+Hauser as Field Service Technician.
        Core skills: pressure transmitter calibration, RTD calibration,
        flow measurement, and level measurement.
        Performs instrumentation calibration at refineries and chemical plants.
        Certifications: NCCER certified, OSHA 10.
        Education: Associate in Instrumentation Technology from Lee College."""
    },
    # --- Executive ---
    {
        "id": "R017",
        "name": "Margaret Sullivan",
        "text": """Margaret Sullivan — VP of Engineering
        25 years of experience in energy industry leadership.
        Currently Vice President of Engineering at ConocoPhillips.
        Led global capital project portfolio worth $4.8B annually.
        Expert in stakeholder alignment, earned value management, and risk register management.
        Drove digital twin development initiative across 12 production assets.
        Certifications: PMP certified, PE license, ISO 9001 Lead Auditor.
        Education: B.S. in Chemical Engineering from Stanford University.
        Executive MBA from Wharton School of Business.
        Previous roles at ExxonMobil and Shell spanning upstream and downstream."""
    },
    {
        "id": "R018",
        "name": "Richard Nakamura",
        "text": """Richard Nakamura — Chief Technology Officer
        22 years of experience in energy technology and digital transformation.
        CTO at ABB Energy Industries overseeing global automation strategy.
        Led deployment of machine learning and predictive modeling for
        predictive maintenance across 200+ industrial facilities.
        Expert in IoT sensor integration, cloud computing for energy, and digital twin development.
        Skilled in advanced process control and model predictive control.
        Certifications: PE license, ISA CAP certified, Certified Energy Manager.
        Education: Ph.D. in Chemical Engineering from Caltech.
        Prior experience at Schneider Electric and Rockwell Automation."""
    },
    # --- Corrosion / Materials ---
    {
        "id": "R019",
        "name": "Fatima Al-Dosari",
        "text": """Fatima Al-Dosari — Corrosion Engineer
        8 years of experience in corrosion engineering and materials selection.
        Working at ADNOC as Senior Corrosion and Materials Engineer.
        Expert in corrosion engineering, material selection, and cathodic protection.
        Managed chemical injection systems for 15 offshore wellhead platforms.
        Skilled in pressure vessel inspection and piping stress analysis.
        Certifications: NACE CIP Level 2, API 571 certified, API 577 certified.
        Education: M.S. in Materials Science from Imperial College London.
        Previously at Petrobras in subsea corrosion management for 3 years."""
    },
    # --- Reliability ---
    {
        "id": "R020",
        "name": "Brian Cooper",
        "text": """Brian Cooper — Reliability Engineer
        10 years of experience in asset reliability and maintenance optimization.
        Employed at Fluor Corporation as Lead Reliability Engineer.
        Skills: reliability centered maintenance, root cause failure analysis,
        bearing failure analysis, and rotating equipment alignment.
        Implemented predictive maintenance program saving $3.2M annually.
        Tools: SAP PM, Maximo, OSIsoft PI, Power BI.
        Certifications: CRE certified, Six Sigma Black Belt, ASNT NDT Level II.
        Education: B.S. in Mechanical Engineering from Purdue University.
        Previous role at NOV in rotating equipment engineering."""
    },
    # --- Hydrogen / Clean Energy ---
    {
        "id": "R021",
        "name": "Sophie Durand",
        "text": """Sophie Durand — Hydrogen Systems Engineer
        4 years of experience in clean energy and hydrogen technology.
        Currently at Enphase Energy as Hydrogen Projects Lead.
        Expertise in green hydrogen electrolysis, hydrogen production,
        and energy storage design for industrial applications.
        Led feasibility study for 50 MW green hydrogen facility.
        Skills: lifecycle assessment, sustainability reporting, carbon capture.
        Certifications: LEED AP, Certified Energy Manager.
        Education: M.S. in Energy Systems from ETH Zurich.
        Prior research at National Renewable Energy Laboratory."""
    },
    # --- Data / Digital ---
    {
        "id": "R022",
        "name": "Kevin Okonkwo",
        "text": """Kevin Okonkwo — Energy Data Scientist
        6 years of experience in data analytics for the energy sector.
        Working at Chevron as Senior Data Scientist.
        Expert in machine learning, predictive modeling, and data visualization.
        Built real-time production forecasting model using Python and R programming.
        Skills: statistical process control, regression analysis, digital twin development.
        Proficient in Tableau, Power BI, SQL, and cloud computing for energy.
        Certifications: Six Sigma Green Belt, CompTIA Security+.
        Education: M.S. in Computer Science from Georgia Tech.
        Previously at Halliburton in IoT sensor integration analytics."""
    },
    # --- Commissioning ---
    {
        "id": "R023",
        "name": "Anthony Reeves",
        "text": """Anthony Reeves — Commissioning Manager
        12 years of experience in commissioning and startup for energy facilities.
        Currently at Wood PLC as Commissioning Lead.
        Skilled in commissioning and startup, electrical commissioning,
        and solar inverter commissioning.
        Managed pre-commissioning through mechanical completion for
        800 MW combined-cycle gas turbine power plant.
        Tools: Primavera P6, Microsoft Project, SmartPlant 3D.
        Certifications: PMP certified, NICET Level III, PE license.
        Education: B.S. in Electrical Engineering from Virginia Tech.
        Prior role at KBR in LNG facility commissioning."""
    },
    # --- Environmental ---
    {
        "id": "R024",
        "name": "Maria Gonzalez",
        "text": """Maria Gonzalez — Environmental Engineer
        9 years of experience in environmental compliance for energy operations.
        Working at Sempra Energy as Lead Environmental Specialist.
        Expertise in environmental impact assessment, air quality monitoring,
        wastewater treatment oversight, and hazardous waste management.
        Led environmental permitting for 3 natural gas power plants.
        Skills: carbon sequestration monitoring, sustainability reporting.
        Certifications: CHMM certified, ISO 14001 Lead Auditor, OSHA 30.
        Education: M.S. in Environmental Science from UC Berkeley.
        Previously at AES Corporation for 4 years."""
    },
    # --- Power / Grid Operations ---
    {
        "id": "R025",
        "name": "Daniel Kim",
        "text": """Daniel Kim — Grid Operations Engineer
        7 years of experience in grid operations and power distribution.
        Employed at Duke Energy as Senior Grid Analyst.
        Skilled in distribution system planning, power factor correction,
        harmonic analysis, and substation design.
        Managed grid interconnection studies for 500 MW of distributed solar.
        Tools: ETAP, Python, MATLAB, Power BI.
        Certifications: PE license, CEM certified.
        Education: M.S. in Electrical Engineering from North Carolina State University.
        Prior role at Southern Company in transmission line design."""
    },
    # --- Construction / Trades ---
    {
        "id": "R026",
        "name": "Marcus Johnson",
        "text": """Marcus Johnson — Master Electrician
        17 years of hands-on experience in industrial electrical systems.
        Currently at Zachry Group as Electrical Foreman.
        Core skills: switchgear maintenance, transformer testing,
        cable sizing and routing, and grounding system design.
        Led electrical installation for petrochemical plant expansion.
        Experienced in crane inspection, rigging and lifting operations.
        Certifications: Master Electrician license, OSHA 30, NCCER certified.
        Education: Journeyman Electrician license from IBEW Local 716.
        Previously at Kiewit for 9 years in heavy industrial construction."""
    },
    # --- Reservoir Engineering ---
    {
        "id": "R027",
        "name": "Olga Petrova",
        "text": """Olga Petrova — Reservoir Engineer
        11 years of experience in reservoir simulation and production optimization.
        Working at Petronas as Senior Reservoir Engineer.
        Expert in reservoir simulation, production optimization,
        and well completion design for complex carbonate reservoirs.
        Increased field recovery factor by 8% through waterflood optimization.
        Skills: artificial lift design, flow assurance, hydrate management.
        Tools: Eclipse, Petrel, Python, MATLAB.
        Certifications: PE license, Six Sigma Green Belt.
        Education: Ph.D. in Petroleum Engineering from Colorado School of Mines.
        Previous role at Repsol in enhanced oil recovery."""
    },
    # --- Fire Protection / Safety Specialist ---
    {
        "id": "R028",
        "name": "Patrick O'Brien",
        "text": """Patrick O'Brien — Fire Protection Engineer
        8 years of experience in fire protection engineering for energy facilities.
        Currently at Jacobs Engineering as Senior Fire Safety Engineer.
        Expertise in fire protection engineering, emergency response planning,
        and process hazard analysis.
        Designed fire suppression systems for 4 offshore production platforms.
        Skilled in permit to work systems and confined space entry procedures.
        Certifications: CFSE certified, PE license, NEBOSH IGC certified.
        Education: B.S. in Mechanical Engineering from University of Maryland.
        Prior experience at McDermott International in safety engineering."""
    },
]

SAMPLE_JOB_DESCRIPTIONS = [
    {
        "id": "JD001",
        "title": "Senior Process Safety Engineer",
        "text": """Senior Process Safety Engineer — Energy Operations
        Location: Houston, TX
        We are seeking a Senior Process Safety Engineer to lead HAZOP
        facilitation and process safety management programs for our
        refining operations. The ideal candidate will have 8+ years
        of experience in oil and gas, strong P&ID review skills,
        and familiarity with API standards. PE license required.
        Must be proficient in risk assessment and MOC management.
        Experience with turnaround execution is a plus.
        Education: B.S. or M.S. in Chemical or Mechanical Engineering."""
    },
    {
        "id": "JD002",
        "title": "Control Systems Engineer",
        "text": """Control Systems Engineer — Midstream Gas Operations
        Location: Midland, TX
        Looking for a Control Systems Engineer with experience in
        SCADA integration, DCS programming, and PLC programming.
        Must have hands-on experience with instrumentation calibration.
        The role involves deploying control system upgrades across
        natural gas compression and processing facilities.
        5+ years of experience required. ISA CCST preferred.
        Education: B.S. in Electrical Engineering or Computer Science."""
    },
    {
        "id": "JD003",
        "title": "Renewable Energy Project Manager",
        "text": """Renewable Energy Project Manager
        Location: Remote (US-based, travel 30%)
        We need a Project Manager with 10+ years of experience
        leading large capital projects in the energy sector.
        Experience with solar, wind, or battery storage projects required.
        Must have strong budget forecasting and vendor management skills.
        FEED studies and greenfield design experience preferred.
        PMP certification required. PE license is a plus.
        Education: B.S. in Engineering, MBA preferred."""
    },
    {
        "id": "JD004",
        "title": "Pipeline Integrity Engineer",
        "text": """Pipeline Integrity Engineer — Midstream Operations
        Location: Tulsa, OK
        Seeking a Pipeline Integrity Engineer with 7+ years of experience
        in pipeline integrity management and inline inspection.
        Must be proficient in cathodic protection and leak detection systems.
        Experience with pig launcher and receiver operations required.
        NACE CIP Level 2 or higher preferred. API 570 certification required.
        Familiarity with DOT PHMSA regulations.
        Education: B.S. in Mechanical or Civil Engineering."""
    },
    {
        "id": "JD005",
        "title": "Offshore Subsea Engineer",
        "text": """Offshore Subsea Engineer — Deepwater Operations
        Location: Houston, TX (rotational offshore)
        Looking for a Subsea Engineer with 10+ years of deepwater experience.
        Must have expertise in subsea tieback design, flow assurance,
        and riser analysis. Experience with FPSO operations is a plus.
        Proficiency in OLGA and Pipesim required.
        PE license preferred. API 1169 certification is a plus.
        Education: M.S. in Petroleum or Mechanical Engineering."""
    },
    {
        "id": "JD006",
        "title": "Wind Farm Commissioning Engineer",
        "text": """Wind Farm Commissioning Engineer
        Location: Sweetwater, TX
        We are hiring a Wind Commissioning Engineer with 5+ years in
        wind turbine commissioning and wind resource assessment.
        Must hold GWO Basic Safety Training certification.
        Experience with onshore and offshore wind projects required.
        Skills in wind turbine blade inspection and SCADA integration.
        Education: B.S. in Mechanical or Electrical Engineering."""
    },
    {
        "id": "JD007",
        "title": "Reliability Engineer",
        "text": """Reliability Engineer — Refining Operations
        Location: Port Arthur, TX
        Seeking a Reliability Engineer with 8+ years in asset reliability.
        Must be experienced in reliability centered maintenance,
        root cause failure analysis, and vibration analysis.
        Proficiency in SAP PM or Maximo required.
        CRE certification preferred. Six Sigma certification is a plus.
        Education: B.S. in Mechanical or Industrial Engineering."""
    },
    {
        "id": "JD008",
        "title": "Energy Data Scientist",
        "text": """Energy Data Scientist — Digital Transformation
        Location: Houston, TX (hybrid)
        Looking for a Data Scientist with 4+ years applying machine learning
        and predictive modeling to energy industry challenges.
        Must be proficient in Python, SQL, and data visualization tools.
        Experience with digital twin development and IoT sensor integration preferred.
        Familiarity with cloud computing for energy platforms.
        Education: M.S. in Computer Science, Data Science, or Engineering."""
    },
    {
        "id": "JD009",
        "title": "HSE Director — Offshore",
        "text": """HSE Director — Offshore Operations
        Location: Houston, TX (travel 40%)
        Seeking an experienced HSE leader with 15+ years in health, safety,
        and environment for offshore oil and gas operations.
        Must have expertise in risk assessment, incident investigation,
        and safety culture development.
        NEBOSH Diploma and CSP certification required.
        Experience leading behavioral based safety programs across multiple platforms.
        Education: B.S. in Engineering or Environmental Science. M.S. preferred."""
    },
]


# ============================================================================
# FINANCIAL SERVICES corpus — added with the FINANCE vertical (2026-04-20).
# Spans all 8 FS departments: InvestmentBanking, EquityResearch, Risk,
# Compliance, WealthManagement, RetailBanking, Fintech, Operations.
# ============================================================================

SAMPLE_RESUMES_FINANCE = [
    # --- Investment Banking ---
    {
        "id": "F001",
        "name": "Daniel Schulman",
        "text": """Daniel Schulman — Vice President, Investment Banking
        9 years of experience in mergers and acquisitions advisory.
        Currently at Goldman Sachs as Vice President in TMT M&A.
        Led execution on $4.2B sponsor-to-sponsor leveraged buyout transaction.
        Expertise in LBO modeling, accretion-dilution analysis, and pitch book authoring.
        Skills: financial modeling, deal execution, client management, due diligence coordination.
        Tools: Excel, FactSet, PitchBook, S&P Capital IQ, Bloomberg Terminal.
        Certifications: Series 7, Series 63, Series 79, Series 24.
        Education: M.B.A. from Wharton School, B.S. in Finance from NYU Stern.
        Previously at Morgan Stanley for 4 years in Healthcare M&A."""
    },
    # --- Equity Research ---
    {
        "id": "F002",
        "name": "Priya Krishnamurthy",
        "text": """Priya Krishnamurthy — Senior Equity Research Analyst
        7 years of experience covering software and semiconductors equities.
        Working at Bernstein as Senior Analyst, US Software coverage.
        Built proprietary financial models for 18 covered names.
        Initiated coverage on 5 software platforms; published thematic deep-dives quarterly.
        Skills: discounted cash flow modeling, sector analysis, channel checks,
        management access, bottom-up forecasting, sum-of-the-parts valuation.
        Tools: Excel, FactSet, Tegus, Visible Alpha, Python for data automation.
        Certifications: CFA Charterholder, Series 86, Series 87.
        Education: B.A. in Economics from Princeton University.
        Previously at JPMorgan Equity Research for 3 years in semiconductor coverage."""
    },
    # --- Risk Management ---
    {
        "id": "F003",
        "name": "Marcus Voss",
        "text": """Marcus Voss — Senior Credit Risk Manager
        11 years of experience in counterparty credit risk and Basel III compliance.
        Currently at Citigroup as Senior Credit Risk Manager, Markets division.
        Designed and validated VaR models, expected shortfall calculations,
        and counterparty exposure aggregation across derivatives portfolio.
        Skills: stress testing, scenario analysis, model risk management,
        credit limits framework, ISDA documentation review.
        Tools: Murex, Calypso, SAS, Python, R programming for risk analytics.
        Certifications: FRM (GARP), PRM (PRMIA), CFA Level III candidate.
        Education: M.S. in Financial Engineering from Columbia University.
        Previously at Deutsche Bank in CCAR / DFAST stress testing for 4 years."""
    },
    # --- Compliance ---
    {
        "id": "F004",
        "name": "Aisha Patel-Mehta",
        "text": """Aisha Patel-Mehta — Compliance Officer, Anti-Money Laundering
        8 years of experience in AML, sanctions, and financial crimes compliance.
        Working at HSBC as Senior AML Compliance Officer.
        Led OFAC sanctions screening program covering 12 international branches.
        Built transaction monitoring rules; reduced false positive rate by 32%.
        Skills: KYC due diligence, suspicious activity reporting, regulatory reporting,
        FinCEN compliance, BSA examinations, enhanced due diligence for HRJ clients.
        Tools: Actimize, NICE, World-Check, LexisNexis Bridger.
        Certifications: CAMS (ACAMS), CRCM (ABA), CFE certified.
        Education: J.D. from Georgetown Law, B.A. in Political Science from Tufts University.
        Previously at Wells Fargo for 4 years in BSA/AML compliance."""
    },
    # --- Wealth Management ---
    {
        "id": "F005",
        "name": "Robert Donahue",
        "text": """Robert Donahue — Senior Wealth Manager, Private Banking
        14 years of experience advising high-net-worth and ultra-high-net-worth families.
        Currently at JPMorgan Private Bank as Vice President, Wealth Management.
        Manages $850M in client assets across 42 family relationships.
        Expertise in portfolio construction, tax-aware investing, estate planning coordination,
        and concentrated equity diversification strategies.
        Skills: alternatives allocation, trust and estate planning, philanthropy advisory,
        family governance, intergenerational wealth transfer.
        Certifications: CFP, CFA Charterholder, Series 7, Series 65, Series 66.
        Education: M.B.A. from Booth School of Business, B.A. in Economics from Yale.
        Previously at Merrill Lynch Wealth Management for 6 years."""
    },
    # --- Retail Banking ---
    {
        "id": "F006",
        "name": "Maria Rodriguez",
        "text": """Maria Rodriguez — Branch Manager, Retail Banking
        12 years of experience in retail banking and branch operations.
        Employed at Bank of America as Vice President, Branch Manager.
        Manages 24-person team across consumer banking and small business lending.
        Drove 18% YoY growth in deposits and 22% growth in residential mortgage originations.
        Skills: consumer lending, mortgage origination, branch P&L management,
        team coaching, customer relationship management, operational risk.
        Tools: Salesforce Financial Services Cloud, Encompass mortgage platform.
        Certifications: NMLS license (mortgage loan originator), Series 6, Series 63.
        Education: B.S. in Business Administration from University of Texas at Austin.
        Previously at Wells Fargo for 5 years as Personal Banker and Assistant Branch Manager."""
    },
    # --- Fintech / Quantitative Engineering ---
    {
        "id": "F007",
        "name": "Yuki Tanaka",
        "text": """Yuki Tanaka — Senior Quantitative Developer
        6 years of experience building low-latency trading systems and quant infrastructure.
        Currently at Citadel as Senior Quantitative Developer, Equities trading.
        Built tick-to-trade execution system serving 15 systematic strategies.
        Optimized order routing latency from 380μs to 90μs through C++ micro-tuning.
        Skills: kdb+/q for tick data, market microstructure, FIX protocol,
        Python for research workflows, distributed systems, real-time risk checks.
        Tools: kdb+/q, Python, C++, FIX engines, Murex, Bloomberg API, Refinitiv Eikon.
        Certifications: CFA Level II candidate, AWS Certified Solutions Architect.
        Education: M.S. in Computational Finance from Carnegie Mellon University.
        Previously at Two Sigma for 3 years in equity research infrastructure."""
    },
    # --- Operations / Middle Office ---
    {
        "id": "F008",
        "name": "Jennifer Okonkwo",
        "text": """Jennifer Okonkwo — Senior Operations Analyst, Middle Office
        9 years of experience in trade settlement and middle office operations.
        Working at State Street as Assistant Vice President, Derivatives Operations.
        Manages settlement and reconciliation for $42B notional OTC derivatives book.
        Led migration from T+2 to T+1 settlement for North American equities portfolio.
        Skills: trade lifecycle management, SWIFT messaging, collateral management,
        reconciliation breaks investigation, regulatory reporting, SOX controls.
        Tools: GlobeOp, Markit Wire, DTCC Omgeo, Bloomberg AIM, Excel power user.
        Certifications: Series 99, IOC (Investment Operations Certificate), CISI Diploma.
        Education: B.S. in Finance from Howard University.
        Previously at BNY Mellon Asset Servicing for 5 years."""
    },
]


SAMPLE_JOB_DESCRIPTIONS_FINANCE = [
    {
        "id": "FJD001",
        "title": "Vice President, M&A Investment Banking",
        "text": """Vice President, M&A Investment Banking — TMT Group
        Location: New York, NY
        Bulge-bracket investment bank seeks an experienced VP for the
        Technology, Media & Telecommunications M&A advisory team.
        Must have 7+ years of execution experience across sell-side
        advisory, buy-side advisory, and leveraged buyout transactions.
        Strong financial modeling, accretion-dilution analysis, and
        pitch book authoring required. Experience leading associates
        and analysts on live deal teams.
        Series 7 and Series 79 required. CFA preferred.
        Education: M.B.A. from a top program; B.S. in Finance, Economics, or Engineering."""
    },
    {
        "id": "FJD002",
        "title": "Senior Equity Research Analyst — Software",
        "text": """Senior Equity Research Analyst — Software Coverage
        Location: New York, NY
        Top-tier sell-side equity research desk seeks a Senior Analyst
        to cover the US enterprise software universe (~15-20 names).
        Must have 5+ years of equity research experience with proven
        track record in financial modeling, sector analysis, and
        publishing buy-side-relevant thematic research.
        Strong client management and management-team access skills.
        CFA Charterholder required. Series 86/87 required to publish.
        Education: B.A./B.S. in Finance, Economics, Computer Science, or related."""
    },
    {
        "id": "FJD003",
        "title": "Senior Credit Risk Manager",
        "text": """Senior Credit Risk Manager — Markets Division
        Location: New York, NY
        Global investment bank seeks Senior Credit Risk Manager for
        OTC derivatives and securities financing portfolio.
        Must have 8+ years of counterparty credit risk experience,
        deep expertise in VaR modeling, expected shortfall, stress
        testing, and Basel III IMM/SA-CCR frameworks.
        ISDA documentation review and CSA negotiation experience required.
        FRM (GARP) certification required. CFA preferred.
        Education: M.S. in Financial Engineering, Quantitative Finance, or Mathematics."""
    },
    {
        "id": "FJD004",
        "title": "Senior AML Compliance Officer",
        "text": """Senior AML Compliance Officer — Global Banking
        Location: Charlotte, NC
        International bank seeks Senior AML Compliance Officer to lead
        the BSA/AML program across the wholesale banking division.
        Must have 6+ years of AML, sanctions, and financial crimes
        compliance experience. Deep expertise in OFAC sanctions screening,
        KYC due diligence, transaction monitoring tuning, and SAR filing.
        Experience with regulatory examinations (OCC, FRB, FinCEN) required.
        CAMS certification required. CRCM strongly preferred.
        Education: J.D., M.B.A., or B.A. in Finance, Law, or Political Science."""
    },
    {
        "id": "FJD005",
        "title": "Senior Wealth Manager — Private Banking",
        "text": """Senior Wealth Manager — Private Banking
        Location: San Francisco, CA
        Top private bank seeks Senior Wealth Manager to grow and
        manage book of HNW/UHNW family relationships ($25M+ per family).
        Must have 10+ years of wealth management or private banking
        experience with verifiable book-of-business of $300M+.
        Strong portfolio construction, tax-aware investing, alternatives
        allocation, and trust and estate planning expertise.
        CFP and Series 7/65 required. CFA Charterholder strongly preferred.
        Education: B.A./B.S. required; M.B.A. preferred."""
    },
]


# ============================================================================
# Industry-aware accessors. Use these in any UI / code path that should
# adapt to the active industry profile. Backwards-compatible: when the
# active industry is Energy (the default), these return the original
# SAMPLE_RESUMES / SAMPLE_JOB_DESCRIPTIONS lists.
# ============================================================================

_RESUMES_BY_INDUSTRY = {
    "energy":  SAMPLE_RESUMES,
    "finance": SAMPLE_RESUMES_FINANCE,
}

_JDS_BY_INDUSTRY = {
    "energy":  SAMPLE_JOB_DESCRIPTIONS,
    "finance": SAMPLE_JOB_DESCRIPTIONS_FINANCE,
}


def get_active_resumes():
    """Return the resume corpus matching the currently-active industry.

    Falls back to Energy if no industry is registered or if the active
    industry has no FS-style sample corpus yet (e.g., a brand-new vertical).
    """
    try:
        from industries import get_industry
        key = get_industry().name
    except Exception:
        key = "energy"
    return _RESUMES_BY_INDUSTRY.get(key, SAMPLE_RESUMES)


def get_active_jds():
    """Return the JD corpus matching the currently-active industry.

    See `get_active_resumes()` for the fallback contract.
    """
    try:
        from industries import get_industry
        key = get_industry().name
    except Exception:
        key = "energy"
    return _JDS_BY_INDUSTRY.get(key, SAMPLE_JOB_DESCRIPTIONS)
