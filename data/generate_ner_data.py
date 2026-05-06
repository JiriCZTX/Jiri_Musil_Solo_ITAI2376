"""
Generate synthetic energy-sector resume NER training data.
Produces BIO-tagged token sequences for fine-tuning DistilBERT on
entity types: SKILL, CERT, DEGREE, EMPLOYER, YEARS_EXP.

Uses the comprehensive energy_taxonomy module (221 skills, 51 certs,
30 degrees, 62 employers) sourced from O*NET, API, NACE, ISA, OSHA.
"""
import json
import random
from pathlib import Path

from data.energy_taxonomy import (
    SKILLS, CERTIFICATIONS, DEGREES, EMPLOYERS,
    YEARS_EXPRESSIONS, SENTENCE_TEMPLATES,
)


def _tag_entity(tokens, entity_type):
    """Apply BIO tags to a list of tokens for a given entity type."""
    if not tokens:
        return []
    tagged = [(tokens[0], f"B-{entity_type}")]
    for t in tokens[1:]:
        tagged.append((t, f"I-{entity_type}"))
    return tagged


def _tokenize_and_tag(text, entities):
    """
    Punctuation-tolerant BIO tagger. Tokens are compared against the
    entity span after stripping surrounding punctuation — so 'license,'
    still matches 'license' and the output preserves the original token.
    Longer spans are tagged first to avoid sub-span collisions.
    """
    import re as _re
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
                break  # Tag first occurrence only

    return tokens, labels


def generate_sample():
    """Generate one synthetic resume snippet with BIO-tagged entities."""
    template = random.choice(SENTENCE_TEMPLATES)
    n_years = random.randint(3, 25)
    years_expr = random.choice(YEARS_EXPRESSIONS).format(n=n_years)

    skill1 = random.choice(SKILLS)
    skill2 = random.choice([s for s in SKILLS if s != skill1])
    skill3 = random.choice([s for s in SKILLS if s not in (skill1, skill2)])
    cert = random.choice(CERTIFICATIONS)
    degree = random.choice(DEGREES)
    employer = random.choice(EMPLOYERS)

    text = template.format(
        years=years_expr, employer=employer,
        skill=skill1, skill2=skill2, skill3=skill3,
        cert=cert, degree=degree,
    )

    entities = {
        skill1: "SKILL", skill2: "SKILL", skill3: "SKILL",
        cert: "CERT", degree: "DEGREE",
        employer: "EMPLOYER", years_expr: "YEARS_EXP",
    }

    tokens, labels = _tokenize_and_tag(text, entities)
    return {"tokens": tokens, "ner_tags": labels, "text": text}


def _hand_labeled_samples():
    """
    Hand-labeled examples from the actual sample resumes.
    These teach the model the exact format it will encounter in the demo.
    Each sample is repeated to increase weight during training.
    """
    samples = [
        {
            "text": "12 years of experience in oil and gas operations. Currently at Chevron as Lead Process Engineer. Expertise in P&ID review, HAZOP facilitation, and process safety management. Led turnaround execution for a 150,000 bpd refinery unit. Proficient in HYSYS, AutoCAD, and Python for process simulation. Certifications: PE license, API 570 certified, Six Sigma Black Belt. Education: M.S. in Chemical Engineering from Texas A&M University. Previously at Shell for 5 years in upstream operations.",
            "entities": {
                "12 years of experience": "YEARS_EXP",
                "Chevron": "EMPLOYER",
                "P&ID review": "SKILL", "HAZOP facilitation": "SKILL",
                "process safety management": "SKILL", "turnaround execution": "SKILL",
                "AutoCAD": "SKILL", "Python": "SKILL",
                "PE license": "CERT", "API 570 certified": "CERT",
                "Six Sigma Black Belt": "CERT",
                "M.S. in Chemical Engineering": "DEGREE",
                "Shell": "EMPLOYER",
            },
        },
        {
            "text": "8 years of experience in power systems and renewable energy. Working at NextEra Energy as Senior Electrical Engineer. Skilled in relay protection, arc flash analysis, and load flow studies. Managed electrical commissioning for 3 solar farms. Tools: ETAP, SKM PowerTools, MATLAB, Python. Certifications: PE license, LEED AP. Education: B.S. in Electrical Engineering from University of Houston. Prior experience at Siemens Energy in gas turbine commissioning.",
            "entities": {
                "8 years of experience": "YEARS_EXP",
                "NextEra Energy": "EMPLOYER",
                "relay protection": "SKILL", "arc flash analysis": "SKILL",
                "load flow studies": "SKILL", "electrical commissioning": "SKILL",
                "MATLAB": "SKILL", "Python": "SKILL",
                "PE license": "CERT", "LEED AP": "CERT",
                "B.S. in Electrical Engineering": "DEGREE",
                "Siemens Energy": "EMPLOYER",
            },
        },
        {
            "text": "15 years of hands-on experience in industrial maintenance. Employed at Baker Hughes as Maintenance Planning Supervisor. Core skills: vibration analysis, NDT testing, welding inspection. Implemented predictive maintenance program reducing downtime by 30%. Familiar with SAP PM, Maximo, and PLC programming. Certifications: AWS CWI, NACE CIP Level 2. Education: Associate in Process Technology from San Jacinto College. Previously at Halliburton for 8 years.",
            "entities": {
                "15 years of hands-on experience": "YEARS_EXP",
                "Baker Hughes": "EMPLOYER",
                "vibration analysis": "SKILL", "NDT testing": "SKILL",
                "welding inspection": "SKILL", "SAP PM": "SKILL",
                "Maximo": "SKILL", "PLC programming": "SKILL",
                "AWS CWI": "CERT", "NACE CIP Level 2": "CERT",
                "Associate in Process Technology": "DEGREE",
                "Halliburton": "EMPLOYER",
            },
        },
        {
            "text": "10 years of experience in health, safety, and environment. Currently HSE Director at TotalEnergies Americas. Expertise in risk assessment, MOC management, and SIL verification. Led safety culture transformation across 4 offshore platforms. Reduced recordable incident rate by 45% over 3 years. Certifications: NEBOSH certified, CSP certified, OSHA 30. Education: B.S. in Environmental Science from Rice University. M.S. in Energy Systems from University of Texas at Austin.",
            "entities": {
                "10 years of experience": "YEARS_EXP",
                "TotalEnergies": "EMPLOYER",
                "risk assessment": "SKILL", "MOC management": "SKILL",
                "SIL verification": "SKILL",
                "NEBOSH certified": "CERT", "CSP certified": "CERT",
                "OSHA 30": "CERT",
                "B.S. in Environmental Science": "DEGREE",
                "M.S. in Energy Systems": "DEGREE",
            },
        },
        {
            "text": "6 years of experience in control systems and automation. Working at Emerson Electric as Senior SCADA Specialist. Skills: SCADA integration, DCS programming, PLC programming, instrumentation calibration, and cybersecurity for OT networks. Deployed SCADA upgrade across 12 natural gas compression stations. Certifications: ISA CCST, CompTIA Security+. Education: B.S. in Computer Science from University of Houston. Python, SQL, and C++ proficiency.",
            "entities": {
                "6 years of experience": "YEARS_EXP",
                "Emerson Electric": "EMPLOYER",
                "SCADA integration": "SKILL", "DCS programming": "SKILL",
                "PLC programming": "SKILL", "instrumentation calibration": "SKILL",
                "Python": "SKILL",
                "ISA CCST": "CERT", "CompTIA Security+": "CERT",
                "B.S. in Computer Science": "DEGREE",
            },
        },
        {
            "text": "20 years of experience in capital projects for energy. Currently at Bechtel as Senior Project Director. Managed LNG export terminal construction project. Expert in shutdown planning, brownfield engineering, and FEED studies. Skilled in budget forecasting, vendor management, and stakeholder alignment. Certifications: PMP certified, PE license, Lean Manufacturing certified. Education: B.S. in Civil Engineering, MBA from Rice University.",
            "entities": {
                "20 years of experience": "YEARS_EXP",
                "Bechtel": "EMPLOYER",
                "shutdown planning": "SKILL", "brownfield engineering": "SKILL",
                "FEED studies": "SKILL", "budget forecasting": "SKILL",
                "vendor management": "SKILL",
                "PMP certified": "CERT", "PE license": "CERT",
                "Lean Manufacturing certified": "CERT",
                "B.S. in Civil Engineering": "DEGREE", "MBA": "DEGREE",
            },
        },
        # --- R007: Subsea Engineer ---
        {
            "text": "14 years of experience in offshore operations and subsea engineering. Currently at TechnipFMC as Principal Subsea Systems Engineer. Expertise in subsea tieback design, flow assurance, and riser analysis. Designed subsea production system for deepwater field at 7,500 ft depth. Proficient in ANSYS for subsea analysis. Certifications: PE license, API 1169 certified. Education: M.S. in Petroleum Engineering from University of Houston. Previously at Saipem for 6 years in marine operations.",
            "entities": {
                "14 years of experience": "YEARS_EXP",
                "TechnipFMC": "EMPLOYER",
                "subsea tieback design": "SKILL", "flow assurance": "SKILL",
                "riser analysis": "SKILL", "ANSYS": "SKILL",
                "PE license": "CERT", "API 1169 certified": "CERT",
                "M.S. in Petroleum Engineering": "DEGREE",
                "Saipem": "EMPLOYER",
            },
        },
        # --- R008: Offshore Operations Manager ---
        {
            "text": "18 years of experience in offshore oil and gas production. Employed at Equinor as Operations Director for Gulf of Mexico assets. Led FPSO operations, dynamic positioning, and topside process design reviews. Managed production optimization across 3 deepwater platforms. Skills include well intervention, mooring system design, and marine operations. Certifications: NEBOSH Diploma, OSHA 30, PMP certified. Education: B.S. in Mechanical Engineering from Texas A&M University. Previous role at BP in jack-up rig operations for 7 years.",
            "entities": {
                "18 years of experience": "YEARS_EXP",
                "Equinor": "EMPLOYER",
                "FPSO operations": "SKILL", "dynamic positioning": "SKILL",
                "topside process design": "SKILL", "production optimization": "SKILL",
                "well intervention": "SKILL", "mooring system design": "SKILL",
                "marine operations": "SKILL",
                "NEBOSH Diploma": "CERT", "OSHA 30": "CERT", "PMP certified": "CERT",
                "B.S. in Mechanical Engineering": "DEGREE",
                "BP": "EMPLOYER",
            },
        },
        # --- R009: Wind Energy Engineer ---
        {
            "text": "9 years of experience in wind energy development and commissioning. Working at Vestas as Senior Wind Turbine Engineer. Skilled in wind resource assessment, wind turbine commissioning, and wind turbine blade inspection. Led commissioning of 300 MW onshore wind farm in West Texas. Tools: Python. Certifications: GWO Basic Safety Training, PE license. Education: M.S. in Mechanical Engineering. Prior experience at Orsted in offshore wind development.",
            "entities": {
                "9 years of experience": "YEARS_EXP",
                "Vestas": "EMPLOYER",
                "wind resource assessment": "SKILL", "wind turbine commissioning": "SKILL",
                "wind turbine blade inspection": "SKILL", "Python": "SKILL",
                "GWO Basic Safety Training": "CERT", "PE license": "CERT",
                "M.S. in Mechanical Engineering": "DEGREE",
                "Orsted": "EMPLOYER",
            },
        },
        # --- R010: Solar Energy Project Engineer ---
        {
            "text": "5 years of experience in solar energy and battery storage systems. Currently at First Solar as Project Engineer. Expertise in solar farm design, solar inverter commissioning, and BESS commissioning for utility-scale projects. Managed grid interconnection studies for 200 MW solar-plus-storage facility. Proficient in ETAP and AutoCAD. Certifications: NABCEP PV Installation Professional, FE certification. Education: B.S. in Electrical Engineering from Arizona State University.",
            "entities": {
                "5 years of experience": "YEARS_EXP",
                "First Solar": "EMPLOYER",
                "solar farm design": "SKILL", "solar inverter commissioning": "SKILL",
                "BESS commissioning": "SKILL", "grid interconnection studies": "SKILL",
                "ETAP": "SKILL", "AutoCAD": "SKILL",
                "NABCEP PV Installation Professional": "CERT", "FE certification": "CERT",
                "B.S. in Electrical Engineering": "DEGREE",
            },
        },
        # --- R011: Pipeline Integrity Engineer ---
        {
            "text": "11 years of experience in midstream operations and pipeline integrity. Working at Enterprise Products Partners as Senior Integrity Engineer. Expert in pipeline integrity management, inline inspection, and cathodic protection systems. Managed integrity assessment program covering 2,500 miles of NGL pipeline. Skills: pig launcher and receiver operations, leak detection systems, SCADA for pipelines. Certifications: NACE CIP Level 3, API 570 certified, ASNT NDT Level II. Education: B.S. in Mechanical Engineering from University of Texas at Austin. Previously at Kinder Morgan for 5 years.",
            "entities": {
                "11 years of experience": "YEARS_EXP",
                "Enterprise Products Partners": "EMPLOYER",
                "pipeline integrity management": "SKILL", "inline inspection": "SKILL",
                "cathodic protection": "SKILL", "leak detection systems": "SKILL",
                "NACE CIP Level 3": "CERT", "API 570 certified": "CERT",
                "ASNT NDT Level II": "CERT",
                "B.S. in Mechanical Engineering": "DEGREE",
                "Kinder Morgan": "EMPLOYER",
            },
        },
        # --- R012: Gas Processing Engineer ---
        {
            "text": "7 years of experience in natural gas processing and LNG operations. Employed at Williams Companies as Process Engineer. Skilled in gas dehydration, amine treating, NGL fractionation, and sulfur recovery unit operations. Optimized LNG processing throughput by 15%. Tools: HYSYS, Aspen Plus, OSIsoft PI. Certifications: PE license, Six Sigma Green Belt. Education: B.S. in Chemical Engineering from Louisiana State University. Prior role at Enbridge in natural gas compression.",
            "entities": {
                "7 years of experience": "YEARS_EXP",
                "Williams Companies": "EMPLOYER",
                "gas dehydration": "SKILL", "amine treating": "SKILL",
                "NGL fractionation": "SKILL", "sulfur recovery": "SKILL",
                "LNG processing": "SKILL", "HYSYS": "SKILL", "Aspen Plus": "SKILL",
                "PE license": "CERT", "Six Sigma Green Belt": "CERT",
                "B.S. in Chemical Engineering": "DEGREE",
                "Enbridge": "EMPLOYER",
            },
        },
        # --- R013: Nuclear Systems Engineer ---
        {
            "text": "16 years of experience in nuclear power plant operations and safety. Currently at Exelon as Lead Nuclear Engineer. Expertise in reactor design, radiation protection, and functional safety assessment. Led SIS design upgrade for emergency core cooling system. Proficient in MATLAB and Simulink for nuclear modeling. Certifications: PE license, TUV Functional Safety Engineer. Education: M.S. in Nuclear Engineering from MIT. Previous experience at GE Vernova in boiling water reactor design for 8 years.",
            "entities": {
                "16 years of experience": "YEARS_EXP",
                "Exelon": "EMPLOYER",
                "reactor design": "SKILL", "radiation protection": "SKILL",
                "functional safety assessment": "SKILL", "SIS design": "SKILL",
                "MATLAB": "SKILL", "Simulink": "SKILL",
                "PE license": "CERT", "TUV Functional Safety Engineer": "CERT",
                "M.S. in Nuclear Engineering": "DEGREE",
                "GE Vernova": "EMPLOYER",
            },
        },
        # --- R014: Senior Drilling Engineer ---
        {
            "text": "13 years of experience in drilling operations and well design. Working at Saudi Aramco as Drilling Superintendent. Expert in drilling optimization, well completion design, and deepwater drilling. Managed 40+ well drilling campaigns in the Arabian Gulf. Skilled in reservoir simulation and artificial lift design. Certifications: API 510 certified, NEBOSH IGC certified, OSHA 30. Education: B.S. in Petroleum Engineering. Previously at Schlumberger for 5 years in directional drilling services.",
            "entities": {
                "13 years of experience": "YEARS_EXP",
                "Saudi Aramco": "EMPLOYER",
                "drilling optimization": "SKILL", "well completion design": "SKILL",
                "deepwater drilling": "SKILL", "reservoir simulation": "SKILL",
                "artificial lift design": "SKILL",
                "API 510 certified": "CERT", "NEBOSH IGC certified": "CERT", "OSHA 30": "CERT",
                "B.S. in Petroleum Engineering": "DEGREE",
                "Schlumberger": "EMPLOYER",
            },
        },
        # --- R015: Junior Instrumentation Engineer ---
        {
            "text": "2 years of experience in control systems and instrumentation. Entry-level engineer at Honeywell UOP. Skills: control loop tuning, HMI development, fieldbus configuration. Assisted with DCS programming for a refinery modernization project. Proficient in LabVIEW, Python, and SQL. Certifications: FE certification. Education: B.S. in Electrical Engineering from University of Houston. Completed internship at Yokogawa in process control.",
            "entities": {
                "2 years of experience": "YEARS_EXP",
                "Honeywell UOP": "EMPLOYER",
                "control loop tuning": "SKILL", "HMI development": "SKILL",
                "fieldbus configuration": "SKILL", "DCS programming": "SKILL",
                "LabVIEW": "SKILL", "Python": "SKILL", "SQL": "SKILL",
                "FE certification": "CERT",
                "B.S. in Electrical Engineering": "DEGREE",
                "Yokogawa": "EMPLOYER",
            },
        },
        # --- R016: Field Technician ---
        {
            "text": "3 years of hands-on experience in industrial instrumentation. Currently at Endress+Hauser as Field Service Technician. Core skills: pressure transmitter calibration, RTD calibration, flow measurement, and level measurement. Performs instrumentation calibration at refineries and chemical plants. Certifications: NCCER certified, OSHA 10. Education: Associate in Instrumentation Technology from Lee College.",
            "entities": {
                "3 years of hands-on experience": "YEARS_EXP",
                "Endress+Hauser": "EMPLOYER",
                "pressure transmitter calibration": "SKILL", "RTD calibration": "SKILL",
                "flow measurement": "SKILL", "level measurement": "SKILL",
                "instrumentation calibration": "SKILL",
                "NCCER certified": "CERT", "OSHA 10": "CERT",
                "Associate in Instrumentation Technology": "DEGREE",
            },
        },
        # --- R017: VP of Engineering ---
        {
            "text": "25 years of experience in energy industry leadership. Currently Vice President of Engineering at ConocoPhillips. Led global capital project portfolio worth $4.8B annually. Expert in stakeholder alignment, earned value management, and risk register management. Drove digital twin development initiative across 12 production assets. Certifications: PMP certified, PE license, ISO 9001 Lead Auditor. Education: B.S. in Chemical Engineering from Stanford University. Executive MBA from Wharton. Previous roles at ExxonMobil and Shell.",
            "entities": {
                "25 years of experience": "YEARS_EXP",
                "ConocoPhillips": "EMPLOYER",
                "stakeholder alignment": "SKILL", "earned value management": "SKILL",
                "risk register management": "SKILL", "digital twin development": "SKILL",
                "PMP certified": "CERT", "PE license": "CERT",
                "ISO 9001 Lead Auditor": "CERT",
                "B.S. in Chemical Engineering": "DEGREE", "Executive MBA": "DEGREE",
                "ExxonMobil": "EMPLOYER", "Shell": "EMPLOYER",
            },
        },
        # --- R018: CTO ---
        {
            "text": "22 years of experience in energy technology and digital transformation. CTO at ABB Energy Industries overseeing global automation strategy. Led deployment of machine learning and predictive modeling for predictive maintenance across 200+ industrial facilities. Expert in IoT sensor integration, cloud computing for energy, and digital twin development. Skilled in advanced process control and model predictive control. Certifications: PE license, ISA CAP certified, Certified Energy Manager. Education: Ph.D. in Chemical Engineering from Caltech. Prior experience at Schneider Electric and Rockwell Automation.",
            "entities": {
                "22 years of experience": "YEARS_EXP",
                "ABB Energy Industries": "EMPLOYER",
                "machine learning": "SKILL", "predictive modeling": "SKILL",
                "predictive maintenance": "SKILL", "IoT sensor integration": "SKILL",
                "cloud computing for energy": "SKILL", "digital twin development": "SKILL",
                "advanced process control": "SKILL", "model predictive control": "SKILL",
                "PE license": "CERT", "ISA CAP certified": "CERT",
                "Certified Energy Manager": "CERT",
                "Ph.D. in Chemical Engineering": "DEGREE",
                "Schneider Electric": "EMPLOYER", "Rockwell Automation": "EMPLOYER",
            },
        },
        # --- R019: Corrosion Engineer ---
        {
            "text": "8 years of experience in corrosion engineering and materials selection. Working at ADNOC as Senior Corrosion and Materials Engineer. Expert in corrosion engineering, material selection, and cathodic protection. Managed chemical injection systems for 15 offshore wellhead platforms. Skilled in pressure vessel inspection and piping stress analysis. Certifications: NACE CIP Level 2, API 571 certified, API 577 certified. Education: M.S. in Materials Science from Imperial College London. Previously at Petrobras in subsea corrosion management for 3 years.",
            "entities": {
                "8 years of experience": "YEARS_EXP",
                "ADNOC": "EMPLOYER",
                "corrosion engineering": "SKILL", "material selection": "SKILL",
                "cathodic protection": "SKILL", "chemical injection systems": "SKILL",
                "pressure vessel inspection": "SKILL", "piping stress analysis": "SKILL",
                "NACE CIP Level 2": "CERT", "API 571 certified": "CERT",
                "API 577 certified": "CERT",
                "M.S. in Materials Science": "DEGREE",
                "Petrobras": "EMPLOYER",
            },
        },
        # --- R020: Reliability Engineer ---
        {
            "text": "10 years of experience in asset reliability and maintenance optimization. Employed at Fluor Corporation as Lead Reliability Engineer. Skills: reliability centered maintenance, root cause failure analysis, bearing failure analysis, and rotating equipment alignment. Implemented predictive maintenance program saving $3.2M annually. Tools: SAP PM, Maximo, OSIsoft PI, Power BI. Certifications: CRE certified, Six Sigma Black Belt, ASNT NDT Level II. Education: B.S. in Mechanical Engineering from Purdue University. Previous role at NOV in rotating equipment engineering.",
            "entities": {
                "10 years of experience": "YEARS_EXP",
                "Fluor Corporation": "EMPLOYER",
                "reliability centered maintenance": "SKILL",
                "root cause failure analysis": "SKILL",
                "bearing failure analysis": "SKILL",
                "rotating equipment alignment": "SKILL",
                "predictive maintenance": "SKILL",
                "SAP PM": "SKILL", "Maximo": "SKILL",
                "CRE certified": "CERT", "Six Sigma Black Belt": "CERT",
                "ASNT NDT Level II": "CERT",
                "B.S. in Mechanical Engineering": "DEGREE",
                "NOV": "EMPLOYER",
            },
        },
        # --- R021: Hydrogen Systems Engineer ---
        {
            "text": "4 years of experience in clean energy and hydrogen technology. Currently at Enphase Energy as Hydrogen Projects Lead. Expertise in green hydrogen electrolysis, hydrogen production, and energy storage design for industrial applications. Led feasibility study for 50 MW green hydrogen facility. Skills: lifecycle assessment, sustainability reporting, carbon capture. Certifications: LEED AP, Certified Energy Manager. Education: M.S. in Energy Systems from ETH Zurich.",
            "entities": {
                "4 years of experience": "YEARS_EXP",
                "Enphase Energy": "EMPLOYER",
                "green hydrogen electrolysis": "SKILL", "hydrogen production": "SKILL",
                "energy storage design": "SKILL", "lifecycle assessment": "SKILL",
                "sustainability reporting": "SKILL", "carbon capture": "SKILL",
                "LEED AP": "CERT", "Certified Energy Manager": "CERT",
                "M.S. in Energy Systems": "DEGREE",
            },
        },
        # --- R022: Energy Data Scientist ---
        {
            "text": "6 years of experience in data analytics for the energy sector. Working at Chevron as Senior Data Scientist. Expert in machine learning, predictive modeling, and data visualization. Built real-time production forecasting model using Python and R programming. Skills: statistical process control, regression analysis, digital twin development. Proficient in Tableau, Power BI, SQL, and cloud computing for energy. Certifications: Six Sigma Green Belt, CompTIA Security+. Education: M.S. in Computer Science from Georgia Tech. Previously at Halliburton in IoT sensor integration analytics.",
            "entities": {
                "6 years of experience": "YEARS_EXP",
                "Chevron": "EMPLOYER",
                "machine learning": "SKILL", "predictive modeling": "SKILL",
                "data visualization": "SKILL", "Python": "SKILL",
                "statistical process control": "SKILL", "regression analysis": "SKILL",
                "digital twin development": "SKILL", "SQL": "SKILL",
                "Six Sigma Green Belt": "CERT", "CompTIA Security+": "CERT",
                "M.S. in Computer Science": "DEGREE",
                "Halliburton": "EMPLOYER",
            },
        },
        # --- R023: Commissioning Manager ---
        {
            "text": "12 years of experience in commissioning and startup for energy facilities. Currently at Wood PLC as Commissioning Lead. Skilled in commissioning and startup, electrical commissioning, and solar inverter commissioning. Managed pre-commissioning through mechanical completion for 800 MW combined-cycle gas turbine power plant. Tools: Primavera P6, Microsoft Project, SmartPlant 3D. Certifications: PMP certified, NICET Level III, PE license. Education: B.S. in Electrical Engineering from Virginia Tech. Prior role at KBR in LNG facility commissioning.",
            "entities": {
                "12 years of experience": "YEARS_EXP",
                "Wood PLC": "EMPLOYER",
                "commissioning and startup": "SKILL", "electrical commissioning": "SKILL",
                "solar inverter commissioning": "SKILL",
                "Primavera P6": "SKILL", "SmartPlant 3D": "SKILL",
                "PMP certified": "CERT", "NICET Level III": "CERT", "PE license": "CERT",
                "B.S. in Electrical Engineering": "DEGREE",
                "KBR": "EMPLOYER",
            },
        },
        # --- R024: Environmental Engineer ---
        {
            "text": "9 years of experience in environmental compliance for energy operations. Working at Sempra Energy as Lead Environmental Specialist. Expertise in environmental impact assessment, air quality monitoring, wastewater treatment oversight, and hazardous waste management. Led environmental permitting for 3 natural gas power plants. Skills: carbon sequestration monitoring, sustainability reporting. Certifications: CHMM certified, ISO 14001 Lead Auditor, OSHA 30. Education: M.S. in Environmental Science from UC Berkeley. Previously at AES Corporation for 4 years.",
            "entities": {
                "9 years of experience": "YEARS_EXP",
                "Sempra Energy": "EMPLOYER",
                "environmental impact assessment": "SKILL", "air quality monitoring": "SKILL",
                "wastewater treatment oversight": "SKILL", "hazardous waste management": "SKILL",
                "carbon sequestration monitoring": "SKILL", "sustainability reporting": "SKILL",
                "CHMM certified": "CERT", "ISO 14001 Lead Auditor": "CERT", "OSHA 30": "CERT",
                "M.S. in Environmental Science": "DEGREE",
                "AES Corporation": "EMPLOYER",
            },
        },
        # --- R025: Grid Operations Engineer ---
        {
            "text": "7 years of experience in grid operations and power distribution. Employed at Duke Energy as Senior Grid Analyst. Skilled in distribution system planning, power factor correction, harmonic analysis, and substation design. Managed grid interconnection studies for 500 MW of distributed solar. Tools: ETAP, Python, MATLAB, Power BI. Certifications: PE license, CEM certified. Education: M.S. in Electrical Engineering from North Carolina State University. Prior role at Southern Company in transmission line design.",
            "entities": {
                "7 years of experience": "YEARS_EXP",
                "Duke Energy": "EMPLOYER",
                "distribution system planning": "SKILL", "power factor correction": "SKILL",
                "harmonic analysis": "SKILL", "substation design": "SKILL",
                "grid interconnection studies": "SKILL",
                "ETAP": "SKILL", "Python": "SKILL", "MATLAB": "SKILL",
                "PE license": "CERT", "CEM certified": "CERT",
                "M.S. in Electrical Engineering": "DEGREE",
                "Southern Company": "EMPLOYER",
            },
        },
        # --- R026: Master Electrician ---
        {
            "text": "17 years of hands-on experience in industrial electrical systems. Currently at Zachry Group as Electrical Foreman. Core skills: switchgear maintenance, transformer testing, cable sizing and routing, and grounding system design. Led electrical installation for petrochemical plant expansion. Experienced in crane inspection, rigging and lifting operations. Certifications: Master Electrician license, OSHA 30, NCCER certified. Education: Journeyman Electrician license from IBEW. Previously at Kiewit for 9 years in heavy industrial construction.",
            "entities": {
                "17 years of hands-on experience": "YEARS_EXP",
                "Zachry Group": "EMPLOYER",
                "switchgear maintenance": "SKILL", "transformer testing": "SKILL",
                "cable sizing and routing": "SKILL", "grounding system design": "SKILL",
                "crane inspection": "SKILL", "rigging and lifting": "SKILL",
                "Master Electrician license": "CERT", "OSHA 30": "CERT",
                "NCCER certified": "CERT",
                "Journeyman Electrician license": "CERT",
                "Kiewit": "EMPLOYER",
            },
        },
        # --- R027: Reservoir Engineer ---
        {
            "text": "11 years of experience in reservoir simulation and production optimization. Working at Petronas as Senior Reservoir Engineer. Expert in reservoir simulation, production optimization, and well completion design for complex carbonate reservoirs. Increased field recovery factor by 8% through waterflood optimization. Skills: artificial lift design, flow assurance, hydrate management. Certifications: PE license, Six Sigma Green Belt. Education: Ph.D. in Petroleum Engineering from Colorado School of Mines. Previous role at Repsol in enhanced oil recovery.",
            "entities": {
                "11 years of experience": "YEARS_EXP",
                "Petronas": "EMPLOYER",
                "reservoir simulation": "SKILL", "production optimization": "SKILL",
                "well completion design": "SKILL", "artificial lift design": "SKILL",
                "flow assurance": "SKILL", "hydrate management": "SKILL",
                "PE license": "CERT", "Six Sigma Green Belt": "CERT",
                "Ph.D. in Petroleum Engineering": "DEGREE",
                "Repsol": "EMPLOYER",
            },
        },
        # --- R028: Fire Protection Engineer ---
        {
            "text": "8 years of experience in fire protection engineering for energy facilities. Currently at Jacobs Engineering as Senior Fire Safety Engineer. Expertise in fire protection engineering, emergency response planning, and process hazard analysis. Designed fire suppression systems for 4 offshore production platforms. Skilled in permit to work systems and confined space entry procedures. Certifications: CFSE certified, PE license, NEBOSH IGC certified. Education: B.S. in Mechanical Engineering from University of Maryland. Prior experience at McDermott International in safety engineering.",
            "entities": {
                "8 years of experience": "YEARS_EXP",
                "Jacobs Engineering": "EMPLOYER",
                "fire protection engineering": "SKILL", "emergency response planning": "SKILL",
                "process hazard analysis": "SKILL", "permit to work systems": "SKILL",
                "confined space entry procedures": "SKILL",
                "CFSE certified": "CERT", "PE license": "CERT",
                "NEBOSH IGC certified": "CERT",
                "B.S. in Mechanical Engineering": "DEGREE",
                "McDermott International": "EMPLOYER",
            },
        },
        # --- R029: Offshore Wind Operations Manager ---
        {
            "text": "12 years of experience in offshore wind development and O&M. Currently at Orsted as Operations Manager for North Sea assets. Led commissioning of 1.2 GW offshore wind farm with monopile foundation design. Skilled in WTIV operations, jack-up vessel operations, and inter-array cable installation. Manages crew transfer operations across 3 wind farms with 90% turbine availability. Tools: WindPRO, Python. Certifications: GWO Basic Safety Training, GWO sea survival, PMP certified. Education: M.S. in Mechanical Engineering from Technical University of Denmark. Previously at Vestas for 5 years in turbine commissioning.",
            "entities": {
                "12 years of experience": "YEARS_EXP",
                "Orsted": "EMPLOYER",
                "monopile foundation design": "SKILL", "WTIV operations": "SKILL",
                "jack-up vessel operations": "SKILL",
                "inter-array cable installation": "SKILL",
                "crew transfer operations": "SKILL", "WindPRO": "SKILL",
                "Python": "SKILL",
                "GWO Basic Safety Training": "CERT", "GWO sea survival": "CERT",
                "PMP certified": "CERT",
                "M.S. in Mechanical Engineering": "DEGREE",
                "Vestas": "EMPLOYER",
            },
        },
        # --- R030: Utility-Scale Solar PV Director ---
        {
            "text": "16 years of experience in utility-scale solar development. Vice President of Engineering at NextEra Energy. Led design and execution of 3.5 GW of solar PV across Texas and Arizona. Expert in solar PV system design, PV string sizing, single-axis tracker commissioning, and grid interconnection studies. Drove rapid shutdown compliance and NEC 690 compliance across the portfolio. Tools: PVsyst, SAM solar simulation, ETAP. Certifications: NABCEP PV Installation Professional, PE license, PMP certified, LEED AP. Education: M.S. in Electrical Engineering from Stanford University. Previously at First Solar for 6 years.",
            "entities": {
                "16 years of experience": "YEARS_EXP",
                "NextEra Energy": "EMPLOYER",
                "solar PV system design": "SKILL", "PV string sizing": "SKILL",
                "single-axis tracker commissioning": "SKILL",
                "grid interconnection studies": "SKILL",
                "rapid shutdown compliance": "SKILL", "NEC 690 compliance": "SKILL",
                "PVsyst": "SKILL", "SAM solar simulation": "SKILL", "ETAP": "SKILL",
                "NABCEP PV Installation Professional": "CERT",
                "PE license": "CERT", "PMP certified": "CERT", "LEED AP": "CERT",
                "M.S. in Electrical Engineering": "DEGREE",
                "First Solar": "EMPLOYER",
            },
        },
        # --- R031: SMR Reactor Operator ---
        {
            "text": "9 years of experience in nuclear reactor operations. Currently at NuScale Power as Senior Reactor Operator on small modular reactor program. Holds NRC Senior Reactor Operator license for SMR design. Skilled in reactor startup procedures, reactor shutdown procedures, control room operations, and emergency operating procedures. Trained on technical specifications compliance and 10 CFR 50 compliance. Performs operator rounds and ALARA program implementation. Certifications: NRC Senior Reactor Operator license, NRC Reactor Operator license, OSHA 30. Education: B.S. in Nuclear Engineering from Penn State University. Previously at Exelon for 4 years in BWR operations.",
            "entities": {
                "9 years of experience": "YEARS_EXP",
                "NuScale Power": "EMPLOYER",
                "reactor startup procedures": "SKILL", "reactor shutdown procedures": "SKILL",
                "control room operations": "SKILL",
                "emergency operating procedures": "SKILL",
                "technical specifications compliance": "SKILL",
                "10 CFR 50 compliance": "SKILL",
                "operator rounds": "SKILL", "ALARA program implementation": "SKILL",
                "NRC Senior Reactor Operator license": "CERT",
                "NRC Reactor Operator license": "CERT",
                "OSHA 30": "CERT",
                "B.S. in Nuclear Engineering": "DEGREE",
                "Exelon": "EMPLOYER",
            },
        },
        # --- R032: Directional Drilling Engineer ---
        {
            "text": "11 years of experience in directional drilling and MWD operations. Working at Schlumberger as Senior Drilling Engineer. Expert in directional drilling engineering, extended reach drilling, MWD measurement while drilling operations, and LWD logging while drilling operations. Designed casing programs for horizontal wells reaching 12,000 ft TVD. Skills: torque and drag modeling, hole cleaning analysis, kick detection and response. Certifications: IADC WellSharp engineer/supervisor well control, API 5CT, OSHA 30, NEBOSH IGC certified. Education: M.S. in Petroleum Engineering from Texas Tech University. Previously at Halliburton for 4 years in cementing engineering.",
            "entities": {
                "11 years of experience": "YEARS_EXP",
                "Schlumberger": "EMPLOYER",
                "directional drilling engineering": "SKILL",
                "extended reach drilling": "SKILL",
                "MWD measurement while drilling operations": "SKILL",
                "LWD logging while drilling operations": "SKILL",
                "torque and drag modeling": "SKILL",
                "hole cleaning analysis": "SKILL",
                "kick detection and response": "SKILL",
                "IADC WellSharp engineer/supervisor well control": "CERT",
                "API 5CT": "CERT", "OSHA 30": "CERT",
                "NEBOSH IGC certified": "CERT",
                "M.S. in Petroleum Engineering": "DEGREE",
                "Halliburton": "EMPLOYER",
            },
        },
        # --- R033: Ammonia / e-Fuels Engineer ---
        {
            "text": "6 years of experience in clean fuels and decarbonization. Currently at Monolith as Senior Process Engineer for green ammonia production. Expertise in ammonia synthesis for energy carrier, methanol synthesis for e-fuels, and PEM electrolyzer operations. Led pilot plant for 100 MW green ammonia facility. Skilled in process simulation in Aspen Plus and hydrogen safety management. Certifications: PE license, Certified Energy Manager, ISA-IEC 62443 cybersecurity expert. Education: M.S. in Chemical Engineering from MIT. Prior experience at Twelve in CO2 utilization research.",
            "entities": {
                "6 years of experience": "YEARS_EXP",
                "Monolith": "EMPLOYER",
                "ammonia synthesis for energy carrier": "SKILL",
                "methanol synthesis for e-fuels": "SKILL",
                "PEM electrolyzer operations": "SKILL",
                "process simulation in Aspen Plus": "SKILL",
                "hydrogen safety management": "SKILL",
                "PE license": "CERT", "Certified Energy Manager": "CERT",
                "ISA-IEC 62443 cybersecurity expert": "CERT",
                "M.S. in Chemical Engineering": "DEGREE",
                "Twelve": "EMPLOYER",
            },
        },
        # --- R034: Direct Air Capture Engineer ---
        {
            "text": "5 years of experience in carbon capture utilization and storage. Working at Climeworks as Direct Air Capture Engineer. Expertise in direct air capture operations, carbon capture, and CO2 utilization. Designed sorbent regeneration cycle for 4,000 ton/year DAC plant. Skilled in lifecycle assessment, sustainability reporting, and GHG inventory accounting. Tools: HYSYS, Python. Certifications: LEED AP, Certified Energy Manager, PE license. Education: M.S. in Chemical Engineering from ETH Zurich. Previously at Heirloom Carbon for 2 years.",
            "entities": {
                "5 years of experience": "YEARS_EXP",
                "Climeworks": "EMPLOYER",
                "direct air capture operations": "SKILL", "carbon capture": "SKILL",
                "lifecycle assessment": "SKILL", "sustainability reporting": "SKILL",
                "GHG inventory accounting": "SKILL",
                "HYSYS": "SKILL", "Python": "SKILL",
                "LEED AP": "CERT", "Certified Energy Manager": "CERT",
                "PE license": "CERT",
                "M.S. in Chemical Engineering": "DEGREE",
                "Heirloom Carbon": "EMPLOYER",
            },
        },
        # --- R035: EV Charging Infrastructure Manager ---
        {
            "text": "8 years of experience in EV charging infrastructure and electrification. Currently at ChargePoint as Director of Network Operations. Led deployment of 12,000 fast-charging stations across North America. Expert in EV fleet management, electric truck charging infrastructure, and grid interconnection engineering. Skilled in V2G vehicle-to-grid integration and demand response program design. Tools: SCADA, Power BI. Certifications: PE license, PMP certified, OSHA 30. Education: M.S. in Electrical Engineering from Georgia Tech. Previously at Tesla for 4 years in supercharger network operations.",
            "entities": {
                "8 years of experience": "YEARS_EXP",
                "ChargePoint": "EMPLOYER",
                "EV fleet management": "SKILL",
                "electric truck charging infrastructure": "SKILL",
                "grid interconnection engineering": "SKILL",
                "V2G vehicle-to-grid integration": "SKILL",
                "demand response program design": "SKILL",
                "SCADA": "SKILL", "Power BI": "SKILL",
                "PE license": "CERT", "PMP certified": "CERT", "OSHA 30": "CERT",
                "M.S. in Electrical Engineering": "DEGREE",
                "Tesla": "EMPLOYER",
            },
        },
        # --- R036: BESS Battery Storage Engineer ---
        {
            "text": "7 years of experience in battery energy storage systems engineering. Working at Form Energy as Senior BESS Engineer. Expert in battery energy storage integration, BESS commissioning, and grid-forming inverter design. Led commissioning of 800 MWh storage project supporting renewable integration. Skilled in IBR stability analysis, frequency response studies, and renewable interconnection studies. Tools: ETAP, MATLAB. Certifications: PE license, NABCEP PV Installation Professional, OSHA 30. Education: M.S. in Electrical Engineering from UC Berkeley. Previously at Tesla for 3 years in Megapack engineering.",
            "entities": {
                "7 years of experience": "YEARS_EXP",
                "Form Energy": "EMPLOYER",
                "battery energy storage integration": "SKILL",
                "BESS commissioning": "SKILL",
                "grid-forming inverter": "SKILL",
                "IBR stability analysis": "SKILL",
                "frequency response studies": "SKILL",
                "renewable interconnection studies": "SKILL",
                "ETAP": "SKILL", "MATLAB": "SKILL",
                "PE license": "CERT",
                "NABCEP PV Installation Professional": "CERT",
                "OSHA 30": "CERT",
                "M.S. in Electrical Engineering": "DEGREE",
                "Tesla": "EMPLOYER",
            },
        },
        # --- R037: Microgrid / DERMS Architect ---
        {
            "text": "10 years of experience in grid modernization and distributed energy resources. Currently at Uplight as Microgrid Solutions Architect. Designed microgrid controllers for 50 critical-infrastructure sites. Expert in microgrid design, DERMS distributed energy resource aggregation, and virtual power plant operations. Skilled in volt/VAR optimization, conservation voltage reduction, and ADMS advanced distribution management. Holds NERC CIP compliance experience. Tools: Python, OSI PI. Certifications: NERC System Operator certification, PE license, ISA CAP certified, CompTIA Security+. Education: M.S. in Electrical Engineering from Carnegie Mellon. Previously at Duke Energy for 5 years.",
            "entities": {
                "10 years of experience": "YEARS_EXP",
                "Uplight": "EMPLOYER",
                "microgrid design": "SKILL",
                "DERMS distributed energy resource aggregation": "SKILL",
                "virtual power plant operations": "SKILL",
                "volt/VAR optimization": "SKILL",
                "conservation voltage reduction": "SKILL",
                "ADMS advanced distribution management": "SKILL",
                "NERC CIP compliance": "SKILL",
                "Python": "SKILL",
                "NERC System Operator certification": "CERT",
                "PE license": "CERT",
                "ISA CAP certified": "CERT",
                "CompTIA Security+": "CERT",
                "M.S. in Electrical Engineering": "DEGREE",
                "Duke Energy": "EMPLOYER",
            },
        },
        # --- R038: Geothermal Reservoir Engineer ---
        {
            "text": "9 years of experience in geothermal energy development. Currently at Fervo Energy as Senior Reservoir Engineer for enhanced geothermal systems. Expert in geothermal resource assessment, reservoir characterization, enhanced geothermal system design, and geothermal well drilling. Optimized binary cycle power plant operations for 50 MW EGS facility. Skilled in geothermal well testing and pressure transient analysis. Tools: Eclipse, Python, MATLAB. Certifications: PE license, IADC WellSharp engineer/supervisor well control. Education: Ph.D. in Petroleum Engineering from Stanford University. Previously at Schlumberger for 3 years in geothermal services.",
            "entities": {
                "9 years of experience": "YEARS_EXP",
                "Fervo Energy": "EMPLOYER",
                "geothermal resource assessment": "SKILL",
                "reservoir characterization": "SKILL",
                "enhanced geothermal system design": "SKILL",
                "geothermal well drilling": "SKILL",
                "binary cycle power plant operations": "SKILL",
                "geothermal well testing": "SKILL",
                "pressure transient analysis": "SKILL",
                "Python": "SKILL", "MATLAB": "SKILL",
                "PE license": "CERT",
                "IADC WellSharp engineer/supervisor well control": "CERT",
                "Ph.D. in Petroleum Engineering": "DEGREE",
                "Schlumberger": "EMPLOYER",
            },
        },
        # --- R039: LNG Terminal Operations Manager ---
        {
            "text": "14 years of experience in LNG terminal operations. Working at Cheniere Energy as Operations Manager for Sabine Pass export terminal. Led LNG liquefaction plant operations for 25 MTPA facility. Skilled in LNG processing, gas dehydration, amine treating, and sulfur recovery unit operations. Manages floating LNG (FLNG) operations contracts across 4 cargo vessels per month. Tools: HYSYS, Aspen Plus, OSIsoft PI. Certifications: API 510 certified, API 570 certified, PE license, NEBOSH IGC certified. Education: B.S. in Chemical Engineering from Texas A&M University. Previously at ExxonMobil for 7 years in LNG operations.",
            "entities": {
                "14 years of experience": "YEARS_EXP",
                "Cheniere Energy": "EMPLOYER",
                "LNG liquefaction plant operations": "SKILL",
                "LNG processing": "SKILL",
                "gas dehydration": "SKILL", "amine treating": "SKILL",
                "sulfur recovery": "SKILL",
                "floating LNG (FLNG) operations": "SKILL",
                "HYSYS": "SKILL", "Aspen Plus": "SKILL",
                "API 510 certified": "CERT", "API 570 certified": "CERT",
                "PE license": "CERT", "NEBOSH IGC certified": "CERT",
                "B.S. in Chemical Engineering": "DEGREE",
                "ExxonMobil": "EMPLOYER",
            },
        },
        # --- R040: Senior Corrosion Specialist ---
        {
            "text": "13 years of experience in corrosion engineering and pipeline integrity. Currently at Aramco Services as Principal Corrosion Specialist. Expert in cathodic protection, internal corrosion direct assessment, and risk-based inspection methodology. Manages corrosion management for 8,000 km of crude oil pipelines. Skilled in API 580 risk-based inspection methodology and damage mechanism analysis. Certifications: NACE CP4, NACE Senior Corrosion Technologist, NACE PCIM, NACE CIP Level 3, API 571 certified, API 580 certified. Education: Ph.D. in Materials Science from Imperial College London. Previously at Saudi Aramco for 6 years.",
            "entities": {
                "13 years of experience": "YEARS_EXP",
                "Aramco Services": "EMPLOYER",
                "cathodic protection": "SKILL",
                "internal corrosion direct assessment": "SKILL",
                "risk-based inspection methodology": "SKILL",
                "API 580 risk-based inspection methodology": "SKILL",
                "damage mechanism analysis": "SKILL",
                "NACE CP4": "CERT",
                "NACE Senior Corrosion Technologist": "CERT",
                "NACE PCIM": "CERT",
                "NACE CIP Level 3": "CERT",
                "API 571 certified": "CERT",
                "API 580 certified": "CERT",
                "Ph.D. in Materials Science": "DEGREE",
                "Saudi Aramco": "EMPLOYER",
            },
        },
    ]

    labeled = []
    for s in samples:
        tokens, tags = _tokenize_and_tag(s["text"], s["entities"])
        entry = {"tokens": tokens, "ner_tags": tags, "text": s["text"]}
        # Repeat each hand-labeled sample 80x to heavily weight domain-specific examples
        # (increased from 20x to ensure energy-sector entities are well-represented)
        labeled.extend([entry] * 80)
    return labeled


def generate_dataset(n_samples=3000, output_path=None):
    """Generate a full NER dataset and optionally save to JSON."""
    data = [generate_sample() for _ in range(n_samples)]
    # Add hand-labeled real resume samples
    data.extend(_hand_labeled_samples())

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {n_samples} synthetic + {len(_hand_labeled_samples())} hand-labeled -> {output_path}")

    return data


if __name__ == "__main__":
    generate_dataset(3000, Path(__file__).parent / "ner_training_data.json")
