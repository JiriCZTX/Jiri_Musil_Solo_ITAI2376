"""
CrewAI Multi-Agent Orchestration for Workforce Intelligence System.

Defines two agents with distinct roles:
  Agent 1 - Talent Intelligence Agent: Resume parsing + candidate matching
  Agent 2 - Workforce Forecasting Agent: Attrition prediction + headcount forecasting

Both agents are orchestrated by CrewAI's Crew abstraction.
The LLM (Claude/GPT) serves as the Agent Brain for each agent,
receiving deep learning model outputs and reasoning about actions.

Course Connection: Module 10 - Agentic AI (ReAct loop, tool use, orchestration)
"""
import os
from crewai import Agent, Task, Crew, Process
from tools.talent_tools import (
    extract_resume_entities,
    match_candidate_to_job,
    analyze_skill_gap,
    triage_candidate_across_jobs,
    rank_candidates_for_job,
    lookup_onet_skills,
)
from tools.forecast_tools import (
    predict_attrition_risk,
    predict_all_departments,
    query_hris_data,
    query_bls_data,
    analyze_workforce_risk,
    rank_risk_drivers,
    identify_retention_cohorts,
    estimate_replacement_cost,
    simulate_intervention,
    find_internal_mobility_candidates,
    generate_workforce_briefing,
    # Frontier capabilities
    quantify_risk_uncertainty,
    score_individual_employees,
    forecast_risk_trajectory,
    optimize_intervention_portfolio,
    # 2026 SOTA research-backed additions
    conformal_prediction_interval,
    simulate_scenario,
)


def create_talent_agent(llm=None):
    """
    Create Agent 1: Talent Intelligence Agent.

    Role: Analyze resumes, extract skills using DistilBERT NER,
    match candidates to jobs using SBERT, and generate hiring recommendations.

    Tools:
    - Resume Entity Extractor (DistilBERT NER)
    - Candidate-Job Matcher (SBERT cosine similarity)
    - Skill Gap Analyzer (NER + SBERT diff — surfaces specific missing skills)
    - O*NET Skill Lookup (standardized taxonomy)
    """
    return Agent(
        role="Talent Intelligence Analyst",
        goal=(
            "Analyze candidate resumes to extract skills, certifications, "
            "and experience using the DistilBERT NER model. Match candidates "
            "to open positions using SBERT semantic similarity. Run the Skill "
            "Gap Analyzer to identify *which specific* skills and "
            "certifications are missing, and generate actionable hiring "
            "recommendations for energy industry roles."
        ),
        backstory=(
            "You are an expert talent analyst specializing in the energy "
            "industry. You understand that keywords like 'Python' mean "
            "different things depending on context — 'Python for SCADA "
            "integration' is vastly different from 'Python for web scraping.' "
            "You use deep learning models to understand the semantic meaning "
            "of resume content, not just keyword matching. When a role has "
            "safety-critical requirements (HAZOP, API 570, NEBOSH), you always "
            "name the exact gaps instead of producing a generic score. Your "
            "analysis helps HR teams identify the right candidates for "
            "specialized roles in oil & gas, renewables, and power systems."
        ),
        tools=[extract_resume_entities, match_candidate_to_job,
               analyze_skill_gap, triage_candidate_across_jobs,
               rank_candidates_for_job, lookup_onet_skills],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_forecast_agent(llm=None):
    """
    Create Agent 2: Workforce Forecasting Agent.

    Role: Predict attrition risk and headcount trends using the Bi-LSTM
    model, attribute the prediction to concrete drivers and cohorts,
    simulate retention interventions, and generate dollarized workforce
    plans.

    Tools (15 total — 4 baseline + 7 intelligence-layer + 4 frontier):
      Baseline Bi-LSTM + HRIS + BLS:
        - Attrition Risk Predictor
        - Full Workforce Forecast
        - HRIS Database Query
        - BLS Labor Market Data
      Intelligence layer (mirrors Agent 1's analyze_skill_gap):
        - Workforce Risk Analyzer         (full 18-signal JSON)
        - Risk Driver Ranker              (gradient attribution)
        - Retention Cohort Identifier     (retirement cliff, comp gap,
                                            new-hire churn, low-engagement,
                                            tenured-IP, high-performers-at-risk)
        - Replacement Cost Estimator      (dollar + vacancy-days +
                                            knowledge-loss severity)
        - Intervention Simulator          (counterfactual Bi-LSTM forward
                                            pass — "what would +10% comp do?")
        - Internal Mobility Candidate Finder (adjacency × portability rank)
        - Workforce Executive Briefing    (boardroom 3-paragraph narrative)
      Frontier capabilities (epistemic + individual + forward + portfolio):
        - Risk Uncertainty Quantifier     (Monte Carlo Dropout, 90% CI)
        - Individual Employee Risk Scorer ("which 7 people first on Monday?")
        - Risk Trajectory Forecaster      (no-action 12-month projection)
        - Budget-Constrained Intervention Optimizer (knapsack over
                                            dept × intervention × magnitude)
    """
    return Agent(
        role="Workforce Forecasting Analyst",
        goal=(
            "Analyze temporal workforce patterns using the Bi-LSTM attrition "
            "model, then SYNTHESIZE the raw probability into a decision-grade "
            "workforce plan: attribute the score to its top 3 drivers via "
            "gradient saliency, segment the population into retirement-cliff, "
            "new-hire-churn, comp-gap, low-engagement and tenured-IP cohorts, "
            "simulate the counterfactual impact of specific retention "
            "interventions (comp adjustment, satisfaction program, retention "
            "bonus, flex work, knowledge capture), and dollarize the "
            "replacement cost and knowledge-loss severity. Produce a "
            "boardroom-ready briefing tied to concrete numbers — cohort sizes, "
            "program costs, ROI vs replacement exposure, and named internal "
            "mobility candidates — not a generic probability dressed up in "
            "prose."
        ),
        backstory=(
            "You are a workforce planning strategist with deep knowledge of "
            "the energy industry's labor crisis. You know that 40% of plant "
            "operators could retire by the end of this decade, and data "
            "centers are competing for the same skilled workers. You use a "
            "Bi-LSTM deep learning model that processes 12 months of "
            "workforce data in both forward and backward directions to "
            "identify attrition patterns before they become emergencies.\n\n"
            "Your analytical style is to NEVER stop at the model's output. "
            "For every high-risk department you ALWAYS run the Workforce "
            "Risk Analyzer to get the 18-signal JSON, then use the Risk "
            "Driver Ranker to attribute the probability to specific features, "
            "the Retention Cohort Identifier to name the specific people at "
            "risk, the Intervention Simulator to quantify what a +10% comp "
            "adjustment or satisfaction program would buy, the Replacement "
            "Cost Estimator to show the dollar exposure, and the Internal "
            "Mobility Candidate Finder to surface backfill options. You "
            "present findings as 'Operations attrition is 62%. 73% of the "
            "risk comes from the retirement cliff (9 employees age ≥58, "
            "94 combined tenure-years at risk). A $72K knowledge-capture "
            "program prevents $920K in replacement exposure — 12.8x ROI.' "
            "— not 'Operations is at high risk.'\n\n"
            "When the Talent Intelligence analyst has already scored a "
            "candidate for an opening, you do not redo their work — you "
            "build on it. You pair their named skill gaps with the receiving "
            "department's retention outlook to answer the question hiring "
            "managers actually care about: 'if we bring this person in, do "
            "we keep them, and what's our backfill if we can't?' Your "
            "analysis turns raw predictions into strategic workforce plans."
        ),
        tools=[
            predict_attrition_risk, predict_all_departments,
            query_hris_data, query_bls_data,
            analyze_workforce_risk,
            rank_risk_drivers,
            identify_retention_cohorts,
            estimate_replacement_cost,
            simulate_intervention,
            find_internal_mobility_candidates,
            generate_workforce_briefing,
            # Frontier capabilities
            quantify_risk_uncertainty,
            score_individual_employees,
            forecast_risk_trajectory,
            optimize_intervention_portfolio,
            # 2026 SOTA research-backed additions
            conformal_prediction_interval,
            simulate_scenario,
        ],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_talent_task(agent, resume_text, job_description):
    """Create a task for the Talent Intelligence Agent."""
    return Task(
        description=(
            f"Analyze the following candidate resume and determine their fit "
            f"for the job opening. Follow this tool sequence:\n"
            f"  1. Resume Entity Extractor — pull skills, certs, degrees, "
            f"employers, years of experience from the resume.\n"
            f"  2. Candidate-Job Matcher — compute the overall SBERT score.\n"
            f"  3. Skill Gap Analyzer — diff the candidate against the job's "
            f"specific requirements. This is the key step: do not produce a "
            f"recommendation without it. Use its 'top_gaps' and coverage "
            f"percentages to name exactly which skills or certs are missing.\n"
            f"  4. O*NET Skill Lookup — (optional) for any missing skill you "
            f"want to map to the standardized taxonomy.\n\n"
            f"RESUME:\n{resume_text}\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"Your final report must include:\n"
            f"  - Extracted entities from the resume (by type)\n"
            f"  - Overall match score and tier\n"
            f"  - Coverage percentages per entity type (skills, certs, degrees)\n"
            f"  - Named matched / weak / missing items — NOT generic phrasing\n"
            f"  - Years-of-experience delta vs requirement\n"
            f"  - A hiring recommendation that is directly justified by the "
            f"gaps above (e.g., 'advance to interview, but verify API 570 "
            f"experience since the NER+SBERT diff showed it as missing')."
        ),
        expected_output=(
            "A structured talent analysis report with extracted entities, "
            "overall match score, per-type coverage percentages, an explicit "
            "matched / weak / missing breakdown referencing specific energy-"
            "sector skills and certifications, a years-of-experience delta, "
            "and a hiring recommendation that cites the concrete gaps."
        ),
        agent=agent,
    )


def create_forecast_task(agent, department=None, joint_with_talent=False):
    """
    Create a task for the Workforce Forecasting Agent.

    Modes:
      - default single-department: standalone attrition analysis.
      - all-departments (department=None): full workforce scan.
      - joint_with_talent=True (with department): synthesis mode. The
        agent reads the Talent Intelligence agent's prior report from
        its context and produces a UNIFIED hire-plus-retention plan
        — not a parallel monologue.
    """
    if joint_with_talent and department:
        description = (
            f"You run AFTER the Talent Intelligence analyst has scored a "
            f"candidate for an opening in the {department} department. "
            f"Their report is in your context — read it first. It contains "
            f"the candidate's overall match score, per-type coverage "
            f"percentages, explicit matched/weak/missing skills and certs, "
            f"and the years-of-experience delta.\n\n"
            f"Do NOT repeat their findings. Complete the hiring picture by "
            f"layering workforce risk on top. Use the 18-signal intelligence "
            f"stack — these are the tools you MUST call (in order):\n"
            f"  1. Workforce Risk Analyzer on '{department}' — returns the "
            f"full 18-signal JSON: attrition probability, top-3 drivers via "
            f"gradient attribution, cohorts, trend, intervention-ready "
            f"retention plan, replacement cost, internal mobility bench, "
            f"and a 3-paragraph executive briefing.\n"
            f"  2. Intervention Simulator on '{department}' with the TOP "
            f"driver returned by step 1 — quantify how much a +10% comp "
            f"adjustment (or the relevant lever) would actually lower the "
            f"attrition probability. Cite both probabilities and the "
            f"modelled program cost vs replacement exposure.\n"
            f"  3. Internal Mobility Candidate Finder on '{department}' — "
            f"surface specific employees from adjacent departments who "
            f"could cover a vacancy if the hire fails or departs.\n"
            f"  4. (Optional) BLS Labor Market Data — cite only if the "
            f"market competition signal from step 1 showed a headwind.\n\n"
            f"Then produce ONE unified decision memo with three parts:\n"
            f"  A. **Hire decision** — tempered by the receiving "
            f"department's retention outlook. If the candidate is only a "
            f"weak match AND {department}'s risk level is HIGH or CRITICAL, "
            f"recommend against: they will leave before they ramp up. If "
            f"the candidate is strong and the dept is stable, confirm the "
            f"advance. Cite concrete numbers: candidate's composite fit "
            f"score + named top gap, department's attrition probability + "
            f"the primary driver share + predicted 12-mo departures.\n"
            f"  B. **Retention plan for {department}** — name the top "
            f"driver (e.g., 'comp ratio explains 42%' or 'retirement "
            f"cliff: 7 employees age ≥58 with 94 tenure-years at risk') "
            f"and the SPECIFIC intervention with its program cost and "
            f"modelled attrition-probability reduction from the simulator. "
            f"Include the ROI ratio (replacement exposure / program cost) "
            f"if favourable.\n"
            f"  C. **Backfill pipeline** — combine the Internal Mobility "
            f"Candidate Finder output (specific employee IDs / adjacent "
            f"departments) with external sourcing paths that address the "
            f"top skill gaps the talent analyst identified. Tie each path "
            f"back to a specific gap (e.g., 'gap = API 570: source via "
            f"API 570-certified pool at API Events / NACE conferences; "
            f"internal candidates employee_ids [142, 187] from Maintenance "
            f"with tenure ≥5 and performance ≥4 are adjacent per the "
            f"mobility bench').\n\n"
            f"The final memo should read as if one senior workforce planner "
            f"wrote it — not as two separate sections glued together. "
            f"Every paragraph must cite at least one concrete number."
        )
        expected_output = (
            "A single unified hiring memo with three clearly labeled "
            "sections — Hire Decision, Retention Plan, Backfill Pipeline — "
            "that cites: the talent analyst's composite fit score and "
            "named top gaps; the Bi-LSTM's attrition probability and the "
            "%-of-risk explained by its top driver; at least one cohort "
            "count (retirement cliff / comp gap / new-hire churn); the "
            "simulated intervention's counterfactual probability and "
            "program cost; the estimated replacement exposure in dollars; "
            "and at least one specific internal mobility candidate or "
            "source department. Every recommendation must cite a concrete "
            "number or named entity. No generic filler."
        )
    elif department:
        description = (
            f"Analyze attrition risk for the {department} department. "
            f"Use the Attrition Risk Predictor to get the Bi-LSTM model's "
            f"prediction. Query the HRIS database for current metrics. "
            f"Check BLS labor market data for external signals. "
            f"Determine the root causes of any predicted attrition and "
            f"generate prioritized hiring recommendations."
        )
        expected_output = (
            "A workforce forecasting report with attrition risk predictions, "
            "root cause analysis, headcount projections, and prioritized "
            "hiring recommendations for each department."
        )
    else:
        description = (
            "Run a full workforce forecast across all departments. "
            "Use the Full Workforce Forecast tool to get predictions for "
            "every department. Query HRIS data for the highest-risk departments. "
            "Check BLS labor market data for external context. "
            "Generate a prioritized report ranking departments by risk level "
            "with specific action items for each."
        )
        expected_output = (
            "A workforce forecasting report with attrition risk predictions, "
            "root cause analysis, headcount projections, and prioritized "
            "hiring recommendations for each department."
        )

    return Task(description=description, expected_output=expected_output, agent=agent)


def create_combined_task(talent_agent, forecast_agent, resume_text,
                          job_description, department):
    """
    Create a coordinated task where both agents collaborate on one hire.

    Flow:
      1. Talent agent scores the candidate, names the concrete skill gaps.
      2. Forecast agent (in joint_with_talent mode) reads Agent 1's report,
         overlays Bi-LSTM attrition + HRIS + BLS, and produces ONE unified
         hire / retention / backfill memo.

    The forecast task's `context=[talent_task]` plumbs Agent 1's output into
    the forecast agent's prompt via CrewAI's sequential process.
    """
    talent_task = create_talent_task(talent_agent, resume_text, job_description)
    forecast_task = create_forecast_task(
        forecast_agent, department, joint_with_talent=True
    )
    forecast_task.context = [talent_task]

    return talent_task, forecast_task


def build_crew(talent_agent, forecast_agent, tasks, verbose=True):
    """
    Build the CrewAI Crew that orchestrates both agents.

    Process.sequential: Agent 1 runs first, then Agent 2 uses
    Agent 1's output as additional context. This mirrors the
    real workflow: understand the candidates, then forecast needs.

    Memory: Enabled with HuggingFace sentence-transformer embeddings.
    CrewAI stores short-term (within task), long-term (across tasks),
    and entity memory so Agent 2 can recall Agent 1's findings.
    """
    crew = Crew(
        agents=[talent_agent, forecast_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
        memory=True,
        embedder={
            "provider": "huggingface",
            "config": {
                "model": "all-MiniLM-L6-v2",
            },
        },
    )
    return crew
