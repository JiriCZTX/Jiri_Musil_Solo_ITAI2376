"""
Workforce Intelligence System - Main Entry Point
=================================================
Multi-Agent System for Energy Industry Workforce Planning

Agent 1: Talent Intelligence (DistilBERT NER + SBERT Matching)
Agent 2: Workforce Forecasting (Bi-LSTM Attrition Prediction)
Framework: CrewAI (Multi-Agent Orchestration)

Jiri Musil | ITAI 2376 Deep Learning | Spring 2026
Houston City College | Professor Patricia McManus

Usage:
  python main.py --mode train     # Train both models
  python main.py --mode demo      # Run demo scenario
  python main.py --mode full      # Train + demo
  python main.py --mode dashboard # Launch Streamlit dashboard
"""
import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env
load_dotenv(Path(__file__).parent / ".env", override=True)

# Project imports
from config.settings import MODELS_DIR
from data.generate_ner_data import generate_dataset
from data.generate_workforce_data import generate_temporal_dataset, save_datasets
from data.sample_resumes import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
from models.ner_model import NEREngine, EnsembleNEREngine
from models.sbert_matcher import SBERTMatcher
from models.bilstm_model import ForecastingEngine
from tools.talent_tools import set_ner_engine, set_sbert_matcher
from tools.forecast_tools import set_forecast_engine, set_workforce_data


def train_models():
    """Train both deep learning models from scratch."""
    print("\n" + "=" * 60)
    print("PHASE 1: Training Deep Learning Models")
    print("=" * 60)

    # --- Agent 1: NER Model (Ensemble: DistilBERT + GLiNER) ---
    print("\n--- Training Ensemble NER Model (Agent 1) ---")
    print("Architecture: Fine-tuned DistilBERT + Zero-shot GLiNER (NAACL 2024)")
    print("Generating synthetic energy-sector NER training data...")
    ner_data = generate_dataset(500)  # Increased from 200 for better coverage

    # Train/val split
    split = int(0.8 * len(ner_data))
    train_data = ner_data[:split]
    val_data = ner_data[split:]
    print(f"Training samples: {len(train_data)}, Validation: {len(val_data)}")

    ner_engine = EnsembleNEREngine()
    ner_engine.distilbert.build_model()
    print(f"Model 1: DistilBERT ({sum(p.numel() for p in ner_engine.distilbert.model.parameters()):,} params)")
    print(f"Device: {ner_engine.distilbert.device}")

    history = ner_engine.distilbert.train(train_data, val_data)
    metrics = ner_engine.distilbert.evaluate(val_data)
    print(f"\nDistilBERT Evaluation - Weighted F1: {metrics['weighted_f1']:.4f}")
    ner_engine.save()

    # Load GLiNER component
    print("\nLoading GLiNER zero-shot extractor...")
    ner_engine.gliner.load()
    ner_engine._loaded = True

    # --- Agent 1: SBERT (pretrained, no fine-tuning needed) ---
    print("\n--- Loading SBERT Matcher (Agent 1) ---")
    sbert = SBERTMatcher()
    print(f"SBERT model: {sbert.model.get_embedding_dimension()}-dim embeddings")

    # Quick test
    score = sbert.compute_match_score(
        "Process safety engineer with HAZOP experience",
        "Looking for a process safety engineer skilled in HAZOP facilitation"
    )
    print(f"Test match score: {score:.4f}")

    # --- Agent 2: Bi-LSTM ---
    print("\n--- Training Bi-LSTM Forecasting Model (Agent 2) ---")
    print("Generating synthetic temporal workforce data...")
    employee_df, monthly_df, individual_df = generate_temporal_dataset()
    print(f"Employees: {len(employee_df)}, Monthly records: {len(monthly_df)}")
    print(f"Individual monthly: {len(individual_df)}, Attrition rate: {individual_df['attrition'].mean():.2%}")

    forecast_engine = ForecastingEngine()
    forecast_engine.build_model()
    param_count = sum(p.numel() for p in forecast_engine.model.parameters())
    print(f"Model: Bi-LSTM ({param_count:,} params)")

    history = forecast_engine.train(individual_df, monthly_df, epochs=50)
    eval_metrics = forecast_engine.evaluate(monthly_df, individual_df)
    print(f"\nBi-LSTM Evaluation:")
    print(f"  AUC-ROC: {eval_metrics.get('auc_roc', 'N/A')}")
    print(f"  F1 Score: {eval_metrics.get('f1_score', 'N/A')}")
    print(f"  MAE Headcount: {eval_metrics.get('mae_headcount', 'N/A')}")
    forecast_engine.save()

    # Save datasets for dashboard
    save_datasets(Path(__file__).parent / "data")

    return ner_engine, sbert, forecast_engine, monthly_df, individual_df


def run_demo(ner_engine=None, sbert=None, forecast_engine=None,
             monthly_df=None, individual_df=None):
    """
    Run a demo scenario showing both agents in action.

    Scenario: A new Senior Process Safety Engineer position opens.
    Agent 1 analyzes candidates and matches them to the role.
    Agent 2 forecasts attrition risk for the Engineering department.
    CrewAI orchestrates both agents to produce a unified recommendation.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Demo - Multi-Agent Workforce Intelligence")
    print("=" * 60)

    # Initialize models if not provided
    if ner_engine is None:
        print("\nLoading trained models...")
        ner_engine = EnsembleNEREngine()
        ner_engine.load()
        sbert = SBERTMatcher()
        forecast_engine = ForecastingEngine()
        forecast_engine.load()
        _, monthly_df, individual_df = generate_temporal_dataset()

    # Register tools with global instances
    set_ner_engine(ner_engine)
    set_sbert_matcher(sbert)
    set_forecast_engine(forecast_engine)
    set_workforce_data(monthly_df, individual_df)

    # --- Demo Part 1: Agent 1 - Talent Intelligence ---
    print("\n--- Agent 1: Talent Intelligence ---")
    job = SAMPLE_JOB_DESCRIPTIONS[0]  # Senior Process Safety Engineer
    print(f"Job Opening: {job['title']}")

    print("\nExtracting entities from candidate resumes...")
    for resume in SAMPLE_RESUMES[:4]:
        entities = ner_engine.extract_entities(resume["text"])
        score = sbert.compute_match_score(resume["text"], job["text"])
        tier = sbert._score_to_tier(score)

        total = sum(len(v) for v in entities.values())
        print(f"\n  Candidate: {resume['name']} ({total} entities found)")
        print(f"  Skills: {entities.get('SKILL', ['none'])}")
        print(f"  Certs: {entities.get('CERT', ['none'])}")
        print(f"  Degrees: {entities.get('DEGREE', ['none'])}")
        print(f"  Employers: {entities.get('EMPLOYER', ['none'])}")
        print(f"  Experience: {entities.get('YEARS_EXP', ['none'])}")
        print(f"  Match Score: {score:.4f} ({tier})")

    print("\n\nFull candidate ranking:")
    rankings = sbert.rank_candidates(SAMPLE_RESUMES, job)
    for i, r in enumerate(rankings, 1):
        print(f"  {i}. {r['candidate_name']}: {r['match_score']:.4f} - {r['match_tier']}")

    # --- Demo Part 2: Agent 2 - Workforce Forecasting ---
    print("\n\n--- Agent 2: Workforce Forecasting ---")
    print("Running attrition predictions across all departments...\n")

    predictions = forecast_engine.predict_all_departments(monthly_df)
    print(f"{'Department':<16} {'Risk Level':<10} {'Attrition Prob':<15} "
          f"{'Current HC':<12} {'6-Mo Proj':<10}")
    print("-" * 63)
    for pred in predictions:
        print(f"{pred['department']:<16} {pred['risk_level']:<10} "
              f"{pred['attrition_probability']:<15.4f} "
              f"{pred['current_headcount']:<12} {pred['projected_headcount_6m']:<10}")

    # --- Demo Part 3: CrewAI Orchestration ---
    print("\n\n--- CrewAI Orchestration ---")
    print("Launching multi-agent workflow...\n")

    try:
        from agents import (
            create_talent_agent, create_forecast_agent,
            create_combined_task, build_crew,
        )

        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if api_key:
            # Configure LLM for CrewAI
            from crewai import LLM

            if os.getenv("ANTHROPIC_API_KEY"):
                llm = LLM(
                    model="anthropic/claude-sonnet-4-20250514",
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    temperature=0.3,
                )
                print("LLM Brain: Anthropic Claude Sonnet 4")
            elif os.getenv("OPENAI_API_KEY"):
                llm = LLM(
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0.3,
                )
                print("LLM Brain: OpenAI GPT-4o-mini")
            elif os.getenv("GOOGLE_API_KEY"):
                llm = LLM(
                    model="gemini/gemini-2.0-flash",
                    api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.3,
                )
                print("LLM Brain: Google Gemini 2.0 Flash")

            talent_agent = create_talent_agent(llm=llm)
            forecast_agent = create_forecast_agent(llm=llm)

            talent_task, forecast_task = create_combined_task(
                talent_agent, forecast_agent,
                SAMPLE_RESUMES[0]["text"],
                SAMPLE_JOB_DESCRIPTIONS[0]["text"],
                "Engineering"
            )

            crew = build_crew(talent_agent, forecast_agent,
                              [talent_task, forecast_task])
            result = crew.kickoff()
            print("\nCrewAI Result:")
            print(result)
        else:
            print("NOTE: No LLM API key found in .env file.")
            print("CrewAI orchestration requires an API key for the LLM brain.")
            print("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in .env")
            print("\nDemonstrating the tool outputs directly instead:\n")

            # Show what the tools return (the DL models work without an API key)
            from tools.talent_tools import extract_resume_entities, match_candidate_to_job
            from tools.forecast_tools import predict_attrition_risk, query_hris_data

            print("Tool: Resume Entity Extractor")
            result = extract_resume_entities.run(resume_text=SAMPLE_RESUMES[0]["text"])
            print(result)

            print("\nTool: Candidate-Job Matcher")
            result = match_candidate_to_job.run(
                candidate_text=SAMPLE_RESUMES[0]["text"],
                job_text=SAMPLE_JOB_DESCRIPTIONS[0]["text"]
            )
            print(result)

            print("\nTool: Attrition Risk Predictor")
            result = predict_attrition_risk.run(department="Engineering")
            print(result)

            print("\nTool: HRIS Database Query")
            result = query_hris_data.run(department="Operations")
            print(result)

    except ImportError as e:
        print(f"CrewAI import error: {e}")
        print("Install CrewAI: pip install crewai crewai-tools")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard.py"
    print(f"\nLaunching Streamlit dashboard...")
    print(f"Open http://localhost:8501 in your browser\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def run_agent_mode(scenario="joint_hire", engine="adaptive"):
    """
    Run the PROPRIETARY agent brain — deterministic, LLM-free.

    This mode replaces the CrewAI+Claude orchestration with a rule-based
    brain that composes the 21 @tool functions into end-to-end workflows
    and synthesizes boardroom memos via templated prose. Zero external
    API dependencies.

    Available scenarios:
      joint_hire        — Agent 1 + Agent 2 unified hire/retention/backfill memo
      defer_hire        — forces a DEFER/DECLINE cell (junior candidate × senior
                          role × high-risk team) to showcase the Counterfactual
                          Section D flip-explainer
      risk_scan         — company-wide risk scan with top-3 action items
      retention         — $500K quarterly retention plan (knapsack-optimized)
      shortlist         — rank sample candidates for the Senior Process Safety role
      triage            — triage one candidate across all sample jobs
      multi_brain       — 3-way consensus across Sophisticated + Conservative
                          + Heuristic brains; escalates on disagreement
      feedback_summary  — show the Human-in-the-Lead learning state
                          (adjustments, votes, pending reviews)

    Engine modes:
      adaptive (default) — Human-in-the-Lead feedback layer active; memos
                            gain learning metadata and may have verdicts
                            adjusted by prior human corrections.
      static             — feedback layer disabled; byte-identical to the
                            pre-feedback brain. Use for reproducibility
                            snapshots and regulatory audits.

    All five DL-backed scenarios share the same models (Ensemble NER,
    SBERT, Bi-LSTM) already loaded in memory.
    """
    print("\n" + "=" * 70)
    print("PROPRIETARY AGENT BRAIN — LLM-free orchestration")
    print("=" * 70)
    print("Loading deep-learning models + workforce data...\n")

    ner_engine = EnsembleNEREngine()
    ner_engine.load()
    sbert = SBERTMatcher()
    forecast_engine = ForecastingEngine()
    forecast_engine.load()

    import pandas as pd
    data_dir = Path(__file__).parent / "data"
    monthly_df = pd.read_csv(data_dir / "monthly_department.csv")
    individual_df = pd.read_csv(data_dir / "individual_monthly.csv")
    try:
        employee_df = pd.read_csv(data_dir / "employees.csv")
    except FileNotFoundError:
        employee_df = None

    # Register tools with their backing DL models.
    set_ner_engine(ner_engine)
    set_sbert_matcher(sbert)
    set_forecast_engine(forecast_engine)
    set_workforce_data(monthly_df, individual_df, employee_df)

    from brain import build_brain
    enable_learning = (engine == "adaptive")
    brain = build_brain(verbose=False, enable_learning=enable_learning)
    engine_label = "adaptive (Human-in-the-Lead ACTIVE)" if enable_learning \
                    else "static (learning DISABLED — reproducibility mode)"
    print(f"Brain instantiated — engine: {engine_label}\n")

    if scenario == "joint_hire":
        print("--- Scenario: Joint Hire Analysis ---")
        print("Candidate: SAMPLE_RESUMES[0]  (Carlos Mendez)")
        print("Job: SAMPLE_JOB_DESCRIPTIONS[0]")
        print("Department: Engineering\n")
        memo = brain.joint_hire_analysis(
            resume_text=SAMPLE_RESUMES[0]["text"],
            job_text=SAMPLE_JOB_DESCRIPTIONS[0]["text"],
            department="Engineering",
        )
    elif scenario == "defer_hire":
        # Borderline candidate (Fatima Al-Dosari, 8yr corrosion engineer
        # — partial match: strong materials background but gaps on
        # pipeline integrity specifics) against a Pipeline Integrity
        # Engineer JD, routed to a HIGH-risk receiving team (Commercial).
        # Designed to land on (INTERVIEW × HIGH) = R11 DEFER so the
        # counterfactual Section D surfaces both candidate-side skill
        # closures AND team-side interventions as flip levers.
        print("--- Scenario: Counterfactual Flip Explainer (forced DEFER) ---")
        print("Candidate: SAMPLE_RESUMES[18]  (Fatima Al-Dosari, 8yr corrosion)")
        print("Job: SAMPLE_JOB_DESCRIPTIONS[3]  (Pipeline Integrity Engineer)")
        print("Department: Commercial  (HIGH-risk team by baseline Bi-LSTM)\n")
        memo = brain.joint_hire_analysis(
            resume_text=SAMPLE_RESUMES[18]["text"],
            job_text=SAMPLE_JOB_DESCRIPTIONS[3]["text"],
            department="Commercial",
        )
    elif scenario == "risk_scan":
        print("--- Scenario: Workforce Risk Scan (all departments) ---\n")
        memo = brain.workforce_risk_scan()
    elif scenario == "retention":
        print("--- Scenario: $500K Quarterly Retention Plan ---\n")
        memo = brain.quarterly_retention_plan(budget_usd=500_000)
    elif scenario == "shortlist":
        print("--- Scenario: Candidate Shortlist for the Senior Process Safety role ---\n")
        candidates = [
            {"id": str(i), "name": r["name"], "text": r["text"]}
            for i, r in enumerate(SAMPLE_RESUMES)
        ]
        memo = brain.rank_candidates_for_req(
            job_text=SAMPLE_JOB_DESCRIPTIONS[0]["text"],
            candidates=candidates,
        )
    elif scenario == "triage":
        print("--- Scenario: Triage Carlos Mendez across all jobs ---\n")
        jobs = [
            {"id": str(i), "title": j["title"], "text": j["text"]}
            for i, j in enumerate(SAMPLE_JOB_DESCRIPTIONS)
        ]
        memo = brain.match_candidate_across_reqs(
            resume_text=SAMPLE_RESUMES[0]["text"],
            jobs=jobs,
        )
    elif scenario == "multi_brain":
        print("--- Scenario: Multi-Brain Consensus ---")
        print("Runs 3 independent brains on the same hire and diffs their")
        print("verdicts. Disagreement → escalate to human hiring committee.\n")
        memo = brain.multi_brain_consensus(
            resume_text=SAMPLE_RESUMES[0]["text"],
            job_text=SAMPLE_JOB_DESCRIPTIONS[0]["text"],
            department="Engineering",
        )
    elif scenario == "feedback_summary":
        print("--- Scenario: Human-in-the-Lead Learning State ---\n")
        from feedback import FeedbackStore, LearningEngine
        store = FeedbackStore()
        engine_obj = LearningEngine(store)
        summary = store.summarize()
        adj = engine_obj.compute_adjustments()
        print(f"Feedback events logged : {summary['total_events']}")
        print(f"Applied (active)        : {summary['applied']}")
        print(f"Pending (below threshold): {summary['pending']}")
        print(f"Latest event timestamp  : {summary['latest_event_timestamp']}")
        print(f"By type: {summary['by_type']}")
        print()
        print(f"Learning version : {adj['learning_version']}")
        print(f"State hash       : {adj['state_hash']}")
        print(f"Rule overrides active  : {len(adj['rule_overrides'])}")
        print(f"Confidence multipliers : {adj['confidence_multipliers']}")
        print(f"Causal updates active  : {len(adj['causal_updates'])}")
        print(f"New interventions      : {len(adj['new_interventions'])}")
        if adj["rule_overrides"]:
            print("\nActive rule overrides:")
            for key, ov in adj["rule_overrides"].items():
                print(f"  - {key:<30} -> {ov['new_verdict']:<30} "
                      f"({ov['source']}, {ov.get('evidence_votes', 1)} votes)")
        return
    else:
        print(f"Unknown scenario: {scenario}")
        return

    # Persist the memo to the history store so the dashboard Review Queue
    # can list it even across sessions. Only summary fields are logged.
    try:
        from feedback import MemoHistoryStore
        history = MemoHistoryStore()
        history.log({
            "memo_id": (memo.metrics or {}).get("memo_id"),
            "timestamp": memo.metrics.get("memo_id") and
                          __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "memo_type": memo.memo_type,
            "headline": memo.headline,
            "verdict": memo.verdict,
            "rule_applied": memo.rule_applied,
            "department": (memo.metrics or {}).get("department"),
            "review_priority": (memo.metrics or {}).get("review_priority"),
            "learning_version": (memo.metrics or {}).get("learning_version"),
            "feedback_state_hash": (memo.metrics or {}).get("feedback_state_hash"),
            "n_adjustments_applied": len(
                (memo.metrics or {}).get("adjustments_applied") or []
            ),
        })
    except Exception:
        # Non-fatal: memo-history logging is best-effort.
        pass

    # Render memo
    print(memo.to_markdown())
    print()
    print("=" * 70)
    print(f"Memo generated in {memo.total_elapsed_ms:.1f} ms "
          f"via {len(memo.execution_trace)} tool calls.")
    print(f"Brain confidence: {memo.confidence.get('level', '—')} "
          f"(score {memo.confidence.get('score', 0):.2f}).")
    print(f"No LLM API calls. No external dependencies. Fully deterministic.")
    print("=" * 70)


def run_feedback_mode():
    """
    Interactive CLI for submitting Human-in-the-Lead feedback events and
    viewing the learning state. Demo-grade — production would use the
    dashboard's 'Submit Feedback' tab with proper auth.
    """
    from feedback import (
        FeedbackStore, LearningEngine, USER_ROLES, EVENT_TYPES,
        capture_verdict_correction, capture_confidence_calibration,
        capture_causal_update, capture_general_comment,
        rollback_last_adjustment,
    )

    store = FeedbackStore()
    engine_obj = LearningEngine(store)

    print("\n" + "=" * 70)
    print("HUMAN-IN-THE-LEAD FEEDBACK  —  interactive CLI")
    print("=" * 70)
    print("Commands: 1=submit verdict correction  2=confidence calibration")
    print("          3=causal update  4=general comment  5=show summary")
    print("          6=rollback last adjustment  q=quit\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            break
        try:
            if cmd == "1":
                memo_id = input("  memo_id: ").strip()
                department = input("  department: ").strip()
                ft = input("  fit_tier (STRONG_HIRE/HIRE/INTERVIEW/CONDITIONAL/DO_NOT_ADVANCE): ").strip()
                rl = input("  risk_level (LOW/MEDIUM/HIGH/CRITICAL): ").strip()
                ov = input("  original verdict: ").strip()
                cv = input("  corrected verdict: ").strip()
                uid = input("  your user_id: ").strip()
                role = input(f"  your role ({'/'.join(USER_ROLES)}): ").strip()
                rationale = input("  rationale: ").strip()
                capture_verdict_correction(
                    store, memo_id=memo_id, department=department,
                    fit_tier=ft, risk_level=rl,
                    original_verdict=ov, corrected_verdict=cv,
                    user_id=uid, user_role=role, rationale=rationale,
                )
                print("  ✓ logged.")
            elif cmd == "2":
                memo_id = input("  memo_id: ").strip()
                lvl = input("  original confidence level (high/medium/low): ").strip()
                wc = input("  was correct in practice? (y/n): ").strip().lower() == "y"
                uid = input("  your user_id: ").strip()
                role = input(f"  your role: ").strip()
                rationale = input("  rationale: ").strip()
                capture_confidence_calibration(
                    store, memo_id=memo_id, original_level=lvl,
                    was_correct=wc, user_id=uid, user_role=role,
                    rationale=rationale,
                )
                print("  ✓ logged.")
            elif cmd == "3":
                interv = input("  intervention name: ").strip()
                ns = input("  new status (CAUSAL/MIXED/CORRELATIONAL/NONE): ").strip().upper()
                uid = input("  your user_id: ").strip()
                role = input(f"  your role: ").strip()
                rationale = input("  rationale: ").strip()
                capture_causal_update(
                    store, intervention=interv, new_status=ns,
                    user_id=uid, user_role=role, rationale=rationale,
                )
                print("  ✓ logged.")
            elif cmd == "4":
                memo_id = input("  memo_id (or blank): ").strip() or None
                comment = input("  comment: ").strip()
                uid = input("  your user_id: ").strip()
                role = input(f"  your role: ").strip()
                capture_general_comment(
                    store, memo_id=memo_id, comment=comment,
                    user_id=uid, user_role=role,
                )
                print("  ✓ logged.")
            elif cmd == "5":
                summary = store.summarize()
                adj = engine_obj.compute_adjustments(force=True)
                print(f"\n  Total events: {summary['total_events']}")
                print(f"  Applied: {summary['applied']}  Pending: {summary['pending']}")
                print(f"  Learning version: {adj['learning_version']}")
                print(f"  State hash: {adj['state_hash']}")
                print(f"  Active rule overrides: {len(adj['rule_overrides'])}")
                for key, ov in adj["rule_overrides"].items():
                    print(f"    {key} -> {ov['new_verdict']} "
                          f"({ov['source']}, {ov.get('evidence_votes', 1)} votes)\n")
            elif cmd == "6":
                rolled = rollback_last_adjustment(store)
                if rolled:
                    print(f"  ✓ rolled back event {rolled.event_id} "
                          f"({rolled.event_type})")
                else:
                    print("  (nothing to roll back)")
            else:
                print("  unknown command")
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"  error: {e}")

    print("\nBye.")


def main():
    parser = argparse.ArgumentParser(
        description="Workforce Intelligence System - Multi-Agent AI for Energy HR"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "demo", "full", "dashboard", "agent", "feedback"],
        default="full",
        help=(
            "Run mode: train (train models) / demo (CrewAI + Claude) / "
            "full (train+demo) / dashboard (Streamlit) / "
            "agent (proprietary LLM-free brain) / "
            "feedback (interactive Human-in-the-Lead submit + summary)"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=["joint_hire", "defer_hire", "risk_scan", "retention",
                 "shortlist", "triage", "multi_brain", "feedback_summary"],
        default="joint_hire",
        help="Scenario for --mode agent (default: joint_hire). "
             "defer_hire forces a DEFER/DECLINE cell to showcase the "
             "counterfactual Section D explainer. "
             "multi_brain runs the 3-way consensus workflow. "
             "feedback_summary shows the learning state.",
    )
    parser.add_argument(
        "--engine",
        choices=["adaptive", "static"],
        default="adaptive",
        help="Brain engine mode for --mode agent. "
             "adaptive (default): Human-in-the-Lead feedback active. "
             "static: byte-identical to pre-feedback brain for "
             "reproducibility snapshots and regulatory audits.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_models()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "full":
        ner_engine, sbert, forecast_engine, monthly_df, individual_df = train_models()
        run_demo(ner_engine, sbert, forecast_engine, monthly_df, individual_df)
    elif args.mode == "dashboard":
        launch_dashboard()
    elif args.mode == "agent":
        run_agent_mode(scenario=args.scenario, engine=args.engine)
    elif args.mode == "feedback":
        run_feedback_mode()


if __name__ == "__main__":
    main()
