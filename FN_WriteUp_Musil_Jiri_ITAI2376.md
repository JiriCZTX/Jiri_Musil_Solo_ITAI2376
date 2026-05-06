# Final Write Up, Multi Agent Workforce Intelligence System for the Energy Industry

Jiri Musil
Department of Science, Technology, Engineering and Math, Houston City College
ITAI 2376 Deep Learning Artificial Intelligence
Professor Patricia McManus
May 2026

## 1. What the agent does

I work as a People Strategy Lead for ABB's Energy Industries division in Houston. The midterm blueprint for this project came directly from my day to day work: workforce planning, HR business partnering, and oversight of staffing across seven countries in the Americas. The system I built reads resumes and job descriptions, reads twelve months of HRIS time series for a department, and produces one boardroom ready memo that combines a hire decision, a retention plan for the receiving team, and a backfill pipeline for the role. Every number in the memo is a direct reference to a verified deep learning model output, not a synthesized approximation. The memo cites the rule that produced the verdict and ships with a full execution trace, so a senior reviewer can see which tools were called and what they returned.

The problem is real and I have lived inside it. The IEA's World Energy Employment 2025 report shows 76 million energy jobs worldwide, with 2.4 workers near retirement for every new hire under 25 in advanced economies. CERAWeek 2026 panels cited that 40 percent of plant operators could retire by the end of this decade. Accenture's research presented at the same event found that only 8 percent of energy organizations have real visibility into the skills their workforce already has, and 85 percent of executives say AI will transform their business while only 18 percent see actual benefits. The system I built attacks the visibility gap directly: it knows what skills a candidate has, knows what skills the team is about to lose, and produces a hiring recommendation that integrates both. The target user is a People Strategy Lead, an HR Business Partner, or a CHRO at a large energy company.

## 2. Option chosen, and the change from the midterm

I chose Option B (multi agent system) and stayed with it. Two distinct agent roles, Talent Intelligence and Workforce Forecasting, communicate through a single orchestrator. Agent 1 reads resumes and job descriptions, runs a routed NER ensemble plus Sentence BERT semantic match, and produces a 19 signal skill gap analysis that names the specific missing skills and certifications rather than a generic match score. Agent 2 reads HRIS time series, runs a Bidirectional LSTM with attention to predict attrition probability and headcount drift, and produces a 19 signal workforce risk analysis that segments cohorts, attributes the prediction to specific drivers, simulates the counterfactual impact of retention interventions, and dollarizes the replacement exposure.

The framework choice changed from the midterm blueprint. I committed to CrewAI plus Claude Sonnet as the agent brain. I built that path (it works, lives in `agents.py`, and remains active when an LLM API key is present) but the primary deliverable is a deterministic LLM free orchestrator I wrote from scratch in `brain.py`. The reason is the same reason I take this project seriously at work: workforce planning decisions are audited, regulated, and contestable. Every "do not advance" recommendation can be challenged on bias grounds, and the EU AI Act treats HR systems as high risk for a reason. An LLM brain produces fluent narratives that cannot explain in plain language what rule produced a specific verdict. A 20 cell decision matrix where every cell carries a named rule (R1 strong stable through R20 decline critical) and an explicit rationale can. CrewAI remains wired as the optional path for cases where free text generation actually adds value (resume coaching, drafting job descriptions, multi turn interview prep). For the actual hire and retention decisions this project solves, the LLM adds latency and cost without adding capability. Measured on the same joint hire scenario, the proprietary brain returns a memo in about 500 milliseconds; CrewAI plus Claude Sonnet 4 takes 15 to 30 seconds and roughly 0.03 to 0.10 dollars per memo.

## 3. Architecture and deep learning models

The architecture follows the INPUT to DL MODEL to AGENT BRAIN to TOOL to OUTPUT flow that the midterm blueprint specified, with one important refinement in the agent brain box. The agent brain is now a deterministic orchestrator by default, with the LLM path retained as an option. Each agent has its own deep learning models and tools, and both feed into the same brain.

| Model | Agent | Course module | Role |
| --- | --- | --- | --- |
| DistilBERT cased, fine tuned | 1 | Module 05 Transformers | Resume NER on the legacy five label classes |
| GLiNER zero shot (NAACL 2024) | 1 | Module 05 Transformers | Generalist NER coverage across all ten labels |
| ModernBERT v11 | 1 | Module 05 Transformers | NER specialist that owns the TOOL class |
| Sentence BERT all MiniLM L6 v2 | 1 | Module 05 Transformers | Cosine semantic match plus Hungarian assignment for the skill gap diff |
| Bidirectional LSTM with attention | 2 | Module 03 RNN and LSTM | Department level attrition probability and headcount forecast |
| Routed per class ensemble | 1 | Module 05 Transformers | Per class hard ownership across the four NER variants above |
| Custom proprietary brain | both | Module 10 Agentic AI | Plan, Act, Observe, Respond loop over 21 tools, no LLM dependency |

The four NER models cooperate through a per class router. Each class is owned by the model that wins on real held out evaluation. The legacy five labels (SKILL, CERT, DEGREE, EMPLOYER, YEARS_EXP) go to the DistilBERT v6 ensemble. TOOL goes to ModernBERT v11. The Gate 2 labels (INDUSTRY, LOCATION, PROJECT, SOFT_SKILL) go to the GLiNER 10 type candidate. This is not one big monolithic model dressed up as an ensemble; it is an explicit hard ownership decision per label class, backed by per class F1 against an adjudicated dev set.

## 4. The agent brain layer

The brain is an orchestrator over 21 tools. Five end to end workflows: joint hire analysis, workforce risk scan, quarterly retention plan, candidate ranking, and candidate to multiple jobs triage. A sixth workflow runs three independent decision procedures in parallel (sophisticated 20 cell matrix, conservative 20 cell matrix, naive two signal heuristic) and surfaces disagreement as an automatic escalation signal: strong consensus when all three brains agree, escalate to senior reviewer on a 2 to 1 split, escalate to a hiring committee when all three disagree. The pattern came out of my reading on EU AI Act Article 14 and on multiple specialized expert consensus methods, and it adds about 80 milliseconds of latency for a meaningful safety property.

A counterfactual flip explainer fires automatically on DEFER and DECLINE verdicts. It enumerates minimal single lever changes that would move the (fit tier by team risk) cell into a positive cell. Two kinds of flip levers are evaluated. Candidate side, where a named skill or certification gap closes, lifting the candidate's tier into a hire family. Team side, where a retention intervention drops the team's predicted attrition probability across a band threshold. Each path is graded as high, medium, or low feasibility based on its causal status (causal, mixed, correlational) and the implied lead time. The result is an explainer the hiring committee can act on without asking the model another question.

The Human in the Lead memory layer (`feedback.py`) gives the brain a small, controlled, traceable, and reversible adaptation channel. Six event types: verdict correction, rule override, confidence calibration, causal update, new intervention, and general comment. Aggregation happens through configurable thresholds (auto apply at three same direction corrections, confidence recalibration at ten events, causal status flip at five). High impact events (rule override, new intervention) require explicit approval rather than threshold voting. The brain consults the feedback store before every hire decision and stamps every memo with a `learning_version`, a `feedback_state_hash`, and a `review_priority`.

## 5. What worked, what did not, and the surprising thing I learned

What worked best was the decision to make the deep learning models a pure tool layer underneath an orchestrator that is independently swappable. The same DistilBERT, GLiNER, ModernBERT, SBERT, and Bi LSTM stack serves the proprietary brain, the optional CrewAI path, and the Streamlit boardroom dashboard. When I caught the Bi LSTM double sigmoid bug late in the build, the fix was a one line change to the model file. Nothing in the agent layer needed to know.

What did not work, twice. The Bi LSTM was bimodal at 0.00 and 1.00 across all eight departments for almost two weeks. AUC alone did not flag it. Only when I plotted the probability distribution did the saturation become visible. The cause was a redundant `nn.Sigmoid()` in the classification head plus `BCEWithLogitsLoss` during training, which applies sigmoid internally. The forward pass was computing `sigmoid(sigmoid(logit))`, gradients on confident examples went to zero, and the model compensated by pushing logits to plus or minus infinity. After the fix the spread became 0.12 to 0.57 and the Brier score landed at 0.232, near the theoretical floor for a balanced label. The second false start was the early NER fine tune reporting F1 of 0.9999 on a synthetic validation split, which was meaningless because the val set saw the same templates as training. Honest held out F1 dropped to 0.74 for DistilBERT alone and 0.78 for the ensemble after I rebuilt the held out set from real annotated resumes.

The surprising thing I learned is that "agentic" does not require an LLM brain. The standard mental model from the course readings is `agent = LLM in a loop`. Working through the five workflows in `brain.py` showed me that "agentic" is really three things wired together: tools, a reasoning loop, and memory. An LLM is one possible implementation of the reasoning step, not the only one. A 20 cell decision matrix with explicit rules is more defensible in an audited domain, runs in 500 milliseconds instead of 30 seconds, costs 0 dollars per memo instead of 0.03 to 0.10, and has no hallucination surface because the prose is templated from verified tool outputs. In multi agent systems for high stakes domains, tool depth substitutes for LLM horsepower in a way that is both under appreciated and easier to engineer.

## 6. What I would build next with another semester

Three concrete extensions, in order of impact for a real ABB pilot.

First, replace the gradient times input driver attribution with SHAP. Gradient saliency is a first order approximation. Shapley values are axiomatically correct and more defensible when a department head challenges a "75 percent of risk comes from comp ratio" claim. Second, add a survival head to the Bi LSTM. The current model predicts whether someone will leave; a survival head (Cox proportional hazards or discrete time Weibull) predicts when. "Median time to exit 4.2 months" is more actionable for CHRO planning than a binary risk score. Third, build a real Workday or SuccessFactors connector to replace the synthesized monthly HRIS data with live reads of the previous 24 months. The architecture is designed for that swap.

## References

1. Vaswani et al., "Attention Is All You Need," NeurIPS 2017. https://arxiv.org/abs/1706.03762
2. Sanh et al., "DistilBERT, a distilled version of BERT," NeurIPS Workshop 2019. https://arxiv.org/abs/1910.01108
3. Reimers and Gurevych, "Sentence BERT," EMNLP 2019. https://arxiv.org/abs/1908.10084
4. Zaratiana et al., "GLiNER, Generalist Model for Named Entity Recognition using Bidirectional Transformer," NAACL 2024. https://arxiv.org/abs/2311.08526
5. Gal and Ghahramani, "Dropout as a Bayesian Approximation," ICML 2016. https://arxiv.org/abs/1506.02142
6. Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction and Distribution Free Uncertainty Quantification," 2021. https://arxiv.org/abs/2107.07511
7. International Energy Agency, "World Energy Employment 2025," December 2025. https://www.iea.org/reports/world-energy-employment-2025
8. Accenture Talent Reinventor research presented at CERAWeek 2026 (5,000+ executives surveyed). https://www.accenture.com/us-en/insights/consulting/talent-reinventors-delivering-value-people-age-ai
9. CrewAI Documentation, accessed April 2026. https://docs.crewai.com
10. EU AI Act, Regulation (EU) 2024/1689, Article 14 on human oversight of high risk AI systems.
11. CERAWeek 2026 sessions, March 23 to 27 2026, Houston Texas.
