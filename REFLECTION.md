# Reflection

**Project:** Multi Agent Workforce Intelligence System for the Energy Industry
**Course:** ITAI 2376 Deep Learning, Spring 2026
**Student:** Jiri Musil (Solo)
**Professor:** Patricia McManus

## What worked well

The single best decision was treating the deep learning models as a tool layer underneath a deliberately separate orchestrator. That split paid off three different times during the build.

The same DistilBERT + GLiNER + ModernBERT + SBERT + Bi LSTM stack serves the proprietary brain (`brain.py`), the optional CrewAI path (`agents.py`), and the Streamlit boardroom dashboard (`dashboard.py`) without duplication. Tool calls go through one set of `set_*_engine` registrations at startup. When I caught the Bi LSTM double sigmoid bug late in the build, the fix was a single line in the model file. Nothing in the agent layer needed to know about it.

The proprietary brain itself worked better than I expected. A 20 cell decision matrix that maps fit tier by team risk to a named verdict (R1 through R20) turns out to be enough policy expression to cover every realistic boardroom question I throw at it, and it produces a memo in around 500 milliseconds with zero external API dependency. The same decision a CrewAI plus Claude Sonnet run would take 15 to 30 seconds and one tenth of a dollar to produce. The brain ships every memo with a full execution trace, so a hiring manager can see exactly which tools were called, in what order, with what arguments, and how long each took.

The multi brain consensus pattern paid off too. Running three independent decision procedures on the same hire (a sophisticated 20 cell matrix, a conservative 20 cell matrix, and a naive 2 signal heuristic baseline) is a way to detect cells where reasonable HR leaders would disagree. When all three agree, the verdict is durable. When two agree and one dissents, the memo automatically routes to a senior reviewer. When all three disagree, the memo routes to a hiring committee. This pattern came out of the EU AI Act Article 14 reading on human oversight for high risk AI, and it adds about 80 milliseconds of latency for a meaningful safety property.

The 18 plus signal Workforce Risk Analyzer was the other place where putting work into the tool layer paid off more than putting it into the reasoning layer. Agent 2 was a probability reader before this analyzer existed; afterward it became a planner that names the top driver, segments the population into retirement cliff, new hire churn, and comp gap cohorts, simulates what a $72,000 knowledge capture program would do to attrition probability, dollarizes the replacement exposure, and produces a three paragraph executive briefing.

## What did not work and how I handled it

The Bi LSTM was bimodal at 0.00 and 1.00 across all eight departments for almost two weeks before I figured out why. AUC alone did not flag it. Only when I plotted the probability distribution did the saturation become visible. There were two separate causes wired together. A `StandardScaler` fit on Colab synthetic data was being loaded against differently distributed local CSVs, and the classification head ended in `nn.Sigmoid()` while training used `BCEWithLogitsLoss` (which applies sigmoid internally). The net effect was a forward pass computing `sigmoid(sigmoid(logit))`, which saturates in the range (0.5, 0.731). Gradients near zero on confident examples drove the loss floor to about 0.31 and the model compensated by pushing every logit toward plus or minus infinity.

After both fixes the probability spread became 0.12 to 0.57 (continuous, not bimodal), the Brier score landed at 0.232 (near the theoretical floor for a 50/50 balanced label), and the intervention simulator started returning differentiated counterfactual numbers like negative 84 percentage points at high risk and negative 26 at mid risk, instead of negative 99.7 percent across the board. The takeaway is now a discipline I apply to every model I touch. AUC and accuracy are necessary but not sufficient. Calibration metrics (Brier score, reliability) and the shape of the probability distribution are first class diagnostics, not optional extras.

A second false start was on the NER side. The early DistilBERT fine tune reported F1 of 0.9999 on a synthetic validation split, which was meaningless because the validation set saw the same templates as training. I caught it when the same model scored 0.60 on five real energy resumes I hand labelled. I rebuilt the held out set from real annotated resumes (twenty from a public dataturks set plus five hand labelled energy resumes) and rescored everything via span level seqeval. Honest held out F1 dropped to 0.74 for DistilBERT alone and 0.78 for the ensemble. The lesson here is that evaluation rigor matters more than headline numbers. I keep the held out set frozen now and rescore against it after every model change.

The third false start was on the framework side. CrewAI works, but the round trip latency of an LLM call inside every reasoning step put the joint hire memo at about thirty seconds end to end. For a CHRO clicking through a queue of fifty open requisitions, that adds up. I built the proprietary brain in parallel, expecting it to be a fallback, and it ended up the default.

## Biggest technical challenge and how I solved it

The hardest call was framework choice, not code. The blueprint committed to CrewAI plus Claude Sonnet 4 as the brain. Building it that way and getting it to work took less than a day. The harder question was whether to ship it as the primary path. Workforce planning decisions are audited, regulated, and contestable. Every "do not advance" recommendation can be challenged on bias grounds, and the EU AI Act treats HR systems as high risk for a reason. An LLM brain produces fluent narratives but cannot explain in plain language what rule led to a specific verdict. A 20 cell matrix with named cells (R1 strong stable through R20 decline critical) can. I ended up writing both runtimes and shipping the deterministic one as the default, with CrewAI kept wired and reachable through `--mode demo` for cases where free text generation actually adds value (resume coaching, job description drafting, multi turn interview prep dialogue).

The technical follow up to that decision was building the counterfactual flip explainer. When the deterministic verdict is DEFER or DECLINE, the natural follow up question from a hiring manager is "what would have to change for this to be hirable". The brain answers it deterministically by enumerating minimal single lever changes that would move the (fit tier, team risk) cell into a positive cell. Two kinds of flip levers are evaluated. Candidate side, where a named skill or certification gap closes, lifting the candidate tier into a hire family. Team side, where a retention intervention drops the team's predicted attrition probability across a band threshold. Each path is graded as high, medium, or low feasibility based on its causal status (causal, mixed, correlational) and the implied lead time. The result is an explainer the hiring committee can act on without asking the model another question.

## Path change from the midterm

I stayed Option B (multi agent). Two distinct agent roles, Talent Intelligence and Workforce Forecasting, communicate through a joint synthesis pipeline. The framework choice changed (custom orchestrator instead of CrewAI as the primary), but the agent count and roles match the midterm blueprint. CrewAI remains wired as the optional LLM path so the original commitment is honored.

What grew beyond the midterm scope was the depth of each agent's tool layer (six tools on Agent 1 and seventeen on Agent 2, totaling 21), the rigor of the evaluation methodology (a 25 document, 840 span adjudicated dev sidecar built through a multi stage review pipeline documented in `docs/EVAL_METHODOLOGY.md`), and the safety patterns layered on top (multi brain consensus, dual MC Dropout plus Split Conformal uncertainty intervals, counterfactual flip explainer, Human in the Lead feedback memory).

## What I would build next with another semester

Three concrete extensions, ranked by impact for an actual ABB pilot.

First, replace the gradient times input driver attribution with SHAP. Gradient saliency is a first order approximation. Shapley values are axiomatically correct and more defensible for HR decisions where a department head will challenge a "75 percent of risk comes from comp ratio" claim. SHAP also lets me run cohort level attribution, not just population.

Second, add a survival head to the Bi LSTM (Cox proportional hazards or a discrete time Weibull). The current model predicts whether someone will leave within six months. A survival head predicts when. "Median time to exit 4.2 months" is more actionable for CHRO planning than a binary risk score, and it composes naturally with the existing intervention simulator (each lever shifts the survival curve, not just the binary probability).

Third, build a real Workday or SuccessFactors connector. The synthetic temporal data inside the repo is enough to demonstrate the architecture, but a production pilot at ABB needs to read from the actual HRIS. Three weeks of integration work would replace `data/processed/individual_monthly.csv` with a live read of the previous 24 months of headcount, terminations, and engagement scores. The system is designed for that swap. Nothing in `tools/forecast_tools.py` cares whether the dataframe came from a CSV or a database query.
