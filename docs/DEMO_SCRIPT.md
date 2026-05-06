# Demo recording script

Target length: 4 minutes. Target audience: Professor McManus and the ITAI 2376 grader. Three scenarios plus a 30 second dashboard tour.

## Setup before recording

1. Open Terminal in the project root.
2. Activate the venv: `source venv/bin/activate`.
3. Open QuickTime, File, New Screen Recording.
4. Select the area that covers Terminal plus a browser window.
5. Have one Terminal pane and one Chrome tab on `http://localhost:8501` ready.

## Script (read this aloud while recording)

### 0:00 to 0:30 Title and problem framing

Hi, my name is Jiri Musil. This is my final project for ITAI 2376 Deep Learning, Spring 2026, with Professor McManus. I work as a People Strategy Lead for ABB Energy Industries in Houston, and the project I built solves a real problem from my day to day work.

The energy industry is short on people. The IEA reports 76 million energy jobs worldwide, with 2.4 workers near retirement for every new hire under 25. CERAWeek 2026 panels said 40 percent of plant operators could retire by the end of this decade. Accenture found that only 8 percent of energy organizations actually know what skills their workforce already has. My system attacks the visibility gap directly. It reads resumes and HRIS data, and produces one boardroom ready memo per hire that combines a hire decision, a retention plan, and a backfill pipeline.

This is Option B, multi agent. Two agents, Talent Intelligence and Workforce Forecasting, communicate through a single orchestrator.

### 0:30 to 1:30 Scenario 1, joint hire memo

Let me run the first scenario.

```
python main.py --mode agent --scenario joint_hire
```

(Wait for output. Point at it as you narrate.)

What you are seeing is the proprietary brain. It loaded four NER models (DistilBERT cased fine tuned, GLiNER, ModernBERT v11, plus a gazetteer overlay), the Sentence BERT semantic matcher, and the Bidirectional LSTM forecasting engine. It then ran twenty one tools across the two agents. About eight hundred milliseconds total.

The verdict here is HIRE, rule R5 hire stable. Look at the memo. Section A says the candidate fits at 74 out of 100, the Engineering team has a 31 percent attrition probability with a 28 to 34 percent confidence interval from MC Dropout, and the primary driver is age and retirement pressure at 24 percent of the model risk score. Section B is the retention plan, a knowledge capture program at $10,500 dollars targeting four people, with a modelled replacement exposure of $1.17 million dollars if we do nothing. That is an ROI of 111 to one. Section C is the backfill pipeline, five internal mobility candidates from the Projects department, named by employee ID, with mobility scores attached. Every number you see traces back to a verified tool output.

### 1:30 to 2:30 Scenario 2, multi brain consensus and counterfactual flip

Now let me show two safety patterns the brain runs by default. First, the multi brain consensus.

```
python main.py --mode agent --scenario multi_brain
```

Three independent decision procedures vote on the same hire. A sophisticated 20 cell matrix, a conservative 20 cell matrix, and a naive two signal heuristic. When all three agree, the verdict is durable. When two of three agree, the memo escalates to a senior reviewer. When all three disagree, the memo escalates to a hiring committee.

Now the counterfactual flip explainer. Run a deferred verdict.

```
python main.py --mode agent --scenario defer_hire
```

The verdict here is DEFER, rule R11 interview high risk. Section D enumerates what would have to change for this to flip. Candidate side levers, like closing a specific certification gap. Team side levers, like a comp adjustment or a satisfaction program at a specific magnitude. Each lever is graded high, medium, or low feasibility based on its causal status. This is an explainer the hiring committee can act on without asking the model another question.

### 2:30 to 3:15 Scenario 3, $500K quarterly retention plan

Let me show the budget bounded portfolio optimizer.

```
python main.py --mode agent --scenario retention
```

The optimizer evaluates 144 candidate actions across 8 departments times 6 interventions times 3 magnitudes. It ranks them by ROI in percentage points reduction per dollar million, and fills a 500 thousand dollar budget under a constraint that no department gets the same intervention twice. The output reads as a board ready table with per department rollup. In this run, 14 actions were selected, 99 percent of the budget got allocated, and the total reduction in attrition probability across the eight departments came out to 185 percentage points. That is the unit a CHRO planning a quarterly retention budget actually wants.

### 3:15 to 3:45 Streamlit boardroom dashboard

Now let me show the dashboard.

```
python main.py --mode dashboard
```

(Switch to browser tab on localhost 8501.)

The dashboard surfaces the same tools through a Streamlit UI. Top left, the Overview card describing the architecture. Talent Intelligence tab on the left has Resume Analysis, Candidate Ranking, and Skill Gap Analysis pages. Workforce Forecasting tab has Risk Overview, Department Drill Down, Headcount Projections, and Deep Risk Analysis. The Boardroom Brief tab runs the full proprietary brain workflows the same way the CLI does. The Human in the Lead tab lets a CHRO submit corrections that update the brain's decision matrix over time.

### 3:45 to 4:00 Wrap

This system is built on the deep learning stack we covered in class. Module 05 Transformers, with DistilBERT, GLiNER, ModernBERT, and Sentence BERT. Module 03 RNN and LSTM, with the Bidirectional LSTM. Module 10 Agentic AI, with the proprietary brain orchestration layer. The full source is on GitHub at the link in the README. Thank you.

## Backup terminal commands (in case the live ones do not run cleanly)

```
# 1. joint hire
python main.py --mode agent --scenario joint_hire

# 2. multi brain consensus
python main.py --mode agent --scenario multi_brain

# 3. counterfactual flip
python main.py --mode agent --scenario defer_hire

# 4. retention plan
python main.py --mode agent --scenario retention

# 5. dashboard
python main.py --mode dashboard
```

## Recording tips

* Keep the screen recording window large enough to read the terminal text. Set Terminal font to 14 point or 16 point before recording.
* Speak slowly. Pauses are fine; the grader watches at normal speed.
* Three minutes is the floor; four minutes is the ceiling. Five minutes is too long.
* Save as `demo.mp4`. Place the file at `demo/demo.mp4` in the repo root before pushing to GitHub. If the file is over 100 MB, re encode with `ffmpeg -i demo.mp4 -vcodec libx264 -crf 28 -preset slow -acodec aac -b:a 96k demo_compressed.mp4`.
