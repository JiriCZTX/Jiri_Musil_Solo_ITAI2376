# Demo recording — verbatim talking points

Target length: 4 minutes (rubric range 3 to 5 minutes). Three scenarios from the Boardroom Brief plus a tour of the Human in the Lead layer. Read everything aloud except the bracketed stage cues.

Setup before recording:

1. Open the dashboard at http://localhost:8501 in Chrome. Pre run the Joint Hire Analysis so the memo for Carlos Mendez is on screen when you start.
2. Open QuickTime, File, New Screen Recording, select the browser window.

---

## 0:00 to 0:20 Opening

Hi, I am Jiri Musil. Final project for ITAI 2376 Deep Learning, Spring 2026, with Professor McManus. I work as a People Strategy Lead at ABB Energy Industries in Houston. The system I built solves the workforce visibility gap the energy industry is facing. Option B, two agents, deterministic LLM free brain over 23 tools, three deep learning models behind it. Three scenarios from the Boardroom Brief, then the Human in the Lead layer.

---

## 0:20 to 1:15 Scenario 1, Joint Hire Analysis

*(Boardroom Brief tab, Joint Hire memo on screen.)*

First scenario, Joint Hire Analysis. Carlos Mendez, Senior Process Safety Engineer, into Engineering. Verdict ADVANCE_TO_INTERVIEW under rule R9 interview stable. 940 milliseconds, 6 tool calls, zero LLM calls.

*(Section A.)*

Carlos at fit 65.8, INTERVIEW tier. The skill gap analyzer named three gaps to verify in interview: familiarity with, risk assessment, and MOC management.

*(Dual interval block.)*

This is the part I want to highlight. Engineering team 30 percent attrition probability. The brain reports two intervals on the same number. MC Dropout 27 to 33 percent. Split Conformal 0 to 100 percent. The gap is itself a diagnostic. The model is internally consistent but the Bi LSTM's held out accuracy cannot support a narrow guaranteed interval. The memo tells the reader to cite the conformal interval for audit.

*(Section B.)*

Top driver age and retirement at 24 percent. Knowledge Capture program for three people at $10,500, flagged causal. Replacement exposure $1.17 million. ROI 111 to 1.

*(Section C.)*

Five internal mobility candidates from Projects. Top is employee 219.

---

## 1:15 to 1:55 Scenario 2, Candidate Shortlist

*(Switch dropdown to Candidate Shortlist, click Generate.)*

Second scenario, Candidate Shortlist. Same role. The brain ranks 28 sample candidates in one tool call.

Carlos is the only candidate to clear the bar at 66, INTERVIEW. Every other top ten candidate lands in DO NOT ADVANCE between 32 and 40. Same pattern across rows: PE license but top gap P&ID review. The analyzer correctly weights the named critical gap above the generic credential.

Latency 23 seconds, one tool call. The 23 seconds is SBERT plus Hungarian assignment across 28 candidates inside that tool. The brain itself orchestrates in milliseconds.

---

## 1:55 to 2:50 Scenario 3, Multi Brain Consensus

*(Switch to Multi-Brain Consensus, click Generate.)*

Third scenario, Multi Brain Consensus. Safety pattern around EU AI Act Article 14, human oversight for high risk AI. Brain runs the same hire through three independent decision procedures in parallel.

*(Headline.)*

Same Carlos into Engineering. Headline: all three brains agree. STRONG_CONSENSUS, family ADVANCE.

*(Per brain table.)*

Three brains. Sophisticated returns ADVANCE_TO_INTERVIEW under R9. Conservative, stricter policy, returns ADVANCE_WITH_CAVEATS. Heuristic baseline, deliberately naive, returns ADVANCE_TO_INTERVIEW.

*(Recommendation.)*

Consensus is at the family level. When all three agree, normal review. Two of three, escalate to a senior reviewer. All three disagree, escalate to a hiring committee. Disagreement between independent procedures is itself a signal. It identifies the cells where reasonable HR leaders would disagree.

Latency 916 milliseconds, three brains in parallel, zero LLM calls.

---

## 2:50 to 3:45 Section 4, Human in the Lead

*(Click into the Human-in-the-Lead tab.)*

Last section, Human in the Lead. The brain learns from human corrections. Transparently, reversibly, and only when evidence crosses a voting threshold. The reviewer is always in the lead.

*(State counters.)*

Learning version v0 empty, state hash e3b0c44, events logged zero, overrides zero. Brain is in its base policy state. As feedback comes in, the version increments and every memo stamps the new hash in its footer.

*(Policy.)*

Verdict corrections auto apply at three same direction votes. Rule overrides and new interventions require explicit CHRO approval. That is the governance gate. A single actor cannot reshape the decision matrix by themselves.

*(Form.)*

Six event types: verdict correction, rule override, confidence calibration, causal update, new intervention, general comment. Every event captures the cell context, the original verdict, the corrected verdict, the user ID, the role, and a required rationale. Append only and reversible. One click rollback unapplies the most recent adjustment. That is the reversibility the EU AI Act expects for high risk AI.

---

## 3:45 to 4:00 Wrap

That is the system. Module 03 LSTM in the Bi LSTM. Module 05 Transformers in the routed NER ensemble and Sentence BERT. Module 10 Agentic AI in the brain. Source and the canonical adjudicated dev sidecar on GitHub. Thank you.

---

## Recording tips

* Browser font 14 or 16 point so the grader can read the screen without zooming.
* Steady pace. Sized for around 4 minutes at normal speaking speed.
* Pauses between sections are fine. The grader watches at normal speed.
* If you go long, cleanest cuts: the IEA opener line, "rule R9c" in scenario 3, the form description in section 4.
* Save as `demo.mp4` at `demo/demo.mp4` in the repo root before pushing. If the file is over 100 MB, recompress with ffmpeg.
