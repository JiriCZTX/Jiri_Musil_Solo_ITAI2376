# Evaluation methodology

This document describes the multi stage review pipeline that produced the canonical adjudicated dev sidecar at `data/processed/eval_dev_v1_adjudicated.{labels.json, provenance.jsonl, manifest.json}`. The sidecar is the held out evaluation set used to decide the per class router ownership for the four NER models in the system (DistilBERT v6, GLiNER, ModernBERT v11, gazetteer overlay).

## Why a separate dev sidecar exists

The original held out v7 dev set is 25 documents. For the legacy five label classes (SKILL, CERT, DEGREE, EMPLOYER, YEARS_EXP) that is enough support to compute meaningful per class F1. For the new five (TOOL, INDUSTRY, LOCATION, PROJECT, SOFT_SKILL) the per class supports drop to 2, 2, 0, and 4. Comparing models on those numbers is statistical theatre.

The adjudicated dev sidecar was built to support a fair per class evaluation across the four NER variants. It is a 25 document, 840 span set with 10 label classes, full PII discipline (masked text only, no source text in any payload), and per span human override records on the spans that were modified during review.

## Schema, ten labels (v7)

| Label | Definition |
| --- | --- |
| SKILL | Role relevant capability or competency |
| CERT | Named professional certification or license |
| DEGREE | Named academic degree, with field of study where applicable |
| EMPLOYER | Named employer organization |
| YEARS_EXP | Tenure surface or years of experience phrase |
| TOOL | Named tool, platform, system, or technology |
| INDUSTRY | Named industry or sector |
| LOCATION | Named place, place of work, or workplace context |
| PROJECT | Named project, program, initiative, or named effort phrase |
| SOFT_SKILL | Hard flagged soft skill surface, limited per the policy registry |

## The fourteen carry forward policy clauses

A policy registry (sha256 `9543c1060d4961a49b462f1e40056e59ff462941f80745ee5c8ae279e773aed8`) carries the fourteen working clauses that govern borderline labelling decisions. Each clause is named A through N and is referenced from per span override records when invoked. Examples: state abbreviation jurisdictions are not LOCATION (clause A); vendor mentions in possessive form are not EMPLOYER (clause D); generic one word PROJECT titles are excluded or held (clause E); awards are not CERT (clause M); professional capability is not SOFT_SKILL (clause N).

## The five canonical phases

Each phase emits its own audit run with a handoff record, validation report, and provenance log. Every phase enforces masked text only coordinates (no source text in any payload) and runs a deep walk forbidden field audit on every emitted JSON to catch leakage of `source_text`, `source_start`, `source_end`, or `local_predictions` keys.

### Phase A, repair overlay

Applies hard blocker repairs (boundary expansions, paired drops on overlap conflicts) on top of the four frozen candidate batches (a 3 document pilot, a 5 document batch, a 9 document batch, and an 8 document batch). Output: 853 spans across 25 documents.

### Phase B, quality preflight

A quality gate that re runs the full validation suite (round trip offsets, label schema check, coordinate space invariant, duplicate check, overlap and nesting, per document masked text hash match, AUTOMOBILE zero LOC invariant, CONSULTANT envelope check, forbidden field deep walk). Readiness: PHASE_B_PASS.

### Phase C, tool and skill apply

Applies the strict explicit reviewer decisions on the five SKILL to TOOL relabels (CAD, MPLS WAN, RAID, VPN, HRIS) and the twelve policy hold drops. Phase C-2 follows it to resolve two remaining canonical promotion blockers in the Payroll context (one keep current with explicit blocker clearance, one policy hold drop). Output: 840 spans, zero canonical promotion blockers remaining.

### Phase D, canonical metadata enrichment

Attaches twelve canonical metadata fields per span: annotation guide hash, policy registry hash, source candidate artifact, source stage, provenance chain, human review status, decision timestamp, coordinate space, masked text hash, candidate status, future canonical status placeholder, validation status. The phase is metadata only by design. It does not mutate any retained span's label, text, or offsets. Twelve hard gates pass. The clean Phase D directory is locked read only on acceptance.

### Phase E, canonical sidecar promotion

The terminal step. It is the only phase that writes to `data/processed/`. It requires explicit human approval before execution. It writes three files:

| File | Purpose |
| --- | --- |
| `eval_dev_v1_adjudicated.labels.json` | 840 records with per record metadata |
| `eval_dev_v1_adjudication_provenance.jsonl` | 840 lines with provenance chain and validation status per span |
| `eval_dev_v1_adjudicated.manifest.json` | Top level manifest with sha256 integrity bracket, override breakdown, and per document masked text hash |

After the write, every retained span carries `canonical_status="canonical_promoted"`. Of the 840 spans, 31 carry an explicit `human_override_record` (22 Phase A boundary expansions, one expand and drop conflict pair, two outer survivors of nested span resolution, five Phase C SKILL to TOOL relabels, one Phase C-2 keep current with blocker clearance). The other 809 carry a null override record because they passed the multi stage pipeline default.

## Multi stage review pipeline (per candidate batch)

Each of the four candidate batches goes through six sub stages before reaching Phase A:

1. **Primary annotation pass** on the masked text bundle. Spans drafted as a data artifact; offset resolution by a small standard library only Python helper.
2. **Secondary annotator pass** independently reviews the primary pass on the same masked text bundle. Produces accept, question, reject, relabel, boundary change, or proposed addition decisions.
3. **Cross check layer** applies an artifact, provenance, schema, and semantic quality assurance pass over the secondary review. Catches drift, schema breaks, and policy clause misapplication.
4. **Human worksheet** organizes unresolved items into policy buckets (the fourteen clauses A through N).
5. **Policy applied** under the carry forward registry. The `keep_primary_label` enum extension records cases where the secondary annotator questioned a primary label that the human reviewer chose to keep.
6. **Human review** with explicit `human_override_record` fields on any overrides. Flags PII surfaces and schema extension TODOs for downstream policy work.

## Audit and safety scorecard

Across the four candidate batches, six pipeline phases, and the canonical promotion run:

| Audit | Result |
| --- | --- |
| Forbidden source key deep walk on every emitted JSON | 0 hits |
| Offset round trip `masked_text[s:e] == text` (840 / 840) | pass |
| Duplicate spans across documents | 0 |
| Coordinate space invariant `coordinate_space == "masked_text"` | all pass |
| Label schema (ten label v7) | all pass |
| Per document masked text hash match against `eval_dev_v1.json` | 25 of 25 |
| `eval_blind_v1.json` accesses (any kind) | 0 |
| `data/processed/` writes outside of Phase E | 0 |
| Frozen candidate directory mutations | 0 |
| Router or model code patches during pipeline | 0 |
| `eval_dev_v1.json` mutations | 0 (sha256 unchanged at `0b5a643f...362a791f`) |
| Canonical promotion blockers remaining | 0 |

## Reproducibility

Every phase script is standard library only Python 3.11. Each emitted manifest includes a sha256 integrity bracket on `eval_dev_v1.json` (pre and post execution) and a per document masked text hash that anyone can recompute against the dev source. The canonical sidecar files in `data/processed/` carry the sha256 of each output file in the manifest, and a separate Phase E validation report in the original audit directory rechecks every count.

## Sanitization note

The names in the primary, secondary, and cross check stages are deliberately neutral throughout this documentation and inside the canonical sidecars (`primary_annotator`, `secondary_annotator`, `cross_check_layer`). The pipeline used model assisted annotation under human supervision. The human reviewer is the source of truth at every decision point. The neutral naming reflects the methodology shape rather than the engine identities.

## Open items for the next stage

These are documented but not addressed inside the current submission. They belong to a downstream training and benchmarking workstream.

1. Run a unified evaluator against the canonical sidecar that scores the four NER variants (DistilBERT v6, GLiNER champion, ModernBERT v11, GLiNER 10 type candidate) under one chunking and post processing pipeline, eliminating the support drift across model wrappers.
2. Reconcile the ModernBERT v11 wrapper truncation. The model trained at `max_len=2048`; the production wrapper truncates at 512.
3. Decide on schema v8, which would add label classes for apprenticeship, credential, training program, regulation, standard, reporting form, and curriculum framework. Twelve policy hold rows from Phase C and one from Phase C-2 are candidates for these new classes.
4. Decide on a true third party reviewer pass for highest confidence canonical promotion. The current methodology has cross engine independence at the secondary review stage but not at the primary annotation stage.
