# Stage 5 — Analytics & Evaluation Changelog

## Purpose
Stage 5 adds the evaluation and analytics layer without rewriting the Stage 4 AI agent.

## Changes
- Added `stage5_evaluation.py` to run the main 87-message evaluation and the separate 14-message crisis-v3 evaluation in one workflow.
- Added `stage5_analytics.py` to export administrative analytics from the SQLite interaction database.
- Retained `eval_classifier.py` as the core message-level evaluation implementation.
- Retained `train_classifier.py` as the model-training and cross-validation implementation.
- Kept the deterministic safety and escalation logic unchanged.
- Kept the deterministic category-to-office action planner unchanged.
- Corrected the office/contact knowledge-base filename alignment in `config.py` so it matches `ugbs_offices_and_contacts.md`.

## Evaluation scope
The main evaluation dataset contains 87 labelled messages. The crisis-v3 adversarial dataset contains 14 labelled crisis messages. Results must be reported with these dataset scopes and must not be generalized to all real-world student welfare cases.

## Model availability
The Stage 4 package does not include trained classifier weights. Therefore, an evaluation run made without `models/welfare_classifier.pt` will use the documented rule-based fallback. Such a run must not be described as a neural-network result.
