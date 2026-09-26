# Stage 5 Code Validation Results

Validation was run against the Stage 4 package after the Stage 5 updates.

## Code validation
- Python syntax compilation: **PASS** for project `.py` files and Streamlit views.
- Stage 4 deterministic agent smoke tests: **PASS**.
- Main evaluation dataset: **87 messages** loaded successfully.
- Crisis-v3 adversarial dataset: **14 messages** loaded successfully.

## Important model-status note
The Stage 4 ZIP contains the classifier code and labelled datasets but does **not** contain trained classifier weights under `models/`. Therefore the evaluation runner correctly used the project's documented rule-based fallback during this validation run.

The fallback results are useful for validating the evaluation pipeline and safety rules, but they are **not** the project's neural-network performance figures.

## Deterministic crisis-v3 re-check
After the Stage 5 adversarial re-check, the deterministic safety layer achieved **100% escalation recall on the 14-message crisis-v3 dataset** in this validation run (12/12 escalation cases caught).

This is a result on the specific 14-message test set only and is not a guarantee of real-world crisis detection.

## Main 87-message fallback validation
With no trained model weights available, the fallback run produced:
- Category accuracy: 65.5%
- Severity accuracy: 69.0%
- Escalation recall: 72.7%
- Escalation precision: 100.0%

These figures are included only to document this reproducible fallback validation run. They should not replace the previously recorded full-system evaluation results obtained with the trained model.
