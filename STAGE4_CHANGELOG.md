# Stage 4 Code Update — UGBS AI Student Welfare & Support Triage System

## What was updated

- Preserved the existing Stage 3 architecture and core triage logic.
- Corrected the knowledge-base filename to `ugbs_offices_and_contacts.md` and renamed the corresponding source file.
- Added the student, admin and settings Streamlit pages under `views/` so the router in `app.py` has its expected page structure.
- Added a lightweight `views/live_admin_interface.py` operational monitor.
- Added `stage4_agent_tests.py` to smoke-test category routing, office recommendations and crisis detection without requiring an LLM or trained neural-network weights.
- Corrected the settings page so its displayed retrieval threshold matches the implemented `CONFIDENCE_THRESHOLD = 1.0` in `views/user_interface.py`.
- Removed hard dependency on optional avatar image files by using Streamlit's default chat avatars when those assets are not present.

## Existing Stage 4 agent flow retained

Student input → crisis check → chitchat handling → query normalization → neural classifier/rule fallback → LLM fallback when unclassified → safety/urgency/escalation finalization → knowledge-base retrieval → grounded answer → deterministic office routing → action plan → SQLite logging → admin analytics.

## Important runtime requirements

The package does not include trained model weights or a generated Chroma vector store. Run the training and vector-store build steps before relying on those components:

```bash
python train_classifier.py
python build_vectorstore.py
streamlit run app.py
```

The configured default LLM provider is Ollama. If Ollama is not available, the application has deterministic/template fallbacks in the existing code.

## Prototype limitation

The admin interface is not production-secured. Authentication, role-based access, retention/deletion controls and a formal privacy governance process are still required before deployment with real student welfare data.
