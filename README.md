# Chat-response-suggestions
# Prod 1
**explain**
**running guide**
- generate the embedding file by running this cmd: python src/APIs/prod_01/generate_embeddings.py
- run the Api : uvicorn src.APIs.prod_01.app:app --reload
- run front_end: port=8007 npm run dev
**future work**
- this version is only for english (multilingual will follow)
- Add spell correction
- Add history saving
# Heavy models weights should be kept outside of version control

Files stored in the [ML shared folder](\\ats-store01\BusinessIntelligence\ML\Chat-Response-Suggestions\models).

Copy the models under the models folder(/models) and **never** commit to git.

### models folder should contain :
- **intent_classifier_with_ANN**
- **ner_demo_replace**
- **rasa**

