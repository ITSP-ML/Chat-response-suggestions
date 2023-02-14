# Chat-response-suggestions

**explain**

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

# Dev
- python src/APIs/prod_01/generate_embeddings.py : generate the embedding file by running this cmd
- uvicorn src.back_end.autocomplete.main:app --reload --port 8010 : uvicorn src.back_end.autocomplete.main:app --reload --port 8010
- uvicorn src.back_end.semantic_search.main:app --reload --port 8011 : run the sementic seaarch Api
- port=8007 npm run dev : run front_end