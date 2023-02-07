from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import set_cwd



from src.preprocess.autocomplete_preprocess import get_agent_msgs, preprocess_msg, change_recursion_limit
from src.src_models.ngrames_model.model import build_trie, get_words_match







app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:5173/",
    "http://localhost:8000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    text: str



# read agent data
dataset_path = 'data/prod_v1/agent_data_3.csv'
dataset = get_agent_msgs(dataset_path)

# change recursion limit
limit = 10000000
change_recursion_limit(limit)

# build trie
validation_threshold = 0.9 # this a threshold that indicate that the child of a parent node will most likely been typed after the parent
# set the the max number of suggestion to be shown
max_nb_suggs = 10
t = build_trie(dataset, validation_threshold, max_number_of_suggestions = max_nb_suggs)

# sementic search
model_name = 'distilbert-base-nli-stsb-mean-tokens'
embbeding_path = "data/prod_v1/doc_embedding.pickle"
print('all_done')






@app.post("/")
async def root(data: Item):
    prefix = data.text
    # preprocess prefix 
    pre_prefix = preprocess_msg(prefix)
    words_match = get_words_match(pre_prefix, t)
    return words_match
    # return get_sementic_match(pre_prefix, dataset_2, model_name, embbeding_path, top_k_hits = 10,
    #                                          min_prob = min_prob,  max_number_of_words = 3)
