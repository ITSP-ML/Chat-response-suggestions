from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import set_cwd
import sys
import pandas as pd
import pickle
import re

from src.back_end.trie import Trie
from src.preprocess.autocomplete_preprocess import get_agent_msgs, preprocess_msg, change_recursion_limit, create_queries, get_words_match, get_sementic_match





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
dataset_path = 'data/prod_v1/agent_data.csv'
dataset = get_agent_msgs(dataset_path)

# change recursion limit
limit = 10000000
change_recursion_limit(limit)

# create query dict
queries = create_queries(dataset)

# build trie
t = Trie()
t.build_tree(queries)


# sementic search
model_name = 'distilbert-base-nli-stsb-mean-tokens'
embbeding_path = "data/prod_v1/doc_embedding.pickle"






@app.post("/")
async def root(data: Item):
    prefix = data.text
    # preprocess prefix 
    pre_prefix = preprocess_msg(prefix)
    # return get_words_match(pre_prefix, t)
    return get_sementic_match(pre_prefix, dataset, model_name, embbeding_path, top_k_hits = 10 )
