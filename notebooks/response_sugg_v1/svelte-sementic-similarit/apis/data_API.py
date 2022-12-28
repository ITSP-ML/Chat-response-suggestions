import pandas as pd
from utils import match_suggs, rank_suggs
import set_cwd
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import set_cwd
import re
import pickle
from clean_chat import get_data, clean_text

import numpy as np

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
from annoy import AnnoyIndex
import random
import os
embedding_size = 768 # Length of item vector that will be indexed
n_trees = 256

data = pd.read_csv('data_dump/response_sugg/chat_data_MLDB_train_v1.csv')
# create a list of responses
agent_reposnes = data[data.user_id > 0]
unique_responses = agent_reposnes.msg.unique()
# clean 
clean_responses = [clean_text(x) for x in unique_responses]
# remove empty reposnes
clean_responses = [x for x in clean_responses if x != '']
# put the in a data frame
cleand_responses_df = pd.DataFrame({'responses': clean_responses})
cleand_responses_df.head()

cleand_responses_df['doc_len'] = cleand_responses_df['responses'].apply(lambda words: len(words.split()))
max_seq_len = np.round(cleand_responses_df['doc_len'].mean() + cleand_responses_df['doc_len'].std()).astype(int)
cleand_responses_df = cleand_responses_df[cleand_responses_df.doc_len <512]
# data encoding
# encoded_data = model.encode(cleand_responses_df.responses.tolist())
# encoded_data = np.asarray(encoded_data.astype('float32'))
encoded_data = []

index_path = 'C:/Users/feress/Documents/myprojects/Chat-response-suggestions/data_dump/response_sugg/sementic_search/agent_messages.index'
embbeding_path = "C:/Users/feress/Documents/myprojects/Chat-response-suggestions/data_dump/response_sugg/sementic_search/doc_embedding.pickle"
with open(embbeding_path, 'rb') as pkl:
    encoded_data = pickle.load(pkl)
if not os.path.exists(index_path):
    # Create Annoy Index
    print("Create Annoy index with {} trees. This can take some time.".format(n_trees))
    annoy_index = AnnoyIndex(embedding_size, 'dot')

    for i in range(len(encoded_data)):
        annoy_index.add_item(i, encoded_data[i])

    annoy_index.build(n_trees)
    annoy_index.save(index_path)
else:
    #Load Annoy Index from disc
    annoy_index = AnnoyIndex(embedding_size, 'dot')
    annoy_index.load(index_path)
# faiss.write_index(index, 'movie_plot.index')

# get_responses(similar_topics)
def transform_chat_to_json(df):
    all_chat_msgs = []
    for index, row  in enumerate(df.iterrows()):
        message_info = {}
        message_info["messageIndex"] =  index +1
        message_info["messageId"] =  row[1].msg_id
        message_info["message"] = row[1].msg
        message_info["sentByMe"] = True if row[1].user_id == 0 else False
        all_chat_msgs.append(message_info)
    response = {'messages':all_chat_msgs }
    return response


app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    id: int
    last_msg: int
    # curr_knowledge: str

class Item2(BaseModel):
    text: str

@app.post("/get_data")
async def root(post_data: Item):
    # global dataset
    example_id = post_data.id
    last_msg_id = post_data.last_msg
    # knowledge = post_data.curr_knowledge
    df = get_data(example_id, last_msg_id)
    # similar_topics = predict(pre_context)#, knowledge)
    # dataset = get_responses(similar_topics)
    # dataset['compare'] = dataset['Response'].apply(lambda x : x.lower())
    json = transform_chat_to_json(df)
    return json


@app.post("/sugg")
async def root(data: Item2):
    print(data.text)
    # print prefix
    prefix = str(data.text)
    print("prefix: ", prefix)  # PREFIX
    # match prefix to suggestions
    # unsorted_suggs = match_suggs(prefix, cleand_responses_df, model, annoy_index, top_k_hits = 10 )
    unsorted_suggs = match_suggs(prefix, cleand_responses_df, model, encoded_data, top_k_hits = 10 )
    final_suggs = rank_suggs(unsorted_suggs)

    return [{"sugg": x} for x in final_suggs]  # desired output shape for the frontend
    


#  uvicorn data_API/app --reaload --port 8034