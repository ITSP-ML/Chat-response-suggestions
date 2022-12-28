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
from clean_chat import get_data
# Load model
from bertopic import BERTopic
import pandas as pd
topic_model = BERTopic.load("models/trained_models/context/BERTopic_v2")
# load_data
data = pd.read_csv('data_dump/response_sugg/context_clusters/data_v1.csv')
# get more topics per context
def show_example(data, topic_numbers, random_state = 2, all = False):
    if all == True:
        return data[data.context_topic_v1.isin(topic_numbers)].sort_values('context_topic_v1')
    return data[data.context_topic_v1.isin(topic_numbers)].sample(10, random_state= random_state).sort_values('context_topic_v1')
def get_responses(topic_list):
    # get responses
    topic_data = show_example(data, topic_list, all=True)
    topic_rank  = {x: i +1  for i, x in enumerate(topic_list)}
    topic_data['topic_rank'] = topic_data.context_topic_v1.map(topic_rank)
    return topic_data.sort_values("topic_rank")


# get_responses(similar_topics)
def transform_chat_to_json(df, similar_topics):
    all_chat_msgs = []
    for index, row  in enumerate(df.iterrows()):
        message_info = {}
        message_info["messageIndex"] =  index +1
        message_info["messageId"] =  row[1].msg_id
        message_info["message"] = row[1].msg
        message_info["sentByMe"] = True if row[1].user_id == 0 else False
        all_chat_msgs.append(message_info)
    response = {'messages':all_chat_msgs , 'similar_topics': similar_topics}
    return response
def predict(context):
    similar_topics, _ = topic_model.find_topics(context, top_n=5)
    return similar_topics


# similar_topics = [2]
# dataset = get_responses(similar_topics)
# dataset['compare'] = dataset['Response'].apply(lambda x : x.lower())
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
    global dataset
    example_id = post_data.id
    last_msg_id = post_data.last_msg
    # knowledge = post_data.curr_knowledge
    df, pre_context = get_data(example_id, last_msg_id)
    similar_topics = predict(pre_context)#, knowledge)
    dataset = get_responses(similar_topics)
    dataset['compare'] = dataset['Response'].apply(lambda x : x.lower())
    json = transform_chat_to_json(df, similar_topics)
    return json


@app.post("/sugg")
async def root(data: Item2):
    print(data.text)
    # print prefix
    prefix = str(data.text)
    print("prefix: ", prefix)  # PREFIX
    # match prefix to suggestions
    print(dataset.head())
    unsorted_suggs = match_suggs(prefix, dataset)
    final_suggs = rank_suggs(unsorted_suggs)

    return [{"sugg": x} for x in final_suggs]  # desired output shape for the frontend