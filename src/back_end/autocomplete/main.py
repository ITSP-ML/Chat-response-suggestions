from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from src.back_end.preprocess.autocomplete_preprocess import get_agent_msgs, preprocess_msg, change_recursion_limit

from src.back_end.autocomplete.model import build_trie, get_words_match
from transformers import AutoTokenizer, AutoModelWithLMHead
# load fine tuned dialogpt model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('models/trained_models/dialogpt/output-small')






app = FastAPI(root_path="/autocomplete")

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
    text: str = "hi there"
    nb_sugg :int = 10

class Item_2(BaseModel):
    text: str = "hi"



# read agent data
dataset_path = 'data/prod_v1/agent_data_3.csv'
dataset = get_agent_msgs(dataset_path)

# change recursion limit
limit = 10000000
change_recursion_limit(limit)

#build the Trie
validation_threshold = 0.9 # this a threshold that indicate that the child of a parent node will most likely been typed after the parent
max_number_of_words = 999
t = build_trie(dataset, validation_threshold)
t_max = build_trie(dataset, validation_threshold, max_number_of_words) # this tree dose't have any limit on the number of dfs


def predict(context):
        new_user_input_ids = tokenizer.encode(context+ tokenizer.eos_token, return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = new_user_input_ids

        # generated a bot_response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id
        )
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_response



@app.get("/")
async def home():
    return 'API is working'



# final autocomplete
@app.post("/search")
async def root(data: Item):
    prefix = data.text
    nb_suggs = data.nb_sugg
    # preprocess prefix 
    pre_prefix = preprocess_msg(prefix) # clean and preprocess the new user prefix
    words_match = get_words_match(pre_prefix, t, nb_suggs) # get matched sentences from the Trie
    return words_match

# final autocomplete
@app.post("/search_no_limit")
async def root(data: Item):
    prefix = data.text
    nb_suggs = data.nb_sugg
    # preprocess prefix 
    pre_prefix = preprocess_msg(prefix) # clean and preprocess the new user prefix
    words_match = get_words_match(pre_prefix, t_max, nb_suggs) # get matched sentences from the Trie
    return words_match


# dialogpt suggestions
@app.post("/dialogpt")
async def root(data: Item_2):
    context = data.text
    # preprocess prefix 
    context = preprocess_msg(context) # clean and preprocess the new user prefix
    return predict(context)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Autocomplete",
        version="1.0.2",
        description="Autocomplete feature with Auto correction of user inputs",
        routes=app.routes,
    )
    # change the input of the API
    openapi_schema["components"]["schemas"]["Item"] = {"title":"Item","required":["text"],"type":"object","properties":{"text":{"title":"Text","type":"string", "default":'hi there', "description": "prefix that agent already typed"},
                    "nb_sugg":{"title":"Nb Sugg","type":"integer", "description": "number of returned suggestions","default":10}}}
    # make the docs visible to the reverse proxy
    openapi_schema["servers"] = [ { "url": "/autocomplete" } ]
    # set the version of Openapi
    openapi_schema["openapi"] = "3.0.2"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi