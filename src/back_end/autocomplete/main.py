from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from src.back_end.preprocess.autocomplete_preprocess import get_agent_msgs, preprocess_msg, change_recursion_limit

from src.back_end.autocomplete.model import build_trie, get_words_match







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



# read agent data
dataset_path = 'data/prod_v1/agent_data_3.csv'
dataset = get_agent_msgs(dataset_path)

# change recursion limit
limit = 10000000
change_recursion_limit(limit)

#build the Trie
validation_threshold = 0.9 # this a threshold that indicate that the child of a parent node will most likely been typed after the parent
t = build_trie(dataset, validation_threshold)




@app.get("/")
async def home():
    return 'API is working'




@app.post("/search")
async def root(data: Item):
    prefix = data.text
    nb_suggs = data.nb_sugg
    # preprocess prefix 
    pre_prefix = preprocess_msg(prefix) # clean and preprocess the new user prefix
    words_match = get_words_match(pre_prefix, t, nb_suggs) # get matched sentences from the Trie
    return words_match


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = {"openapi":"3.0.2","info":{"title":"Autocomplete","description":"Autocomplete feature with Auto correction of user inputs","version":"1.0.1"},
                      "paths":{"/":{"get":{"summary":"Home","operationId":"home__get","responses":{"200":{"description":"Successful Response",
                        "content":{"application/json":{"schema":{}}}}}}},"/autocomplete":{"post":{"summary":"Root","operationId":"root_autocomplete_post",
                        "requestBody":{"content":{"application/json":{"schema":{"$ref":"#/components/schemas/Item"}}},"required":True},"responses":{"200":{"description":"Successful Response",
                        "content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}}},
                    "components":{"schemas":{"HTTPValidationError":{"title":"HTTPValidationError","type":"object","properties":{"detail":{"title":"Detail","type":"array",
                    "items":{"$ref":"#/components/schemas/ValidationError"}}}},"Item":{"title":"Item","required":["text"],"type":"object","properties":{"text":{"title":"Text","type":"string", "default":'hi there', "description": "prefix that agent already typed"},
                    "nb_sugg":{"title":"Nb Sugg","type":"integer", "description": "number of returned suggestions","default":10}}},"ValidationError":{"title":"ValidationError","required":["loc","msg","type"],"type":"object",
                    "properties":{"loc":{"title":"Location","type":"array","items":{"anyOf":[{"type":"string"},{"type":"integer"}]}},"msg":{"title":"Message","type":"string"},
                    "type":{"title":"Error Type","type":"string"}}}}}}
    app.openapi_schema = openapi_schema
    return app.openapi_schema




app.openapi = custom_openapi
