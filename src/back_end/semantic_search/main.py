from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from src.back_end.preprocess.autocomplete_preprocess import get_agent_msgs, preprocess_msg
from src.back_end.semantic_search.model import get_sementic_match







app = FastAPI(root_path= '/semantic_search')

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
    nb_suggs: int = 10
    min_prob: int = 0



# read agent data
dataset_path = 'data/prod_v1/agent_data.csv'
dataset = get_agent_msgs(dataset_path)


# sementic search
model_name = 'distilbert-base-nli-stsb-mean-tokens'
embbeding_path = "data/prod_v1/doc_embedding.pickle"







@app.get("/")
async def home():
    return 'API is working'

@app.post("/search")
async def root(data: Item):
    prefix = data.text
    nb_suggs = data.nb_suggs
    min_prob = data.min_prob
    # preprocess prefix 
    pre_prefix = preprocess_msg(prefix)
    return get_sementic_match(pre_prefix, dataset, model_name, embbeding_path, nb_suggs = nb_suggs, min_prob = min_prob)







def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Semantic Search",
        version="1.0.1",
        description="match the user input by meanings",
        routes=app.routes,
    )
    # change the input of the API
    openapi_schema["components"]["schemas"]["Item"] ={"title":"Item","required":["text"],"type":"object","properties":{"text":{"title":"Text","type":"string", "default":'hi there', "description": "prefix that agent already typed"},
                    "nb_suggs":{"title":"Nb Suggs", "description": "number of returned suggestions","type":"integer","default":10},
                    "min_prob":{"title":"Min Prob","type":"integer", "description": "ensure that all generated suggestions have a minimum probability ","default":0}}}

    # set the version of Openapi
    openapi_schema["openapi"] = "3.0.2"

    app.openapi_schema = openapi_schema
    return app.openapi_schema






app.openapi = custom_openapi



