from fastapi import FastAPI
from flask import jsonify
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import set_cwd
import re



# data = pd.read_csv("data_dump/response_sugg/data_v1.csv")




# agent_msgs = []
# for msg in agent_unique:
#     splited = re.split(". |, |;|\n|\? |\! ", msg)
#     agent_msgs.extend(splited)


app = FastAPI()

origins = ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    text: str


@app.post("/sugg")
async def root(data: Item):
    print(data.text)
    # print prefix
    prefix = str(data.text).lower()
    print("prefix: ", prefix)  # PREFIX
    # match prefix to suggestions
    unsorted_suggs = match_suggs(prefix, dataset)
    final_suggs = rank_suggs(unsorted_suggs)

    return [{"sugg": x} for x in final_suggs]  # desired output shape for the frontend
