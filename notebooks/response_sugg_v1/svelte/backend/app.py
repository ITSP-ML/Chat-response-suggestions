from fastapi import FastAPI
from flask import jsonify
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import set_cwd
import re

# data = pd.read_csv("data_dump/response_sugg/data_v1.csv")
data = pd.read_csv(
    "C:/Users/feress/Documents/myprojects/Chat-response-suggestions/data_dump/response_sugg/data_v1.csv"
)
data = data.dropna()
agent_msgs = data[data.user_id > 0]

# agent_unique = agent_msgs.value_counts().reset_index()
agent_unique = agent_msgs.msg.unique()

agent_msgs = []
for msg in agent_unique:
    splited = re.split(". |, |;|\n|\? |\! ", msg)
    agent_msgs.extend(splited)

# languages = [
#     "C++",
#     "Python",
#     "PHP",
#     "Java",
#     "C",
#     "Ruby",
#     "R",
#     "C#",
#     "Dart",
#     "Fortran",
#     "Pascal",
#     "Javascript",
# ]


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


# @app.post("/post")
# async def root(data: Item):
#     # print(data.text)
#     # filtered_dict = [v for v in languages if data.text in v]
#     # result = [{"sugg": x} for x in filtered_dict]

#     return data


@app.post("/")
async def root(data: Item):
    print(data.text)
    prefix = data.text.lower().split()

    filtered_dict = [
        v for v in agent_msgs if data.text.lower() == v.lower()[: len(data.text)]
    ]
    # test = True
    # for x in prefix:
    #     if x in v
    # filtered_dict = [
    #     v for v in agent_unique if all([item in v.lower() for item in prefix])
    # ]
    # filtered_dict = [" ".join(x.split()[:6]) for x in filtered_dict]
    result = [{"sugg": x} for x in filtered_dict]

    return result


# @app.get("/")
# async def root():
#     # print(data.text)
#     filtered_dict = [v for v in languages if "p" in v]
#     result = [{"sugg": x} for x in filtered_dict]
#     print("end succ")
#     return result
