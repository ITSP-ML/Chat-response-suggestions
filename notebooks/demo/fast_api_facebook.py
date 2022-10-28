from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Conversation,
    ConversationalPipeline,
)
from fastapi import FastAPI
from pydantic import BaseModel
from flask import jsonify

from transformers import pipeline

pipe = pipeline(model="facebook/blenderbot-1B-distill")
conversation_1 = Conversation("customer support")


app = FastAPI()


class Item(BaseModel):
    text: str


@app.post("/")
async def root(data: Item):
    print(data.text)
    conversation_1.add_user_input(data.text)
    return pipe(conversation_1)


@app.post("/default")
async def root(data: Item):
    print(data.text)
    return pipe(conversation_1)
