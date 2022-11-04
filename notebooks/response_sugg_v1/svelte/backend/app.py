from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.autocomplete_preprocess import get_agent_msgs, get_probable_continuations

agent_msgs = get_agent_msgs()

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


@app.post("/")
async def root(data: Item):
    prefix = data.text
    suggestions = get_probable_continuations(prefix, agent_msgs, prob_threshold=0.02, max_n=10)
    return [{"prefix": prefix, "sugg": sugg, "prob": row["prob"]} for sugg,
                                                                 row in suggestions.iterrows()]