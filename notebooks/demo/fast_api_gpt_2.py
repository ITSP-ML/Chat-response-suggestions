from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2")
set_seed(42)


def get_suggs(data):
    return generator(
        data,
        max_length=30,
        num_return_sequences=3,
        # do_sample=True,
        # min_length=5
        # top_k_top_p_filtering=0.9,
        # TemperatureLogitsWarper=1,
    )


app = FastAPI()


class Item(BaseModel):
    text: str


@app.post("/")
async def root(data: Item):
    print(data.text)
    return get_suggs(data.text)
