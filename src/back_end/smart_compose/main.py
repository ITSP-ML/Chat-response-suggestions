import tensorflow as tf
import numpy as np 
import nltk
import warnings
import pickle
from nltk import word_tokenize
import tensorflow as tf
import os
from tensorflow.python.keras import backend as K
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from keras_preprocessing.sequence import pad_sequences

from src.back_end.smart_compose.model import AttentionLayer
from src.back_end.preprocess.autocomplete_preprocess import smart_compose_processing




nltk.download('punkt')
warnings.filterwarnings("ignore")

# set model path 
my_decoder_model =  "src/back_end/smart_compose/model_weights/my_last_decoder_model.h5"
my_encoder_model =  "src/back_end/smart_compose/model_weights/my_last_model.h5"
my_data =  "data/smart_compose/train.pickle"



# load tokenizer
with open(my_data, 'rb') as f:
    _, _, _,_,x_tokenizer,y_tokenizer,max_inp_len,max_out_len = pickle.load(f)

# load the encoder and decoder
encoder_model = tf.keras.models.load_model(my_encoder_model, custom_objects={'AttentionLayer': AttentionLayer}, compile=False)
decoder_model = tf.keras.models.load_model(my_decoder_model, custom_objects={'AttentionLayer': AttentionLayer}, compile=False)


reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index
input_word_index= x_tokenizer.word_index


def bring_my_sentence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sos']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='eos'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eos'  or len(decoded_sentence.split()) >= (max_out_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c
    return decoded_sentence


#converting tensor vector to sentence
def seq2output(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sos']) and i!=target_word_index['eos']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString
#converting input tensor to sentence
def seq2input(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=input_word_index['sos']) and i!=input_word_index['eos']):
            newString=newString+reverse_source_word_index[i]+' '
    return newString




app = FastAPI(root_path="/smart_compose")

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



def predict(input):
        input_seq = x_tokenizer.texts_to_sequences([input])
        x = pad_sequences(input_seq, maxlen=max_inp_len, padding='post')
        prediction = bring_my_sentence(x.reshape(1,max_inp_len))
        return {"suggestions": [prediction], "scores": [1]}



@app.get("/")
async def home():
    return 'API is working'



# final autocomplete
@app.post("/search")
async def root(data: Item):
    prefix = data.text
    # preprocess prefix 
    pre_prefix = smart_compose_processing(prefix) # clean and preprocess the new user prefix
    words_match = predict(pre_prefix) # get matched sentences from the Trie
    return words_match




def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="smart_compose",
        version="1.0.2",
        description="smart_compose feature with Auto correction of user inputs",
        routes=app.routes,
    )
    # # change the input of the API
    # openapi_schema["components"]["schemas"]["Item"] = {"title":"Item","required":["text"],"type":"object","properties":{"text":{"title":"Text","type":"string", "default":'hi there', "description": "prefix that agent already typed"},
    #                 "nb_sugg":{"title":"Nb Sugg","type":"integer", "description": "number of returned suggestions","default":10}}}
    # make the docs visible to the reverse proxy
    openapi_schema["servers"] = [ { "url": "/smart_compose" } ]
    # set the version of Openapi
    openapi_schema["openapi"] = "3.0.2"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

