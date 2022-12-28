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
from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
# load fine tuned dialogpt model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('output-small')

print('****************************************************************')
import re

# remove new lines and tabs and emojis and extra white spaces
def remove_emojis(data):
    emoj = re.compile("["
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def clean_text(text):
    text = re.sub('http://\S+|https://\S+', 'URL', text) # replace any url with 'URL'
    text = re.sub('\[[^)]*\]', 'File_Sent', text) # remove words  between []
    text =  ' '.join(text.splitlines()) # remove all line breakers
    text = remove_emojis(text)
    # text = re.sub('[^a-zA-Z0-9 ]+', '', text) # keep only letters and numbers and space
    text = re.sub(' +', ' ', text) # remove extra white spaces
    return text
def clean_chat(row_chat):
    clean_chat = pd.DataFrame(row_chat.iloc[0].to_dict(), index = [0])
    last_user_id = 0
    clean_chat
    for i, row in enumerate(row_chat.iloc[1:].iterrows()):
        if row[1].user_id != last_user_id:
            mode = 'add'
            clean_chat.loc[i+1] = row[1]
            # get the output to GODEL foramt
        else:
            mode = 'concat'
            clean_chat.loc[clean_chat.index[-1],'msg'] = clean_chat.iloc[-1].msg +  ' ' + row[1].msg
            if row[1].isna().canned_msg ==False:

                    if clean_chat.iloc[-1].isna().canned_msg == False:
                        clean_chat.loc[clean_chat.index[-1],'canned_msg'] = clean_chat.iloc[-1].canned_msg +  ' ' + row[1].canned_msg
                        clean_chat.loc[clean_chat.index[-1],'score'] = np.max([clean_chat.iloc[-1].score, row[1].score])
                        clean_chat.loc[clean_chat.index[-1],'title'] = clean_chat.iloc[-1].title +  ' ' + row[1].title
                    else:
                        clean_chat.loc[clean_chat.index[-1],'canned_msg'] = row[1].canned_msg
                        clean_chat.loc[clean_chat.index[-1],'score'] = row[1].score
                        clean_chat.loc[clean_chat.index[-1],'title'] = row[1].title
        last_user_id = row[1].user_id
    return clean_chat

import torch
import pandas as pd
import numpy as np
import torch
def preprocess_chat(chat):
    contexts = []
    responses = []
    conc_chat = clean_chat(chat)
    conc_chat.msg = conc_chat.msg.apply(lambda x : clean_text(x))
    for i in range (2, len(conc_chat) +2, 2):
            sub_chat = conc_chat.iloc[i-2: i]
            if len(sub_chat)>1 and len(sub_chat.user_id.unique()) == 2:
                assert sub_chat.user_id.values[0] == 0, f"this pairs dose't starts with customer debug with chat_id"
            try:
                context, response = sub_chat.msg.values
                contexts.append(context)
                responses.append(response)
            except:
                continue

    data_dict = {}
    data_dict['response'] = responses
    data_dict['context'] = contexts
    final_dataframe = pd.DataFrame(data = data_dict)
    final_dataframe['words_context'] = final_dataframe.context.apply(lambda x: [word.lower() for word in x.split()])
    final_dataframe['words_response'] = final_dataframe.response.apply(lambda x: [word.lower() for word in x.split()])

    return final_dataframe[['response' , 'context']]





# get_responses(similar_topics)
def transform_chat_to_json(df, suggs_list):
    all_chat_msgs = []
    for index, row  in enumerate(df.iterrows()):
        message_info = {}
        message_info["messageIndex"] =  index +1
        message_info["messageId"] =  row[1].msg_id
        message_info["message"] = row[1].msg
        message_info["sentByMe"] = True if row[1].user_id == 0 else False
        all_chat_msgs.append(message_info)
    response = {'messages':all_chat_msgs , 'dialogpt_suggs': suggs_list}
    return response
def predict(df):

    pre = preprocess_chat(df)
    print('lenght of preprocessed chat' , len(pre))
    print(pre)
    bot_suggs = []
    for step, row in enumerate(pre.iterrows()):
        context = row[1].context
        new_user_input_ids = tokenizer.encode(context+ tokenizer.eos_token, return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = new_user_input_ids

        # generated a bot_response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id
        )
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        bot_suggs.append(bot_response)
    return bot_suggs
    return  [{"id": i, 'text': x}for i, x in enumerate(bot_suggs)]

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
    df = get_data(example_id, last_msg_id)
    print("*********************")
    suggs_list = predict(df)#, knowledge)
    # suggs_list = []
    json = transform_chat_to_json(df, suggs_list)
    return json




# @app.post("/sugg")
# async def root(data: Item2):
#     print(data.text)
#     # print prefix
#     prefix = str(data.text)
#     print("prefix: ", prefix)  # PREFIX
#     # match prefix to suggestions
#     print(dataset.head())
#     unsorted_suggs = match_suggs(prefix, dataset)
#     final_suggs = rank_suggs(unsorted_suggs)

#     return [{"sugg": x} for x in final_suggs]  # desired output shape for the frontend