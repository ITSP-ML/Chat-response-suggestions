import re
import pandas as pd
import numpy as np
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
    text = re.sub('http://\S+|https://\S+', '', text) # replace any url with 'URL'
    text = re.sub('\[[^)]*\]', '', text) # remove words  between []
    text = re.sub('[\t\n\r\f\v].', '', text) # same as ermoving  [ \t\n\r\f\v]
    text = remove_emojis(text)
    # text = re.sub('[^a-zA-Z0-9 ]+', '', text) # keep only letters and numbers and space
    text = re.sub(' +', ' ', text) # remove extra white spaces
    return text

data = pd.read_csv('data_dump/response_sugg/GODEL/data_v1.csv')
def get_data (chat_id, last_id):
    row_chat = data[data.chat_id == chat_id].iloc[:last_id]

 

    return row_chat