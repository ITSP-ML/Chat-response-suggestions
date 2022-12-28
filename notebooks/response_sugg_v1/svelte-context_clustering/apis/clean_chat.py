import pandas as pd
import re
import numpy as np
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
    text = re.sub('[\t\n\r\f\v].', '', text) # same as ermoving  [ \t\n\r\f\v]
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

def get_godel_format(df, chat_id, max_context_length = 200):
    output = []
    for i, x in enumerate(df.iterrows()):
        if x[1].user_id != 0:
                json = {}
                context = clean_text(' EOS '.join(df.iloc[:i].msg.to_list()))
                iter = 0
                while len(context.split()) > max_context_length:
                    context = clean_text(' EOS '.join(df.iloc[iter:i].msg.to_list()))
                    iter = iter +1
                json['Context'] = context
                json['Knowledge'] = clean_text(df.iloc[i].canned_msg if df.iloc[i].isna().canned_msg == False else '')
                json['Response'] = clean_text(df.iloc[i].msg)
                json['score'] = df.iloc[i].score if df.iloc[i].isna().score == False else 0
                json['chat_id'] = chat_id
                output.append(json)
    if output == []:
                json = {}
                context = clean_text(' EOS '.join(df.msg.to_list()))
                iter = 0
                while len(context.split()) > max_context_length:
                    context = clean_text(' EOS '.join(df.iloc[iter:].msg.to_list()))
                    iter = iter +1
                json['Context'] = context
                json['Knowledge'] = ''
                json['Response'] = ''
                json['score'] = 0
                json['chat_id'] = chat_id
                output.append(json)
    return output

def clean(row):
    keep = True
    # remove long response or context
    if len(row['Context'].split())> 200 or len(row['Response'].split())>200:
        keep = False
    # remove file only response
    files = ['URL' , 'File_Sent']
    for file in files:
        if file in row['Context'] or file in row['Response'] :
            keep = False
    # remove empty response or context
    if row["Context"] == '' or row['Response'] == '':
        keep = False
    return keep


data = pd.read_csv('data_dump/response_sugg/GODEL/data_v1.csv')
def get_data (chat_id, last_id):
    row_chat = data[data.chat_id == chat_id].iloc[:last_id]

    pre_processed_chat = clean_chat(row_chat)
    pre_processed_chat = get_godel_format(pre_processed_chat, chat_id)[-1]
    pre_processed_chat['score'] = int(np.round(pre_processed_chat['score'], 2) *100)
    pre_processed_chat['chat_id'] = int(pre_processed_chat['chat_id'])
    # a = [json.loads(pre_processed_chat)]
    # pre_processed_chat = pd.DataFrame(pre_processed_chat, index=[0])
    keep = clean(pre_processed_chat)
    print(pre_processed_chat)
    return row_chat, pre_processed_chat