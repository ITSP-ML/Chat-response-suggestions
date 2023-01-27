""" this module will contains all the preprocessing function used in generating the suggestion dataset or processing the user prefix"""
import re




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

# remove spaces at the start and the end of each message
def remove_space(msg):
    start, end = 0, len(msg)-1
    if msg[0] == " ":
        start = start +1
    elif msg[-1] == " ":
        end = end-1
    return msg[start: end +1]

def preprocess_msg(msg):
    msg = re.sub('http://\S+|https://\S+', '', msg) # replace any url with 'URL'
    msg = re.sub('\[[^)]*\]', '', msg) # remove words  between [] as they are files
    msg = msg.replace("\n", '')
    msg = msg.replace("\t", '')
    msg = msg.replace("\r", '')
    msg = msg.replace("\f", '')
    msg = msg.replace("\v", '')
    msg = remove_emojis(msg)
    msg = re.sub(r'[^\w\s]', '', msg) # remove pounctuation
    # text = re.sub('[^a-zA-Z0-9 ]+', '', text) # keep only letters and numbers and space
    msg = re.sub(' +', ' ', msg) # remove extra white spaces
    msg = msg.lower() # get every thing to lowercase
    if len(msg.split())>=1:
        msg = remove_space(msg) # remove space at the start and at the end
    return msg

# # apply
# agent_data['processed_msg'] = agent_data.msg.apply(preprocess_msg)
# agent_data