import pandas as pd
import re
import sys



# turn off chained_assignment warnings
pd.options.mode.chained_assignment = None  # default='warn'


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


def get_agent_msgs(data_path):
    # read the preprocessed agent data
    dataset = pd.read_csv(data_path)
    dataset = dataset[dataset.freq >= 2]
    dataset['len'] = dataset.processed_msg.apply(lambda x: len(x.split()))
    dataset = dataset[dataset.len <= 50]
    dataset = dataset.reset_index(drop = True)
    del dataset['len']
    return dataset

def change_recursion_limit(new_limit):
    # set higher number of recursions
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(new_limit)
    print(sys.getrecursionlimit())

# remove all extentions
def remove_extensions(text):
    '''
    We removed attachments while extracting body but not the name of these attachments
    removing attachment_names based on what i encountered in subject and body
    '''
    ext_patterns = ["\S+\.doc","\S+\.jpeg","\S+\.jpg","\S+\.gif","\S+\.csv","\S+\.ppt","\S+\.dat","\S+\.xml","\S+\.xls","\S+\.sql","\S+\.nsf","\S+\.jar","\S+\.bin",'\S+\.jpg', "\S+\.png"]
    pattern = '|'.join(ext_patterns)
    text = re.sub(pattern,'',text)
    return text


#Decontraction of text
def decontracted(phrase):
    """
    Returns decontracted phrases
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def remove_personal_name(text):
    '''
    Helper function to Filter out names using NER
    '''

    s = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    for ele in s:
        if isinstance(ele, nltk.Tree):
            if ele.label()=='PERSON':
                for word,pos_tag in ele:
                    try:     # words containing a special character will raise an error so handling it, these words weren't a name so we can safely skip it
                        val = re.sub(word,'',text)
                        text = val
                    except:
                        continue
    return text



def smart_compose_processing(text):
        # remove url and email-id's
        remove_url = r'(www|http)\S+'
        remove_email = r'\S+@\S+' 
        remove_space = r'\s+'
        remove_phone =  "^(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"
        pattern_list_1 = [remove_url,remove_phone,remove_email]

        for pattern in pattern_list_1:
            text = re.sub(pattern,'',text)

        # remove attachment_names
        text = remove_extensions(text)

        # remove any word with digit for coupons
        text = re.sub(r'\w*\d\w*', '', text)

        # remove any digit
        text = re.sub('\d','',text)

        # remove text between <>,()
        remove_tags = r'<.*>'
        remove_brackets = '\[.*?\]'
        remove_parentheses = '\(.*?\)'
        # remove_brackets = r'.*'
        remove_special_1 = r'\|-'  # remove raw backslash or '-'
        remove_colon = r'\b[\w]+:' # removes 'something:'

        pattern_list_2 = [remove_tags,remove_brackets,remove_special_1,remove_colon, remove_parentheses]
        for pattern in pattern_list_2:
            text = re.sub(pattern,'',text)
            

        # remove anything which is not a character,apostrophy ; remember to give a space on replacing with this
        remove_nonchars = r'[^A-Za-z\']'
        text = re.sub(remove_nonchars,' ',text)

        # remove AM/PM as we have a lot of timestamps in emails
        # text = remove_timestamps(text)

        # remove personal names using named entity recognition
        # text = remove_personal_name(text) # personal name removal will be done using the words frequency threshold

        # takes care of \t & \n ; remember to give a space on replacing with this
        remove_space = r'\s+'
        text = re.sub(remove_space,' ',text)

        # take care of apostrophies
        text = decontracted(text)

        # remove other junk
        text = text.replace("IMAGE",'')
        text = re.sub(r"\bth\b",'',text)

        return text.strip()   