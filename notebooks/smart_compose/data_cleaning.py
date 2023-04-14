# %%
# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
%autoreload 2

# %%
# read data
import pandas as pd 
row_data = pd.read_csv('data/smart_compose/smart_compose_row.csv')
row_data.head(50)

# %%
# get a sample of conversation  
row_data[row_data.chat_id == 18677971]

# %%
row_data.chat_id.unique()[:10]

# %%
"""
# no previous messages
"""

# %%
# keep only agent chat 
agent_data = row_data[row_data.user_id >0]
agent_data.head()

# %%
"""
### data Cleaning
"""

# %%
# remove null messages
agent_data = agent_data[agent_data.msg.isna() != True]

# %%
# sample 30 random agent_messages
agent_data.sample(30)

# %%
# remove all special messages that are between []
def remove_brackets_messages(msg):
    msg = re.sub('\[.*?\]', '', msg) # remove words  between [] as they are files
    # msg = re.sub('\[[^)]*\]', '', msg) # remove words  between [] as they are files
    return msg

# test it
sample_messages = agent_data[agent_data.msg.str.contains('\[')].sample(10, random_state= 0).msg
for msg in sample_messages:
    print('message before: ', msg)
    print('message after: ', remove_brackets_messages(msg))

# %%
# apply for all data
agent_data.msg = agent_data.msg.apply(remove_brackets_messages)

# %%
import re
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
# test it
sample_messages = agent_data[agent_data.msg.str.contains('.jpg')].sample(10, random_state= 0).msg
for msg in sample_messages:
    # print('message before: ', msg)
    print('message after: ', remove_extensions(msg))

# %%
# apply for all data
agent_data.msg = agent_data.msg.apply(remove_extensions)

# %%

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

# %%

def remove_timestamps(text):
    '''
    Remove all types of 'text' data from timestamps
    '''
    text = text.replace('AM','')
    text = text.replace('PM','')
    text = text.replace('A.M.','')
    text = text.replace('P.M.','')
    text = text.replace('a.m.','')
    text = text.replace('p.m.','')
    text = re.sub(r"\bam\b",'',text)
    text = re.sub(r"\bpm\b",'',text)
    return text

# %%
import nltk
nltk.download("popular")
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


# %%
remove_personal_name('Hello Rafael')

# %%
def transform_msg(text):
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

# %%
# test it on asmple of messages
sample_messages = agent_data.sample(10, random_state= 1).msg
for msg in sample_messages:
    print('message before: ', msg)
    print('message after: ', transform_msg(msg))

# %%
# apply for all data
agent_data.msg = agent_data.msg.apply(transform_msg)

# %%
# lets check distibution of words
import matplotlib.pyplot as plt
import seaborn as sns
agent_data['msg_wct'] = [len(x.split()) for x in agent_data['msg'].tolist()] 
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)
sns.distplot(agent_data['msg_wct'], ax=ax)  
plt.title("Distribution of words in email-body",fontsize=18)
plt.show()


# %%
"""
**most of agent messages are less than 50 words**
"""

# %%
import numpy as np
##print 90 to 100 percentile values with step size of 1. 
for i in range(90,101,1):
    print("{}th percentile is".format(i),np.percentile(agent_data.msg_wct,i))

# %%


temp = agent_data.msg.str.len()
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)
sns.distplot(temp, ax=ax)  
plt.title("Distribution of chars in email-body",fontsize=18)
plt.xlabel("#chars in body")
plt.show()

# %%

##print 90 to 100 percentile values with step size of 1. 
for i in range(90,101,1):
    print("{}th percentile is".format(i),np.percentile(temp,i))

# %%
"""
**most msg have msg length less than 300
"""

# %%

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import wordcloud
body = ' '.join(agent_data.sample(5000)['msg'])
fig, ax = plt.subplots(figsize=(16, 12))
wc = wordcloud.WordCloud(width=800, 
                         height=600, 
                         max_words=200,
                         stopwords=ENGLISH_STOP_WORDS).generate(body)
ax.imshow(wc)
ax.axis("off")
plt.show()

# %%
# duplicates
agent_data.drop_duplicates(subset=['msg'], keep='first', inplace=True, ignore_index=True)

# nulls
agent_data = agent_data.dropna(axis=0,subset=['msg'])

# %%
# save the data
agent_data.to_csv('data/smart_compose/agent_data_cleaned.csv', index= False)

# %%
