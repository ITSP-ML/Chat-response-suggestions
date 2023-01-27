import pandas as pd
import re
import sys
import pickle
from sentence_transformers import util, SentenceTransformer

import set_cwd

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
    return dataset

def change_recursion_limit(new_limit):
    # set higher number of recursions
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(new_limit)
    print(sys.getrecursionlimit())

def create_queries(dataset):
    queries = {}
    for i, row in dataset.iterrows():
        msg, freq, l = row
        queries[msg] = freq
    return queries

def get_words_match(prefix, t):
    
    res = t.search(prefix)
    match_prefix = [{"sugg": msg} for msg, freq in res if msg!=prefix]
    return match_prefix

def get_sementic_match(prefix, dataset, model_name, embbeding_path, top_k_hits = 10 ):
    # load encoded data
    with open(embbeding_path, 'rb') as pkl:
        encoded_data = pickle.load(pkl)
    # load model
    model = SentenceTransformer(model_name)
    question_embedding = model.encode(prefix)
    correct_hits = util.semantic_search(question_embedding, encoded_data, top_k=top_k_hits)[0]
    result = dataset[dataset.index.isin([x['corpus_id'] for x in correct_hits])]
    result['score'] = [x['score'] for x in correct_hits]
    return  [{"sugg": msg} for msg in result.processed_msg.to_list() if msg!=prefix]


def get_probable_continuations(prefix, candidate_msgs, prob_threshold=0.02, conditional_prob=1,
                               max_n=9999):
    # filter out messages that start with prefix
    pre_prefix = preprocess_msg(prefix)
    continuations = candidate_msgs[candidate_msgs.processed_msg.str.startswith(pre_prefix)]
    # take only the continuations after prefix
    continuations.processed_msg = continuations.processed_msg.str[len(prefix):]
    split_initial_non_word_chars = continuations.processed_msg.str.extract(r'^(\W*)(.*)')
    continuations['stripped_msg'] = split_initial_non_word_chars[1]
    continuations['non_word_chars'] = split_initial_non_word_chars[0]

    # get the first words of continuations
    continuations['next_word'] = continuations.non_word_chars + continuations.stripped_msg.str.split(r'\W', regex=True).str[0]

    # compute frequencies of next words
    nexts = pd.DataFrame(continuations.groupby('next_word')['freq'].sum().sort_values(ascending=False))
    # remove empty nexts (end of message reached)
    nexts = nexts.drop(index='', errors='ignore')

    # if the maximal length of remaining suggestion drops to or below 0,
    # we only continue further if the next word is unique
    # otherwise we stop and return no possible continuations
    if (max_n <= 0) and (len(nexts)>1):
        return pd.DataFrame(columns=['freq', 'prob'])

    # compute probabilities of next words
    nexts['prob'] = (nexts['freq']/nexts['freq'].sum())*conditional_prob

    # remove improbable next words
    nexts = nexts[nexts.prob >= prob_threshold]

    #print('Nexts computed:')
    #print(nexts)

    ngrams = nexts.copy()
    for next, row in nexts.iterrows():
        prob = row.prob
        subs = get_probable_continuations(next, continuations[['processed_msg', 'freq']],
                                          prob_threshold=prob_threshold, conditional_prob=prob,
                                          max_n=max_n-1)
        if len(subs)>0:
            subs.index = next + subs.index
            ngrams = ngrams.drop(index=next)
            ngrams = pd.concat([ngrams, subs])

    ngrams = ngrams.sort_values(by='prob', ascending=False)

    return ngrams
