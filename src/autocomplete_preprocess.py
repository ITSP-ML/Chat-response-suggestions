import pandas as pd

import set_cwd

# turn off chained_assignment warnings
pd.options.mode.chained_assignment = None  # default='warn'

def get_agent_msgs():
    data = pd.read_csv("data_dump/response_sugg/data_v1.csv")
    data = data.dropna()
    data.msg = data.msg.str.replace('\n', ' ')
    data.msg = data.msg.str.replace('\r', '')

    agent_msgs = data[data.user_id > 0].msg.value_counts().reset_index()
    agent_msgs.columns = ['msg', 'count']
    return agent_msgs


def get_probable_continuations(prefix, candidate_msgs, prob_threshold=0.02, conditional_prob=1,
                               max_n=9999):
    # filter out messages that start with prefix
    continuations = candidate_msgs[candidate_msgs.msg.str.lower().str.startswith(prefix.lower())]
    # take only the continuations after prefix
    continuations.msg = continuations.msg.str[len(prefix):]
    split_initial_non_word_chars = continuations.msg.str.extract(r'^(\W*)(.*)')
    continuations['stripped_msg'] = split_initial_non_word_chars[1]
    continuations['non_word_chars'] = split_initial_non_word_chars[0]

    # get the first words of continuations
    continuations['next_word'] = continuations.non_word_chars + continuations.stripped_msg.str.split(r'\W', regex=True).str[0]

    # compute frequencies of next words
    nexts = pd.DataFrame(continuations.groupby('next_word')['count'].sum().sort_values(ascending=False))
    # remove empty nexts (end of message reached)
    nexts = nexts.drop(index='', errors='ignore')

    # if the maximal length of remaining suggestion drops to or below 0,
    # we only continue further if the next word is unique
    # otherwise we stop and return no possible continuations
    if (max_n <= 0) and (len(nexts)>1):
        return pd.DataFrame(columns=['count', 'prob'])

    # compute probabilities of next words
    nexts['prob'] = (nexts['count']/nexts['count'].sum())*conditional_prob

    # remove improbable next words
    nexts = nexts[nexts.prob >= prob_threshold]

    #print('Nexts computed:')
    #print(nexts)

    ngrams = nexts.copy()
    for next, row in nexts.iterrows():
        prob = row.prob
        subs = get_probable_continuations(next, continuations[['msg', 'count']],
                                          prob_threshold=prob_threshold, conditional_prob=prob,
                                          max_n=max_n-1)
        if len(subs)>0:
            subs.index = next + subs.index
            ngrams = ngrams.drop(index=next)
            ngrams = pd.concat([ngrams, subs])

    ngrams = ngrams.sort_values(by='prob', ascending=False)

    return ngrams
