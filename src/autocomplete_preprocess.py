import pandas as pd

import set_cwd

def get_agent_msgs():
    data = pd.read_csv("data_dump/response_sugg/data_v1.csv")
    data = data.dropna()

    agent_msgs = data[data.user_id > 0].msg.value_counts().reset_index()
    agent_msgs.columns = ['msg', 'count']
    return agent_msgs


def get_probable_continuations(prefix, candidate_msgs, prob_threshold=0.02):
    # filter out messages that start with prefix
    continuations = candidate_msgs[candidate_msgs.msg.str.lower().str.startswith(prefix.lower())]
    # take only the continuations after prefix
    continuations.msg = continuations.msg.str[len(prefix):]
    split_initial_non_word_chars = continuations.msg.str.extract(r'^(\W*)(.*)')
    continuations.msg = split_initial_non_word_chars[1]
    continuations['non_word_chars'] = split_initial_non_word_chars[0]

    # get the first words of continuations
    continuations['next_word'] = continuations.non_word_chars + continuations.msg.str.split(r'\W',
                                                                                            regex=True).str[0]

    # compute frequencies/probabilities of next words
    nexts = pd.DataFrame(continuations.groupby('next_word')['count'].sum().sort_values(ascending=False))
    nexts['prob'] = nexts['count']/nexts['count'].sum()
    # remove improbable next words
    nexts = nexts[nexts.prob >= prob_threshold]

    #continuations.drop(columns='next_word')

    return nexts
