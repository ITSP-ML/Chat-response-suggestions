import pandas as pd
from src.preprocess.autocomplete_preprocess import get_agent_msgs

def get_ngrames(sentance):
    """
    function that get all ngrames and their status from a senatance
    level of an ngrame will be use it to filter the Trie
    """
    ngram_level_dict ={}
    words = sentance.split()
    tmp = ''
    for i, word in enumerate(words):
        tmp = tmp + word + ' '
        ngram_level_dict[tmp[:-1]] = i # this will indicate the level of ngrame (the level increase with the number of words) | also the same as the number of words 

    # ngram_level_dict[sentance] = -1 # indicate this is the original sentance
    return ngram_level_dict

def generate_ngrams_dataset(old_dataset, path= None):
    """
    get all ngrames from sentances
    """
    output = {}
    final_ngrame_dict = {}
    for i, row in old_dataset.iterrows():
        msg, freq, l= row
        ngrames = get_ngrames(msg)
        for msg, level in ngrames.items():
            final_ngrame_dict[msg] = level # add or update the level of the ngram

            if msg not in list(output.keys()):
                output[msg] = freq
            else:
                output[msg] += freq
    final_result = {"processed_msg": list(output.keys()), 'freq':  list(output.values()), "ngram_level": list(final_ngrame_dict.values())}
    final_data = pd.DataFrame(final_result)

    # save this dataset
    if path is not None:
        final_data.to_csv(path, index= False)

    return final_data



if __name__ == "__main__":

    old_data_path = 'data/prod_v1/agent_data.csv'
    save_path = "data/prod_v1/agent_data_3.csv"
    old_data = get_agent_msgs(old_data_path)
    generate_ngrams_dataset(old_data, save_path)