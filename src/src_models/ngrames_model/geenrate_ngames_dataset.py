import pandas as pd
from src.preprocess.autocomplete_preprocess import get_agent_msgs
def get_ngrames(sentance):
    """
    function that get all ngrames from a sentance
    """
    final_output =[]
    words = sentance.split()
    tmp = ''
    for word in words:
        tmp = tmp + word + ' '
        final_output.append(tmp[:-1])
    return (final_output)

def generate_ngrams_dataset(old_dataset, path = None):
    """
    get all ngrames from sentances
    """
    output = {}
    for i, row in old_dataset.iterrows():
        msg, freq, l= row
        ngrames = get_ngrames(msg)
        for x in ngrames:
            if x not in list(output.keys()):
                output[x] = freq
            else:
                output[x] += freq
    final_dict = {"processed_msg": list(output.keys()), 'freq':  list(output.values())}
    final_data = pd.DataFrame(final_dict)

    # save this dataset
    if path is not None:
        final_data.to_csv(path, index= False)


    return final_data



if __name__ == "__main__":

    old_data_path = 'data/prod_v1/agent_data.csv'
    save_path = "data/prod_v1/agent_data_2.csv"
    old_data = get_agent_msgs(old_data_path)
    generate_ngrams_dataset(old_data, save_path)