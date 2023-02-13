import pickle
from sentence_transformers import util, SentenceTransformer



def get_sementic_match(prefix, dataset, model_name, embbeding_path, nb_suggs = 10, min_prob = 0):
    # load encoded data
    with open(embbeding_path, 'rb') as pkl:
        encoded_data = pickle.load(pkl)
    # load model
    model = SentenceTransformer(model_name)
    prefix_embedding = model.encode(prefix)
    # get the scores
    correct_hits = util.semantic_search(prefix_embedding, encoded_data, top_k= len(dataset))[0]
    # get corpus ids
    filtred_messages = {}
    for x in correct_hits:
        if x['score'] >= min_prob :
            if len(filtred_messages) < nb_suggs:
                filtred_messages[x['corpus_id']] = x['score']
            else:
                break
    # get messages corresponding to those index
    result = dataset[dataset.index.isin(list(filtred_messages.keys()))]
    result['score'] = list(filtred_messages.values())
    result = result[['processed_msg', 'score']]
    # rename columns
    result.columns = ['suggestions', 'probabilities']
    return  result.iloc[:nb_suggs].to_dict('list')