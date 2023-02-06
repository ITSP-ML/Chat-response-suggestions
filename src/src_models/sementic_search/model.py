import pickle
from sentence_transformers import util, SentenceTransformer



def get_sementic_match(prefix, dataset, model_name, embbeding_path, top_k_hits = 10,min_prob = 0,  max_number_of_words = 3):
    # load encoded data
    with open(embbeding_path, 'rb') as pkl:
        encoded_data = pickle.load(pkl)
    # load model
    model = SentenceTransformer(model_name)
    question_embedding = model.encode(prefix)
    correct_hits = util.semantic_search(question_embedding, encoded_data, top_k=top_k_hits)[0]
    result = dataset[dataset.index.isin([x['corpus_id'] for x in correct_hits])]
    result['score'] = [x['score'] for x in correct_hits]
    result['valid_length'] = result.processed_msg.apply(lambda x: True if len(x.split()) <=len(prefix.split()) + max_number_of_words else False)
    # result = result[(result.valid_length == True) & (result.score >= min_prob)]
    return  [{"sugg": msg} for msg in result.processed_msg.to_list() if msg!=prefix]