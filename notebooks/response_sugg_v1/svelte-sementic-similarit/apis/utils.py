import time
import re
from sentence_transformers import util
# def match_suggs(prefix, dataset, model, annoy_index, top_k_hits = 10 ):
    
#     start_time = time.time()
#     question_embedding = model.encode(prefix)
#     corpus_ids, scores = annoy_index.get_nns_by_vector(question_embedding, top_k_hits, include_distances=True)
#     hits = []
#     for id, score in zip(corpus_ids, scores):
#             hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})
#     end_time = time.time()
#     print("Results (after {:.3f} seconds):".format(end_time-start_time))
#     result = dataset[dataset.index.isin([x['corpus_id'] for x in hits])]
#     result['score'] = [x['score'] for x in hits]
#     soreted_results = result.sort_values('score')
#     return soreted_results
def match_suggs(prefix, dataset, model, encoded_data, top_k_hits = 10 ):
    question_embedding = model.encode(prefix)
    correct_hits = util.semantic_search(question_embedding, encoded_data, top_k=top_k_hits)[0]
    # cleand_responses_df[cleand_responses_df.index.isin([x['corpus_id'] for x in correct_hits])]
    result = dataset[dataset.index.isin([x['corpus_id'] for x in correct_hits])]
    result['score'] = [x['score'] for x in correct_hits]
    return result
def rank_suggs(suggs):
    # sort by frequency
    # suggs = suggs.sort_values('freq', ascending = False)
    return suggs['responses'].to_list()

def filter_unique(dataframe,  how = 'before', preprocessing = True):

    return dataframe['responses'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x)).unique()
