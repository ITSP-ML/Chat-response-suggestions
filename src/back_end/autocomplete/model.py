from src.back_end.autocomplete.trie import Trie



def get_words_match(prefix, t, nb_suggs):
    final_results = {}
    res = t.search(prefix, nb_suggs)
    # remove the prefix if its in suggestions
    if prefix in list(res.keys()):
        del res[prefix]
    final_results['suggestions'] = list(res.keys())
    final_results['scores'] = list(res.values())
    return final_results


def build_trie(dataset, validation_threshold, max_number_of_words = 3):
    queries = {}
    for i, row in dataset.iterrows():
        msg, freq, ngram_level = row
        queries[msg] = [freq, ngram_level]

    # build trie
    t = Trie(validation_threshold, max_number_of_words)
    t.build_tree(queries)
    return t

