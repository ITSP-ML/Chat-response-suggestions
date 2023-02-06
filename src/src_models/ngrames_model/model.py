from src.back_end.trie import Trie



def get_words_match(prefix, t):
    res = t.search(prefix)
    match_prefix = [{"sugg": msg} for msg, freq in res if msg!=prefix]
    return match_prefix


def build_trie(dataset, validation_threshold, max_number_of_suggestions = 10):
    queries = {}
    for i, row in dataset.iterrows():
        msg, freq, ngram_level = row
        queries[msg] = [freq, ngram_level]

    # build trie
    t = Trie(validation_threshold, max_number_of_suggestions = max_number_of_suggestions)
    t.build_tree(queries)
    return t

