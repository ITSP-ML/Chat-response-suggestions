from src.back_end.trie import Trie

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


def build_trie(dataset):
    # create query dict
    queries = create_queries(dataset)

    # build trie
    t = Trie()
    t.build_tree(queries)
    return t

