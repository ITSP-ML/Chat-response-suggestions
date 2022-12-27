def match_suggs(prefix, dataset, column = 'compare'):
    lower_prefix = prefix.lower()
    filtered_suggestions = dataset[dataset[column].str.contains(prefix)]
    # rank them by the suggestions
    return filtered_suggestions[['msg', 'freq']]


def rank_suggs(suggs):
    # sort by frequency
    suggs = suggs.sort_values('freq', ascending = False)
    return suggs.msg.iloc[:10].to_list()