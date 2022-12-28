# Clustering autocomplete
- Define a dataset of context (agent + customer till the last customer messages)
- run a crusting algorithm (BERTtopic) on them and fine tune the results
- map each context to the list of corresponding clusters and save this on the dataset
- given a new context try to predict the clusters and based on this cluster filter the responses that have the same cluster 
- Either show this response as a list of suggestion or use the autocomplete feature for this 