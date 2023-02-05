class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char, isleaf = False, parent_word_count = -1):
        # the character stored in this node
        self.char = char
        self.parent_word_count = parent_word_count
        # a flag to mark the end of the query and store its frequency.
        self.count = -1
        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}
        self.ngram_level = -1 # this will indicate the ngram level
        self.isleaf = isleaf
        self.is_valid = False

class Trie(object):
    def __init__(self, validation_threshold):
        """
        Initiate the trie with an empty char
        """
        self.root = TrieNode("")
        self.validation_threshold = validation_threshold

    def build_tree(self, queries_dict):
        """
        :param queries_dict (dict): format {query string : frequency}
        """
        for query in queries_dict.keys():
            self.insert(query, queries_dict)

    def insert(self, query, queries_dict):
        """Insert a query into the trie"""
        node = self.root
        # Loop through each character in the query string
        # if there is no child containing the character, create a new child for the current node
        for char in query:
                if char in node.children:
                    node.isleaf = False
                    node = node.children[char]
                else:
                    # If a character is not found, create a new node in the trie
                    new_node = TrieNode(char)
                    node.children[char] = new_node
                    node = new_node
        # Mark the end of a query with isleaf seted to True
        node.isleaf = True
        node.count = queries_dict[query][0]  # get freq
        node.is_ngrame = queries_dict[query][1]  # this will indicate if the query is an ngrame or not

    def dfs(self, node, prefix, cache = None, prefix_cache = None):
        """Depth-first traversal of the trie
        :param node (TrieNode): the root of the subtree
        :param prefix (string): the current prefix to form the full queries when traversing the trie
        """
        if node.count != -1:
          
            if cache is None:
       
                cache = node # first cache
                prefix_cache = prefix
                # print(prefix_cache)
                # print(cache.char)
                for child in node.children.values():
                    self.dfs(child, prefix + node.char, cache = cache, prefix_cache = prefix_cache) # search in child
            else:
                if node.count < cache.count * self.validation_threshold and node.count > cache.count * (1-self.validation_threshold):
                    cache.is_valid == True
                    # print(prefix_cache, cache.char)
                    # self.output.append((prefix_cache + cache.char, cache.count))
                    self.output[prefix_cache + cache.char] = cache.count
                if node.isleaf == True:
                        self.output[prefix + node.char] = node.count
                cache = node
                prefix_cache = prefix 
                for child in node.children.values():
                    self.dfs(child, prefix + node.char, cache=cache,  prefix_cache = prefix_cache) # search in child

        else:
     
            if cache is None:
                for child in node.children.values():
                    self.dfs(child, prefix + node.char) # search in child
            else:
                cache = cache
                prefix_cache = prefix_cache
                for child in node.children.values():
                    self.dfs(child, prefix + node.char, cache= cache, prefix_cache =prefix_cache) # search in child


        # if node.is_valid == True :#and node.count != -1:
        #     print('yessss')
        #     self.output.append((prefix + node.char, node.count))

    def search(self, x, top_n=50):
        """Given an input (a prefix), find all queries stored in
        the trie starting with the perfix, sort and return top_n queries based on the occurences.
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = {} # deny multiple chailds to validate their commun parent (this can be ovoided more effitiontly in the future)
        node = self.root
        
        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return []
        
        # Traverse the trie to get all candidates
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return
        # self.output.sort(key=lambda x: x[1], reverse=True)
        self.output = sorted(self.output.items(), key=lambda x:x[1], reverse= True)[:top_n]
        
        return self.output
