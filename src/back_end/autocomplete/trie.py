from src.back_end.autocomplete import spell

from collections import Counter
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
    def __init__(self, validation_threshold, max_number_of_words):
        """
        Initiate the trie with an empty char
        """
        self.root = TrieNode("")
        self.validation_threshold = validation_threshold
        self.max_number_of_words = max_number_of_words
        self.checker = spell.Spell()
    def trim_sugg(sugg, max_number_of_words):
        return sugg[:max_number_of_words]

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

    def dfs(self, node, prefix, cache = None, prefix_cache = None, number_of_words = 0):
        """Depth-first traversal of the trie
        :param node (TrieNode): the root of the subtree
        :param prefix (string): the current prefix to form the full queries when traversing the trie
        """
        search_term = prefix + node.char
        if node.count != -1 :
            # do spell checking 
            if number_of_words >= self.max_number_of_words:
                self.output[search_term] = node.count

            else:
                number_of_words +=1
                if cache is None:
                        cache = node # first cache
                        prefix_cache = prefix
                        # print(prefix_cache)
                        # print(cache.char)
                        for child in node.children.values():
                            self.dfs(child, search_term, cache = cache, prefix_cache = prefix_cache, number_of_words=number_of_words) # search in child
                else:
                    if node.count < cache.count * self.validation_threshold and node.count > cache.count * (1-self.validation_threshold):
                        cache.is_valid == True
                        # print(prefix_cache, cache.char)
                        # self.output.append((prefix_cache + cache.char, cache.count))
                        self.output[prefix_cache + cache.char] = cache.count
                    if len(node.children.values() )== 0:
                            self.output[search_term] = node.count
                    cache = node
                    prefix_cache = prefix 
                    for child in node.children.values():
                        self.dfs(child, search_term, cache=cache,  prefix_cache = prefix_cache, number_of_words=number_of_words) # search in child
        else:
     
            if cache is None:
                for child in node.children.values():
                    self.dfs(child, search_term) # search in child
            else:
                cache = cache
                prefix_cache = prefix_cache
                for child in node.children.values():
                    self.dfs(child, search_term, cache= cache, prefix_cache =prefix_cache, number_of_words=number_of_words) # search in child


        # if node.is_valid == True :#and node.count != -1:
        #     print('yessss')
        #     self.output.append((prefix + node.char, node.count))

    def check_if_valid(self, new_prefix):
        node= self.root
        is_valid = True
        for char in new_prefix:
                if char in node.children:
                    node = node.children[char]
                else:
                    is_valid = False
        return node, is_valid

    def spell_correction(self, original_prefix):
                other_combinations = []
                words = original_prefix.split()
                correction_target = words[-1]
                corrections = self.checker.most_likely_replacements(correction_target, 4)
                for correct_word in corrections:
                        new_prefix = " ".join(words[:-1]) + " " + correct_word 
                        other_combinations.appned(new_prefix)

                        node, is_valid = self.check_if_valid(new_prefix)
                        if is_valid:
                                self.dfs(node, new_prefix[:-1])
    
    def get_combinations(self, curr_combos, misspeled_word):
                final_combos = set()
                for combo in curr_combos:

                    corrections = self.checker.most_likely_replacements(misspeled_word, 3)
                    for corrected_word in corrections:
                            new_prefix = combo.replace(misspeled_word, corrected_word)
                            final_combos.add(new_prefix)
                return final_combos
    # def is_correct(self, correct_word):
    #      node =self.root:
    #      for char in 
    #      return True

    def get_all_possible_corrections(self, words, spell_error_index, treated_corrections):
        # first we need to change the misspeled words with the correct words form the correction list
        new_prefix_list = set()
        misspelled_word = words[spell_error_index]
        correction_candidate = self.checker.most_likely_replacements(misspelled_word, 4)
        for candidate in correction_candidate:
                        if candidate not in treated_corrections:
                            treated_corrections.add(candidate)
                            words[spell_error_index] = candidate
                            new_prefix_list.add(' '.join(words))
        return new_prefix_list, treated_corrections


    def search_x(self, x, treated_corrections, cache):
        """Given an input (a prefix), find all queries stored in
        the trie starting with the perfix, sort and return top_n queries based on the occurences.
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = dict() # deny multiple chailds to validate their commun parent (this can be ovoided more effitiontly in the future)
        node = self.root
        spell_error = False
        # Check if the prefix is in the trie
        words = x.split()
        self.correct_prefix_combinations = {x}
        spell_error_index = 0
        for char in x:
            if char in node.children:
                if char == ' ':
                     spell_error_index += 1
                node = node.children[char]
            else:
                spell_error = True
                # get list of correct prefix possibilities
                misspelled_word = words[spell_error_index]
                correction_candidate = self.checker.most_likely_replacements(misspelled_word, 4)
                for candidate in correction_candidate:

                                if candidate not in treated_corrections:
                                    treated_corrections.add(candidate)
                                    words[spell_error_index] = candidate
                                    new_prefix = ' '.join(words)
                                    self.output.update(self.search_x(new_prefix, treated_corrections=treated_corrections, cache= candidate))
                break

                # return new_prefix_list, treated_corrections
                # new_prefix_list, treated_corrections = self.get_all_possible_corrections(words, word_count, treated_corrections)

        if not spell_error:
            self.dfs(node, x[:-1])
        # remove the candidate back from the mist
 
        return self.output

    def search(self, x, top_suggs):
        self.output = {}
        self.final_output = self.search_x(x, set(), cache = '')
        # self.sorted_outputs = sorted(self.final_output.items(), key=lambda x:x[1], reverse= True)
        return dict(Counter(self.final_output).most_common(top_suggs))
