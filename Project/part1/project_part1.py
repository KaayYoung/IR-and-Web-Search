## Import Libraries and Modules here...
import spacy
import pickle
import math
import copy
from collections import defaultdict
from itertools import combinations


# function used to add the item into its classification(entity or token)
def storeItem(i, item, classification): 
    if item not in classification:
        posting = defaultdict(int)
        posting[i] = 1
        classification[item] = posting
    elif i not in classification[item]:
        posting = classification[item]
        posting[i] = 1
    else:
        posting = classification[item]
        posting[i] = posting[i] + 1
    
    return classification


# Get corrsponding tokens for each entities combination
def getTokens(query, entities):
    for e in entities:
        for word in e.split():
            query = query.replace(word, '', 1)

    return query.split()


class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = defaultdict(dict)
        self.tf_entities = defaultdict(dict)

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = defaultdict(dict)
        self.idf_entities = defaultdict(dict)

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        nlp = spacy.load("en_core_web_sm")
        
        # Iterate over each document
        for index, value in documents.items():
            doc = nlp(value)
            single_ent_pos = []
            
            # Find entity first, if it is entity, add it into self.tf_entities
            # Add all positions of single entities into array
            for ent in doc.ents:
                self.tf_entities = storeItem(index, ent.text, self.tf_entities)
                if (ent.end - ent.start) == 1:
                    single_ent_pos.append(ent.start)

            # add all validated token into self.tf_tokens
            for token in doc:
                if not token.is_stop and not token.is_punct and token.text and token.i not in single_ent_pos:
                    self.tf_tokens = storeItem(index, token.text, self.tf_tokens)

        N = len(documents)
        # calculate idf for tokens and entities
        for token, post in self.tf_tokens.items():
            self.idf_tokens[token] = 1.0 + math.log(N / (1.0 + len(post)))
        for entity, post in self.tf_entities.items():
            self.idf_entities[entity] = 1.0 + math.log(N / (1.0 + len(post)))


    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):

        split_res = defaultdict(dict)
        split_dict = defaultdict(list)

        split_dict['entities'] = []
        split_dict['tokens'] = Q.split()
        split_res[0] = split_dict

        max_len = 1
        copy_Q = copy.copy(Q)

        # Get the maximum length of entities
        for entity in DoE:
            len_entities = len(entity.split())
            if max_len < len_entities:
                max_len = len_entities

        token_count = defaultdict(int)
        # Get the appearance time of each token
        for ele in copy_Q.split():
            if ele not in token_count:
                token_count[ele] = 1
            else:
                token_count[ele] += 1

        # Get all combinations which may be valiadated
        candidate = []
        for index in range(1, max_len + 1):
            combination_arr = list(combinations(copy_Q.split(), index))
            for tup in combination_arr:
                str = ' '.join(tup)
                if str in DoE and str not in candidate:
                    candidate.append(str)
        
        i = 1
        for index in range(1, len(candidate) + 1):
            combination_arr = list(combinations(candidate, index))
            for tup in combination_arr:
                combination_count = defaultdict(int)
                split_dict = defaultdict(list)
                flag = 0
                for ele in list(tup):
                    for word in ele.split():
                        if word not in combination_count:
                            combination_count[word] = 1
                        else:
                            combination_count[word] += 1
                # Check whether the number of appearance time of each token is bigger than it in orignal dict
                for ele in combination_count:
                    if combination_count[ele] > token_count[ele]:
                        flag = 1
                        break
                if flag == 0:
                    split_dict['tokens'] = getTokens(copy_Q, list(tup))
                    split_dict['entities'] = list(tup)
                    split_res[i] = split_dict
                    i += 1

        return split_res


    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        
        max_combined = 0
        max_index = 0
        max_list = []

        for index in query_splits:

            scores = defaultdict(int)
            tokens_entities = query_splits[index]
            tokens_score = 0
            entities_score = 0

            for ele in query_splits[index]['tokens']:
                if ele in self.tf_tokens and doc_id in self.tf_tokens[ele]:
                    tokens_score += (1 + math.log(1 + math.log(self.tf_tokens[ele][doc_id]))) * self.idf_tokens[ele]
            scores['tokens'] = tokens_score

            for ele in query_splits[index]['entities']:
                if ele in self.tf_entities and doc_id in self.tf_entities[ele]:
                    entities_score += (1 + math.log(self.tf_entities[ele][doc_id])) * self.idf_entities[ele]
            scores['entities'] = entities_score

            scores['combined'] = scores['entities'] + 0.4 * scores['tokens']

            if max_combined < scores['combined']:
                max_combined = scores['combined']
                max_index = index

        max_list.append(max_combined)
        max_list.append(query_splits[max_index])
        return tuple(max_list)
