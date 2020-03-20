import pickle
import math
import spacy
import numpy as np
import xgboost as xgb
from collections import defaultdict 

import sys


# transform to DMatrix
def transform_data(features, groups, labels=None):
    xgb_data = xgb.DMatrix(data=features, label=labels)
    xgb_data.set_group(groups)
    return xgb_data


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    tf_tokens = defaultdict(int)
    idf_tokens = defaultdict(int)
    tf_tokens, idf_tokens = index_documents(tf_tokens, idf_tokens, men_docs)
    nlp = spacy.load("en_core_web_sm")

    mentions_candidates = []
    labels_train = []
    train_groups = []
    train_data = []
    train_data = np.array(train_data)

    for index in train_mentions:
        cur_train = train_mentions[index]
        target = train_labels[index]['label']
        title = cur_train['doc_title']
        train_groups.append(len(cur_train['candidate_entities']))

        for candidate in cur_train['candidate_entities']:

            correspond_doc = parsed_entity_pages[candidate]

            can_arr = candidate.lower().split('_')
            men_arr = cur_train['mention'].lower().split()
            
            # Feature 1: tf-idf score
            tokens_score = 0
            for token in correspond_doc:
                if token[1] in tf_tokens and title in tf_tokens[token[1]]:
                    tokens_score += (1 + math.log(1 + math.log(tf_tokens[token[1]][title]))) * idf_tokens[token[1]]

            # Feature 2: check the similarity between each candidate and the mention
            num_intersection = 0
            if len(can_arr) > len(men_arr):
                for ele in men_arr:
                    if ele in can_arr:
                        num_intersection += 1
                num_intersection = num_intersection / len(can_arr)
            else:
                for ele in can_arr:
                    if ele in men_arr:
                        num_intersection += 1
                num_intersection = num_intersection / len(men_arr)
                
            # Feature 3: check length diff between each candidate and the mention
            length_diff = abs(len(candidate) - len(cur_train['mention']))

            train_data = np.append(train_data, [tokens_score, num_intersection, length_diff])

            if candidate != target:
                labels_train.append(0)
            else:
                labels_train.append(1)
    
    train_groups = np.array(train_groups)
    train_data = np.reshape(train_data, (-1, 3))
    labels_train = np.array(labels_train) 

    xgboost_train = transform_data(train_data, train_groups, labels_train)

    # generate features for the test data: same as training data
    labels_test = []
    test_groups = []
    test_data = []
    test_data = np.array(test_data)

    for index in dev_mentions:
        cur_test = dev_mentions[index]
        title = cur_test['doc_title']
        test_groups.append(len(cur_test['candidate_entities']))

        for candidate in cur_test['candidate_entities']:

            correspond_doc = parsed_entity_pages[candidate]

            can_arr = candidate.lower().split('_')
            men_arr = cur_test['mention'].lower().split()
            
            tokens_score = 0
            for token in correspond_doc:
                if token[1] in tf_tokens and title in tf_tokens[token[1]]:
                    tokens_score += (1 + math.log(1 + math.log(tf_tokens[token[1]][title]))) * idf_tokens[token[1]]

            num_intersection = 0
            if len(can_arr) > len(men_arr):
                for ele in men_arr:
                    if ele in can_arr:
                        num_intersection += 1
                num_intersection = num_intersection / len(can_arr)
            else:
                for ele in can_arr:
                    if ele in men_arr:
                        num_intersection += 1
                num_intersection = num_intersection / len(men_arr)

            length_diff = abs(len(candidate) - len(cur_test['mention']))

            test_data = np.append(test_data, [tokens_score, num_intersection, length_diff])

    test_data = np.reshape(test_data, (-1, 3))
    labels_test = np.array(labels_test)
    xgboost_test = transform_data(test_data, test_groups)

    ## Train the model
    param = {'max_depth': 8, 'eta': 0.07, 'silent': 1, 'objective': 'rank:pairwise',
         'min_child_weight': 0.015, 'lambda':90, 'n_estimators':5000}
    classifier = xgb.train(param, xgboost_train, num_boost_round=4900)
    
    ##  Predict test data...
    preds = classifier.predict(xgboost_test)

    idx = 0
    result_labels = {}
    
    for iter_, group in enumerate(test_groups):
        list_iter = preds[idx:idx+group].tolist()
        result_labels[iter_ + 1] = dev_mentions[iter_ + 1]['candidate_entities'][list_iter.index(max(list_iter))]
        idx+=group
    
    return result_labels
                

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


def index_documents(tokens_tf, tokens_idf, documents):
    nlp = spacy.load("en_core_web_sm")
    
    tf_entities = defaultdict(int)
    # Iterate over each document
    for index, value in documents.items():
        doc = nlp(value)

        single_ent_pos = []
        for ent in doc.ents:
            entities = storeItem(index, ent.text, tf_entities)
            if (ent.end - ent.start) == 1:
                single_ent_pos.append(ent.start)
        # add all validated token into self.tf_tokens
        for token in doc:
            if not token.is_stop and not token.is_punct and token.text and token.i not in single_ent_pos:
                tokens_tf = storeItem(index, token.text, tokens_tf)

    N = len(documents)
    # calculate idf for tokens and entities
    for token, post in tokens_tf.items():
        tokens_idf[token] = 1.0 + math.log(N / (1.0 + len(post)))
    
    return tokens_tf, tokens_idf


