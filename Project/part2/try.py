import pickle
import math
import spacy
import numpy as np
import xgboost as xgb
from collections import defaultdict 

import sys


def transform_data(features, groups, labels=None):
    xgb_data = xgb.DMatrix(data=features, label=labels)
    xgb_data.set_group(groups)
    return xgb_data


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    tf_tokens = defaultdict(int)
    idf_tokens = defaultdict(int)
    tf_tokens, idf_tokens = index_documents(tf_tokens, idf_tokens, men_docs)
    nlp = spacy.load("en_core_web_sm")
    print("index done")

    mentions_candidates = []
    labels_train = []
    train_groups = []
    train_data = []
    train_data = np.array(train_data)

    all_mentions_att = []

    for index in train_mentions:
        max_score = 0
        max_candidate = ""
        candidate_scores = defaultdict(int)
        
        length_equal = 0
        
        cur_train = train_mentions[index]
        target = train_labels[index]['label']
        title = cur_train['doc_title']
        train_groups.append(len(cur_train['candidate_entities']))
        all_mentions_att.append(cur_train['mention'])

        # get tf-idf score for the mention token
        i = 0
        for candidate in cur_train['candidate_entities']:
            men_in_can = 0
            can_in_men = 0
            #name_substring = 0
            text_substring = 0
            check_first = 0
            appear_times = 0
            correspond_doc = parsed_entity_pages[candidate]

            can_arr = candidate.lower().split('_')
            men_arr = cur_train['mention'].lower().split()
            first_letter = ""
            
            for ele in can_arr:
                if ele == '':
                    continue
                first_letter += ele[0]

            tokens_score = 0
            for token in correspond_doc:
                if token[1] in tf_tokens and title in tf_tokens[token[1]] and token[4] != 'O':
                    #print(token[1])
                    tokens_score += (1 + math.log(1 + math.log(tf_tokens[token[1]][title]))) * idf_tokens[token[1]]
                if token[1].lower() in men_arr:
                    appear_times += 1  
                # elif token[2] in tf_tokens and title in tf_tokens[token[2]]:
                #     tokens_score += (1 + math.log(1 + math.log(tf_tokens[token[2]][title]))) * idf_tokens[token[2]]

            sum_union = 0
            if len(can_arr) > len(men_arr):
                for ele in men_arr:
                    if ele in can_arr:
                        can_arr.remove(ele)
                sum_union = len(can_arr)
                if sum_union == 0:
                    sum_union == 1
                else:
                    sum_union = 1 / sum_union
            else:
                for ele in can_arr:
                    if ele in men_arr:
                        men_arr.remove(ele)
                sum_union = len(men_arr)
                if sum_union == 0:
                    sum_union == 1
                else:
                    sum_union = 1 / sum_union
            # if candidate.lower() in cur_train['mention'].lower():
            #     can_in_men = len(can_arr)
            # if cur_train['mention'].lower() in candidate.lower():
            #     men_in_can = 1/len(can_arr)
            can_sub_men = 0
            if len(can_arr) == 1 and len(men_arr) == 1 and can_arr[0] in men_arr[0]:
                can_sub_men = 1
            i_1 = 0
            
            if len(can_arr) == len(men_arr):
                while i_1 < len(can_arr):
                    if can_arr[i_1] == '':
                        continue
                    if men_arr[i_1] == '':
                        continue
                    if can_arr[i_1].lower() == men_arr[i_1].lower():
                        text_substring = 1
                    else:
                        text_substring = 0
                        break
                    i_1 += 1
            # if candidate.lower() == cur_train['mention'].lower():
            #     text_substring = 1
            if first_letter == cur_train['mention'].lower():
                check_first = 1
            
            if len(candidate) - len(cur_train['mention']) < 0:
                length_equal = len(cur_train['mention']) - len(candidate)
            else:
                length_equal = len(candidate) - len(cur_train['mention'])
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
            train_data = np.append(train_data, [tokens_score, num_intersection, appear_times])

            if candidate != target:
                labels_train.append(0)
            else:
                labels_train.append(1)
            i += 1
    #sys.exit()
    # for ele in all_mentions_att:
    #     print('mention: ' + ele + '     tag: ' + nlp(ele))
    # sys.exit()
    train_groups = np.array(train_groups)
    train_data = np.reshape(train_data, (-1, 3))

    labels_train = np.array(labels_train) 

    xgboost_train = transform_data(train_data, train_groups, labels_train)
    print("train_data transform done")

    # generate features for the test data:

    labels_test = []
    test_groups = []
    test_data = []
    test_data = np.array(test_data)
    all_candidates = []
    all_labels = []

    for index in dev_mentions:
        candidate_scores = defaultdict(int)
        max_score = 0
        max_candidate = ""
        cur_test = dev_mentions[index]
        title = cur_test['doc_title']
        all_labels.append(cur_test['mention'])
        test_groups.append(len(cur_test['candidate_entities']))

        for candidate in cur_test['candidate_entities']:
            # Each tuple contains (mention, candidate_entity)
            all_candidates.append(candidate)

            men_in_can = 0
            can_in_men = 0
            text_substring = 0
            appear_times = 0
            check_first = 0
            correspond_doc = parsed_entity_pages[candidate]

            can_arr = candidate.lower().split('_')
            men_arr = cur_test['mention'].lower().split()
            first_letter = ""
            for ele in can_arr:
                if ele == '':
                    continue
                first_letter += ele[0]
            
            length_equal = 0
            tokens_score = 0
            for token in correspond_doc:
                if token[1] in tf_tokens and title in tf_tokens[token[1]]:
                    tokens_score += (1 + math.log(1 + math.log(tf_tokens[token[1]][title]))) * idf_tokens[token[1]]
                if token[1].lower() in men_arr:
                    appear_times += 1  
                # elif token[2] in tf_tokens and title in tf_tokens[token[2]] :
                #     tokens_score += (1 + math.log(1 + math.log(tf_tokens[token[2]][title]))) * idf_tokens[token[2]]

            # candidate_scores[candidate] = tokens_score
            # if tokens_score > max_score:
            #     max_score = tokens_score
            #     max_candidate = candidate
            sum_union = 0
            if len(can_arr) > len(men_arr):
                for ele in men_arr:
                    if ele in can_arr:
                        can_arr.remove(ele)
                sum_union = len(can_arr)
                if sum_union == 0:
                    sum_union == 1
                else:
                    sum_union = 1 / sum_union
            else:
                for ele in can_arr:
                    if ele in men_arr:
                        men_arr.remove(ele)
                sum_union = len(men_arr)
                if sum_union == 0:
                    sum_union == 1
                else:
                    sum_union = 1 / sum_union
            # if candidate.lower() in cur_test['mention'].lower():
            #     can_in_men = len(can_arr)
            # if cur_test['mention'].lower() in candidate.lower():
            #     men_in_can = 1/len(can_arr)
            can_sub_men = 0
            if len(can_arr) == 1 and len(men_arr) == 1 and can_arr[0] in men_arr[0]:
                can_sub_men = 1
            # if candidate.lower() == cur_test['mention'].lower():
            #     text_substring = 1
            if len(can_arr) == len(men_arr):
                while i_1 < len(can_arr):
                    if can_arr[i_1] == '':
                        continue
                    if men_arr[i_1] == '':
                        continue
                    if can_arr[i_1].lower() == men_arr[i_1].lower():
                        text_substring = 1
                    else:
                        text_substring = 0
                        break
                    i_1 += 1
            if first_letter == cur_test['mention'].lower():
                check_first = 1

            if len(candidate) - len(cur_train['mention']) < 0:
                length_equal = len(cur_test['mention']) - len(candidate)
            else:
                length_equal = len(candidate) - len(cur_test['mention'])
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
            test_data = np.append(test_data, [tokens_score, num_intersection, appear_times])

    test_data = np.reshape(test_data, (-1, 3))
    labels_test = np.array(labels_test)
    # # Form groups
    # idxs = np.where(labels_test == 1)[0]

    # test_groups = np.append(np.delete(idxs, 0), len(labels_test)) - idxs
    xgboost_test = transform_data(test_data, test_groups)

    param = {'max_depth': 7, 'eta': 0.06, 'silent': 1, 'objective': 'rank:pairwise',
         'min_child_weight': 0.015, 'lambda':90, 'n_estimators':5000}

    classifier = xgb.train(param, xgboost_train, num_boost_round=4900)
    ##  Predict test data...
    preds = classifier.predict(xgboost_test)

    idx = 0
    result_labels = {}
    
    for iter_, group in enumerate(test_groups):

        list_iter = preds[idx:idx+group].tolist()
        #print(list_iter.index(max(list_iter)))
        if iter_ > 100:
            print("mention for predict: " + all_labels[iter_])
            print("Prediction scores for Group {} = {}".format(iter_,preds[idx:idx+group]))
            print(all_candidates[idx:idx+group])
        result_labels[iter_ + 1] = dev_mentions[iter_ + 1]['candidate_entities'][list_iter.index(max(list_iter))]
        #print(result_labels[iter_])
        idx+=group
    
    print("prediction done")
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


## We will be using the following function to compute the accuracy...
def compute_accuracy(result, data_labels):

    assert set(list(result.keys())) - set(list(data_labels.keys())) == set()
    TP = 0.0
    for id_ in result.keys():
        if result[id_] == data_labels[id_]['label']:
            TP +=1
        else:
            if id_ > 100:
                print(str(id_) + ' pred: ' + result[id_] + '   true: ' + data_labels[id_]['label'])
    assert len(result) == len(data_labels)
    return TP/len(result)


if __name__ == "__main__":

    source_folder = './Data2/'
    ### Read the Training Data
    train_file = source_folder + 'train.pickle'
    train_mentions = pickle.load(open(train_file, 'rb'))

    ### Read the Training Labels...
    train_label_file = source_folder + 'train_labels.pickle'
    train_labels = pickle.load(open(train_label_file, 'rb'))
    # print(train_mentions)
    # print(train_labels)
    # sys.exit()
    ### Read the Dev Data... (For Final Evaluation, we will replace it with the Test Data)
    dev_file = source_folder + 'dev.pickle'
    dev_mentions = pickle.load(open(dev_file, 'rb'))

    ### Read the Parsed Entity Candidate Pages...
    fname = source_folder + 'parsed_candidate_entities.pickle'
    parsed_entity_pages = pickle.load(open(fname, 'rb'))

    ### Read the Mention docs...
    mens_docs_file = source_folder + "men_docs.pickle"
    men_docs = pickle.load(open(mens_docs_file, 'rb'))
    #print(men_docs)
    ## Result of the model...
    result = disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages)
    
    # for key in list(result)[:5]:
    #     print('KEY: {} \t VAL: {}'.format(key,result[key]))

    ### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data)
    dev_label_file = source_folder + 'dev_labels.pickle'
    dev_labels = pickle.load(open(dev_label_file, 'rb'))
    
    accuracy = compute_accuracy(result, dev_labels)
    print("Accuracy = ", accuracy)

