import os
import json
from sqlitedict import SqliteDict
import shutil
from tqdm import tqdm
import spacy

from utils_db import dict_save_json, dict_load_json, load_jsonl, dict_save_json, write_jsonl
from text_database import TextDatabase, Text
from vocabulary import Vocabulary, iter_phrases
from vocabulary import count_n_grams
from tfidf_database import TFIDFDatabase
from utils_doc_results import add_score_to_results, Claim, ClaimDocTokenizer

import config

def get_selection(path_predicted_documents, K, tf_idf_db, claim_data_set, nlp, title_tf_idf_flag_normalise):
    # description: 
    # input
    #   - path_predicted_documents: 
    #   - K: 

    if os.path.isfile(path_predicted_documents):
        print('file already exists')
    else:
        # === variables === #
        delimiter ='\k'
        method_tokenization = tf_idf_db.vocab.method_tokenization 

        # === initialise databases === #
        path_title_ids = tf_idf_db.path_ids_dict
        path_title_tf_idf = tf_idf_db.path_tf_idf_dict
        mydict_ids = SqliteDict(path_title_ids)
        mydict_tf_idf = SqliteDict(path_title_tf_idf)

        # === load claims === #
        claim_data_set_names = ['dev']
        if claim_data_set not in claim_data_set_names:
            raise ValueError('claim_data_set not in claim_data_set_names', claim_data_set, claim_data_set_names)
        path_dev_set = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR, claim_data_set + ".jsonl")
        results = load_jsonl(path_dev_set)
        
        if tf_idf_db.source == 'title' and title_tf_idf_flag_normalise == True:
            mydict_total_tf_idf = dict_load_json(tf_idf_db.path_total_tf_idf_dict)

        batch_sln = 10000
        list_claims = []
        for i in range(len(results)):
            claim = Claim(results[i])
            list_claims.append(claim.claim_without_dot)

        i = 0
        for doc in tqdm(nlp.pipe(iter_phrases(list_claims)), desc='pipeline', total = len(list_claims)):
            claim_doc_tokenizer = ClaimDocTokenizer(doc)
            n_grams, nr_words = claim_doc_tokenizer.get_n_grams(method_tokenization, tf_idf_db.vocab.n_gram)

            dictionary = {}

            for word in n_grams:
                try:
                    word_id_list = mydict_ids[word].split(delimiter)[1:]
                    word_tf_idf_list = mydict_tf_idf[word].split(delimiter)[1:]
                except KeyError:
                    word_id_list = []
                    word_tf_idf_list = []
                for j in range(len(word_id_list)):
                    id = int(word_id_list[j])       
                    tf_idf = float(word_tf_idf_list[j])
                    try:
                        dictionary[id] = dictionary[id] + tf_idf
                    except KeyError:
                        dictionary[id] = tf_idf

            if tf_idf_db.source == 'title' and title_tf_idf_flag_normalise == True:
                for id in dictionary:
                    total_tf_idf = mydict_total_tf_idf[str(id)]
                    dictionary[id] = dictionary[id] / float(total_tf_idf)

            keys_list = list(dictionary.keys())
            tf_idf_list = list(dictionary.values())

            dictionary = {}

            # make K best selection based on score
            selected_ids = sorted(range(len(tf_idf_list)), key=lambda l: tf_idf_list[l])[-K:]
            selected_ids = [keys_list[l] for l in selected_ids]

            results[i]['docs_selected'] = selected_ids
            i += 1
            
        write_jsonl(path_predicted_documents, results)

def compute_score(path_predicted_documents, score_method, tf_idf_db, nlp):
    # description: 1. load the predictions. 2. iterate through claims and compute the score
    # input:
    #   - path_predicted_documents : path to file with predictions (list of dictionaries)
    # output:
    #   - score

    results = load_jsonl(path_predicted_documents)

    nr_claims = 0
    nr_no_evidence = 0
    nr_title_not_in_dict = 0
    nr_supports = 0
    nr_not_enough_info = 0
    nr_refutes = 0

    label_list = ['SUPPORTS', 'NOT ENOUGH INFO', 'REFUTES']
    method_list = ["min_one", "overall_score"]

    if score_method == "f_score":
        score = 0.0
        for i in tqdm(range(len(results)), desc='scoring'):
            claim = Claim(results[i])

            nr_claims += 1

            score_flag = "incorrect"
            for interpreter in claim.evidence:
                for proof in interpreter:
                    title_proof = proof[2]
                    if title_proof == None:
                        score_flag = "no_evidence"
                    else:
                        try:
                            id_proof = tf_idf_db.vocab.title_2_id_dict[title_proof]
                            if id_proof in claim.docs_selected:
                                score_flag = "correct"
                        except KeyError:
                            score_flag = "title_not_in_dictionary"
                            break
            
            if score_flag == "correct":
                score += 1.0
                results = add_score_to_results(1, score_method, results, i)
            elif score_flag == 'title_not_in_dictionary':
                nr_title_not_in_dict += 1
                results = add_score_to_results('title_not_in_dictionary', score_method, results, i)
            elif score_flag == "no_evidence":
                nr_no_evidence += 1
                results = add_score_to_results('no_evidence', score_method, results, i)
            elif score_flag in ["valid_claim", "correct", "incorrect"]:
                results = add_score_to_results(0, score_method, results, i)
            else:
                raise ValueError('not a valid score_flag', score_flag)

        score = score / float(nr_claims - nr_no_evidence - nr_title_not_in_dict + 0.000001)

    elif score_method == "e_score":
        score = 0.0
        for i in tqdm(range(len(results)), desc='scoring'):
            claim = Claim(results[i])

            nr_claims += 1

            score_flag = "valid_claim"
            nr_interpreters = len(claim.evidence)
            score_item = 0.0
            for interpreter in claim.evidence:
                nr_proofs = len(interpreter)
                for proof in interpreter:
                    title_proof = proof[2]
                    if title_proof == None:
                        score_flag = "no_evidence"
                    else:
                        try:
                            id_proof = tf_idf_db.vocab.title_2_id_dict[title_proof]
                            if id_proof in claim.docs_selected:
                                score_item += 1 / float(nr_interpreters * nr_proofs)
                        except KeyError:
                            score_flag = "title_not_in_dictionary"
                            break
            score += score_item
            if score_flag == 'title_not_in_dictionary':
                nr_title_not_in_dict += 1
                results = add_score_to_results('title_not_in_dictionary', score_method, results, i)
            elif score_flag == "no_evidence":
                nr_no_evidence += 1
                results = add_score_to_results('no_evidence', score_method, results, i)
            elif score_flag == "valid_claim":
                results = add_score_to_results(score_item, score_method, results, i)
            else:
                raise ValueError('not a valid score_flag', score_flag)
        score = score / float(nr_claims - nr_no_evidence - nr_title_not_in_dict + 0.000001)

    elif score_method == "f_score_labelled":
        score = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_title_not_in_dict = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_no_evidence = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        count = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}

        for i in tqdm(range(len(results)), desc='scoring'):
            claim = Claim(results[i])

            nr_claims += 1
            count[claim.label] += 1

            score_flag = "incorrect"
            for interpreter in claim.evidence:
                for proof in interpreter:
                    title_proof = proof[2]
                    if title_proof == None:
                        score_flag = "no_evidence"
                    else:
                        try:
                            id_proof = tf_idf_db.vocab.title_2_id_dict[title_proof]
                            if id_proof in claim.docs_selected:
                                score_flag = "correct"
                        except KeyError:
                            score_flag = "title_not_in_dictionary"
                            break
            
            if score_flag == "correct":
                results = add_score_to_results(1, score_method, results, i)
                score[claim.label] += 1.0
            elif score_flag == 'title_not_in_dictionary':
                results = add_score_to_results('title_not_in_dictionary', score_method, results, i)
                nr_title_not_in_dict[claim.label] += 1
            elif score_flag == "no_evidence":
                results = add_score_to_results('no_evidence', score_method, results, i)
                nr_no_evidence[claim.label] += 1
            elif score_flag == "incorrect":
                results = add_score_to_results(0, score_method, results, i)
            else:
                raise ValueError('not a valid score_flag', score_flag)

        for label in label_list:
            score[label] = score[label] / float(count[label] - nr_title_not_in_dict[label] - nr_no_evidence[label] + 0.00001)

    elif score_method == "e_score_labelled":
        score = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_title_not_in_dict = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_no_evidence = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        count = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}

        for i in tqdm(range(len(results)), desc='scoring'):
            claim = Claim(results[i])

            nr_claims += 1
            count[claim.label] += 1

            score_flag = "valid_claim"
            nr_interpreters = len(claim.evidence)

            for interpreter in claim.evidence:
                nr_proofs = len(interpreter)
                for proof in interpreter:
                    title_proof = proof[2]
                    if title_proof == None:
                        score_flag = "no_evidence"
                    else:
                        try:
                            id_proof = tf_idf_db.vocab.title_2_id_dict[title_proof]
                            if id_proof in claim.docs_selected:
                                score[claim.label] += 1 / float(nr_interpreters * nr_proofs)
                        except KeyError:
                            score_flag = "title_not_in_dictionary"
                            break
            
            if score_flag == 'title_not_in_dictionary':
                nr_title_not_in_dict[claim.label] += 1
            elif score_flag == "no_evidence":
                nr_no_evidence[claim.label] += 1
            elif score_flag not in ["valid_claim", "correct", "incorrect"]:
                raise ValueError('not a valid score_flag', score_flag)

        for label in label_list:
            score[label] = score[label] / float(count[label] - nr_title_not_in_dict[label] - nr_no_evidence[label] + 0.0001)

    else:
        raise ValueError('no valid score_method', score_method)

    write_jsonl(path_predicted_documents, results)
    return score

if __name__ == '__main__':
    # === variables === #
    experiment_nr_list = [6,7,8,9,10]
    list_K = [5, 10, 20, 40]
    score_list = ['e_score', 'f_score', 'e_score_labelled', 'f_score_labelled']
    title_tf_idf_flag_normalise = True
    claim_data_set = 'dev'

    for experiment_nr in experiment_nr_list:
        
        nlp = spacy.load('en', disable=["parser", "ner"])

        file_name = 'experiment_%.2d.json'%(experiment_nr)
        path_experiment = os.path.join(config.ROOT, config.CONFIG_DIR, file_name)

        with open(path_experiment) as json_data_file:
            data = json.load(json_data_file)

        vocab_db = Vocabulary(path_wiki_database = os.path.join(config.ROOT, data['path_large_wiki_database']), 
            table_name_wiki = data['table_name_wiki'], n_gram = data['n_gram'],
            method_tokenization = data['method_tokenization'], source = data['vocabulary_source'])

        tf_idf_db = TFIDFDatabase(vocabulary = vocab_db, method_tf = data['method_tf'], method_df = data['method_df'],
            delimiter = data['delimiter'], threshold = data['threshold'], source = data['tf_idf_source'])
        
        if title_tf_idf_flag_normalise == True:
            dir_results = '01_results' + '_tf_idf_normalise'
        else:
            dir_results = '01_results'    

        if not os.path.isdir(os.path.join(tf_idf_db.base_dir, dir_results)):
            os.makedirs(os.path.join(tf_idf_db.base_dir, dir_results))

        
        file_name = 'score.json'
        path_score = os.path.join(tf_idf_db.base_dir, dir_results, file_name)

        if os.path.isfile(path_score):
            score_dict = dict_load_json(path_score)
        else:
            score_dict = {}

        for K in list_K:
            for score_method in score_list:
                file_name = 'predicted_labels_' + str(K) + '.json'

                path_predicted_documents = os.path.join(tf_idf_db.base_dir, dir_results, file_name)

                experiment_performed = False
                if str(K) in score_dict:
                    if score_method in score_dict[str(K)]:
                        experiment_performed = True

                if experiment_performed == False:
                    get_selection(path_predicted_documents, K, tf_idf_db, claim_data_set, nlp, title_tf_idf_flag_normalise)
                
                    score = compute_score(path_predicted_documents, score_method, tf_idf_db, nlp)
                    
                    if str(K) not in score_dict:
                        score_dict[str(K)] = {}

                    score_dict[str(K)][score_method] = score 

                    dict_save_json(score_dict, path_score)

