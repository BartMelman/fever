from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import json
from sqlitedict import SqliteDict
import shutil
from tqdm import tqdm
import spacy

from utils_db import dict_save_json, dict_load_json, load_jsonl, dict_save_json, write_jsonl, HiddenPrints, mkdir_if_not_exist
from utils_doc_results import ClaimDatabase
from wiki_database import WikiDatabaseSqlite
from vocabulary import VocabularySqlite, iter_phrases, count_n_grams, Text
from tfidf_database import TFIDFDatabaseSqlite
from utils_doc_results import add_score_to_results, Claim, ClaimDocTokenizer
from utils_doc_results_db import get_tf_idf_from_exp

import config

def get_selection(tf_idf_db, claim_database, title_tf_idf_flag_normalise):
    # description: 
    # input
    #   - path_predicted_documents: 
    #   - K: 

    if claim_database.scoring_flag == config.TRUE:
        print('claim database already exists for experiment')
    else:
        # === variables === #
        delimiter ='\k'
        method_tokenization = tf_idf_db.vocab.method_tokenization 

        # === initialise databases === #
        path_title_ids = tf_idf_db.path_ids_dict
        path_title_tf_idf = tf_idf_db.path_tf_idf_dict
        mydict_ids = SqliteDict(path_title_ids)
        mydict_tf_idf = SqliteDict(path_title_tf_idf)
        
        if title_tf_idf_flag_normalise == True:
            mydict_total_tf_idf = dict_load_json(tf_idf_db.path_total_tf_idf_dict)

        batch_sln = 10000
        list_claims = []
        for id in range(claim_database.nr_claims):
            claim = Claim(claim_database.get_claim_from_id(id))
            list_claims.append(claim.claim_without_dot)

        i = 0
        for doc in tqdm(tf_idf_db.vocab.wiki_database.nlp.pipe(iter_phrases(list_claims)), desc='pipeline', total = len(list_claims)):
            claim_doc_tokenizer = ClaimDocTokenizer(doc, tf_idf_db.vocab.delimiter_words, tf_idf_db.delimiter_tag_word, tf_idf_db.list_pos_tokenization)
            n_grams, nr_words = claim_doc_tokenizer.get_n_grams(method_tokenization, tf_idf_db.vocab.n_gram, tf_idf_db.tags_in_db_flag)

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

            if title_tf_idf_flag_normalise == True:
                for id in dictionary:
                    total_tf_idf = mydict_total_tf_idf[str(id)]
                    dictionary[id] = dictionary[id] / float(total_tf_idf)

            keys_list = list(dictionary.keys())
            tf_idf_list = list(dictionary.values())

            dictionary = {}

            # make K best selection based on score
            selected_ids = sorted(range(len(tf_idf_list)), key=lambda l: tf_idf_list[l])[-claim_database.K:]
            selected_ids = [keys_list[l] for l in selected_ids]

            claim_dict = claim_database.get_claim_from_id(i)
            claim_dict['docs_selected'] = selected_ids
            claim_database.write_claim_2_db(i, claim_dict)

            i += 1
            
        claim_database.set_scoring_flag(config.TRUE)

def compute_score(claim_database, score_method, tf_idf_db):
    # description: 1. load the predictions. 2. iterate through claims and compute the score
    # input:
    #   - path_predicted_documents : path to file with predictions (list of dictionaries)
    # output:
    #   - score

    # results = load_jsonl(path_predicted_documents)

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
        for i in tqdm(range(claim_database.nr_claims), desc='scoring'):
            claim = Claim(claim_database.get_claim_from_id(i))

            nr_claims += 1

            score_flag = "incorrect"
            for interpreter in claim.evidence:
                for proof in interpreter:
                    title_proof = proof[2]
                    if title_proof == None:
                        score_flag = "no_evidence"
                    else:
                        try:
                            id_proof = tf_idf_db.vocab.wiki_database.get_id_from_title(title_proof)
                            if id_proof in claim.docs_selected:
                                score_flag = "correct"
                        except KeyError:
                            print('title_not_in_dictionary', title_proof)
                            score_flag = "title_not_in_dictionary"
                            break
            
            if score_flag == "correct":
                score += 1.0
                claim_database.add_score(1, score_method, i)
            elif score_flag == 'title_not_in_dictionary':
                nr_title_not_in_dict += 1
                claim_database.add_score('title_not_in_dictionary', score_method, i)
            elif score_flag == "no_evidence":
                nr_no_evidence += 1
                claim_database.add_score('no_evidence', score_method, i)
            elif score_flag in ["valid_claim", "correct", "incorrect"]:
                claim_database.add_score(0, score_method, i)
            else:
                raise ValueError('not a valid score_flag', score_flag)

        score = score / float(nr_claims - nr_no_evidence - nr_title_not_in_dict + 0.000001)

    elif score_method == "e_score":
        score = 0.0
        for i in tqdm(range(claim_database.nr_claims), desc='scoring'):
            claim = Claim(claim_database.get_claim_from_id(i))

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
                            id_proof = tf_idf_db.vocab.wiki_database.get_id_from_title(title_proof)
                            if id_proof in claim.docs_selected:
                                score_item += 1 / float(nr_interpreters * nr_proofs)
                        except KeyError:
                            print('title_not_in_dictionary', title_proof)
                            score_flag = "title_not_in_dictionary"
                            break
            score += score_item
            if score_flag == 'title_not_in_dictionary':
                nr_title_not_in_dict += 1
                claim_database.add_score('title_not_in_dictionary', score_method, i)
            elif score_flag == "no_evidence":
                nr_no_evidence += 1
                claim_database.add_score('no_evidence', score_method, i)
            elif score_flag == "valid_claim":
                claim_database.add_score(score_item, score_method, i)
            else:
                raise ValueError('not a valid score_flag', score_flag)
        score = score / float(nr_claims - nr_no_evidence - nr_title_not_in_dict + 0.000001)

    elif score_method == "f_score_labelled":
        score = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_title_not_in_dict = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_no_evidence = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        count = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}

        for i in tqdm(range(claim_database.nr_claims), desc='scoring'):
            claim = Claim(claim_database.get_claim_from_id(i))

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
                            id_proof = tf_idf_db.vocab.wiki_database.get_id_from_title(title_proof)
                            if id_proof in claim.docs_selected:
                                score_flag = "correct"
                        except KeyError:
                            print('title_not_in_dictionary', title_proof)
                            score_flag = "title_not_in_dictionary"
                            break
            
            if score_flag == "correct":
                claim_database.add_score(1, score_method, i)
                score[claim.label] += 1.0
            elif score_flag == 'title_not_in_dictionary':
                claim_database.add_score('title_not_in_dictionary', score_method, i)
                nr_title_not_in_dict[claim.label] += 1
            elif score_flag == "no_evidence":
                claim_database.add_score('no_evidence', score_method, i)
                nr_no_evidence[claim.label] += 1
            elif score_flag == "incorrect":
                claim_database.add_score(0, score_method, i)
            else:
                raise ValueError('not a valid score_flag', score_flag)

        for label in label_list:
            score[label] = score[label] / float(count[label] - nr_title_not_in_dict[label] - nr_no_evidence[label] + 0.00001)

    elif score_method == "e_score_labelled":
        score = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_title_not_in_dict = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        nr_no_evidence = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}
        count = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 0, 'REFUTES': 0}

        for i in tqdm(range(claim_database.nr_claims), desc='scoring'):
            claim = Claim(claim_database.get_claim_from_id(i))

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
                            id_proof = tf_idf_db.vocab.wiki_database.get_id_from_title(title_proof)
                            if id_proof in claim.docs_selected:
                                score[claim.label] += 1 / float(nr_interpreters * nr_proofs)
                        except KeyError:
                            print('title_not_in_dictionary', title_proof)
                            score_flag = "title_not_in_dictionary"
                            break
            
            if score_flag == 'title_not_in_dictionary':
                nr_title_not_in_dict[claim.label] += 1
                claim_database.add_score('title_not_in_dictionary', score_method, i)
            elif score_flag == "no_evidence":
                nr_no_evidence[claim.label] += 1
                claim_database.add_score('no_evidence', score_method, i)
            elif score_flag == "valid_claim":
                claim_database.add_score(1, score_method, i)
            else:
                raise ValueError('not a valid score_flag', score_flag)

        for label in label_list:
            score[label] = score[label] / float(count[label] - nr_title_not_in_dict[label] - nr_no_evidence[label] + 0.0001)

    else:
        raise ValueError('no valid score_method', score_method)

    # write_jsonl(path_predicted_documents, results)
    return score

class PerformanceTFIDF():
    def __init__(self, wiki_database, experiment_nr, claim_data_set, K, score_method, title_tf_idf_normalise_flag):
        # === process inputs === #
        # self.wiki_database = wiki_database
        self.experiment_nr = experiment_nr
        self.claim_data_set = claim_data_set
        self.K = K
        self.score_method = score_method
        self.title_tf_idf_normalise_flag = title_tf_idf_normalise_flag

        # === process === #
        print('PerformanceTFIDF', experiment_nr, claim_data_set, K, score_method, title_tf_idf_normalise_flag)
        self.path_raw_data = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR)

        with HiddenPrints():
            self.tf_idf_db = get_tf_idf_from_exp(self.experiment_nr, wiki_database)

        if self.title_tf_idf_normalise_flag == True:
            self.path_dir_experiment = os.path.join(self.tf_idf_db.base_dir, 'results_experiment_' + self.claim_data_set + '_tf_idf_normalise_' + str(K))
        elif self.title_tf_idf_normalise_flag == False:
            self.path_dir_experiment = os.path.join(self.tf_idf_db.base_dir, 'results_experiment_' + self.claim_data_set + str(K))
        else:
            raise ValueError('tf idf normalise flag is not True or False', self.title_tf_idf_normalise_flag)

        self.path_score_dict = os.path.join(self.path_dir_experiment, 'score.json')

        mkdir_if_not_exist(self.path_dir_experiment)

        if os.path.isfile(self.path_score_dict):
            self.score_dict = dict_load_json(self.path_score_dict)
        else:
            self.score_dict = {}

        self.claim_database = ClaimDatabase(path_dir_database = self.path_dir_experiment, path_raw_data = self.path_raw_data, claim_data_set = self.claim_data_set, K = K)

        if self.score_method not in self.score_dict:
            self.get_selection()
            score = self.compute_score()
            self.score_dict[score_method] = score 
            self.save_score_dict()
        else:
            print('score already computed')

    def save_score_dict(self):
        dict_save_json(self.score_dict, self.path_score_dict)

    def get_selection(self):
        get_selection(self.tf_idf_db, self.claim_database, self.title_tf_idf_normalise_flag)

    def compute_score(self):
        return compute_score(self.claim_database, self.score_method, self.tf_idf_db)

if __name__ == '__main__':
    # === constants === #
    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
    
    # === variables === #
    claim_data_set = 'train_adj' # 
    experiment_nr_list = [31, 37] # [31,32,33,34,35,36,37,38,39]
    list_K = [5, 100]
    score_list = ['e_score', 'f_score', 'e_score_labelled', 'f_score_labelled']
    title_tf_idf_flag_normalise_list = [True, False]

    # === process === #
    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)
    
    for experiment_nr in experiment_nr_list:  
        for title_tf_idf_normalise_flag in title_tf_idf_flag_normalise_list:
            for K in list_K:
                for score_method in score_list:
            
                    PerformanceTFIDF(wiki_database, experiment_nr, claim_data_set, K, score_method, title_tf_idf_normalise_flag)
