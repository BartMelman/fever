import os
import json
from sqlitedict import SqliteDict
import shutil
from tqdm import tqdm
import spacy
import unicodedata

from wiki_database import Text
from utils_wiki_database import normalise_text
from utils_db import dict_save_json, dict_load_json, load_jsonl, dict_save_json, write_jsonl, HiddenPrints
from vocabulary import VocabularySqlite
from vocabulary import count_n_grams
from tfidf_database import TFIDFDatabaseSqlite

import config

def add_score_to_results(score, method, results, i):
    results_key_name = 'results'
    if results_key_name not in results[i]:
        results[i][results_key_name] = {}

    results[i][results_key_name][method] = score

    return results

class Claim:
    def __init__(self, claim_dictionary):
        self.id = claim_dictionary['id']
        self.verifiable = claim_dictionary['verifiable']
        self.label = claim_dictionary['label']
        self.claim = claim_dictionary['claim']
        self.claim_without_dot = self.claim[:-1]
        self.evidence = claim_dictionary['evidence']
        # self.nlp = nlp

        if 'docs_selected' in claim_dictionary:
            self.docs_selected = claim_dictionary['docs_selected']

class ClaimDocTokenizer:
    def __init__(self, doc, delimiter_words, delimiter_tags, list_pos_tokenization):
        self.doc = doc
        self.delimiter_words = delimiter_words
        self.delimiter_tags = delimiter_tags
        self.list_pos_tokenization = list_pos_tokenization

    def get_tokenized_claim(self, method_tokenization):
        # claim_without_dot = self.claim[:-1]  # remove . at the end
        # doc = self.nlp(claim_without_dot)
        text = Text(self.doc)
        tokenized_claim = text.process(method_tokenization)
        return tokenized_claim

    def get_n_grams(self, method_tokenization, n_gram, tags_in_db_flag):
        tokenized_claim = self.get_tokenized_claim(method_tokenization)
        # print(tags_in_db_flag, method_tokenization, self.list_pos_tokenization)
        if tags_in_db_flag == 0 and method_tokenization[0] in self.list_pos_tokenization:
            tmp = []
            for tokenized_word in tokenized_claim :
                tag, word = get_tag_word_from_wordtag(tokenized_word, self.delimiter_tags)
                tmp.append(word)
            tokenized_claim = tmp
        return count_n_grams(tokenized_claim, n_gram, 'str', self.delimiter_words)

def get_tag_word_from_wordtag(key, delimiter):
    splitted_key = key.split(delimiter)

    return splitted_key[0], splitted_key[1]

class ClaimDatabase:
    def __init__(self, path_dir_database, path_raw_data, claim_data_set, K = None):
        # description: create a database in which the 
        self.path_dir_database = path_dir_database
        self.path_raw_data = path_raw_data
        self.claim_data_set = claim_data_set
        self.K = K
        
        self.path_dir_database_claims = os.path.join(self.path_dir_database, 'claims_' + str(self.claim_data_set)) #'claims_' + str(self.claim_data_set) + '_' + str(K))
        self.path_raw_claims = os.path.join(path_raw_data, str(self.claim_data_set) + '.jsonl')
        self.path_settings = os.path.join(self.path_dir_database_claims, 'settings.json')
        
        print('claim database')
        if not os.path.isdir(self.path_dir_database_claims):
            print('- create claim database')
            os.makedirs(self.path_dir_database_claims)
            self.create_database()
        else:
            print('- claim database already exists')
        
        if os.path.isfile(self.path_settings):
            self.settings = dict_load_json(self.path_settings)
            self.nr_claims = self.settings['nr_claims']
            self.scoring_flag = self.settings['scoring_flag']
        else:
            raise ValueError('settings file should exist')
        
    def create_database(self):
        list_claim_dicts = load_jsonl(self.path_raw_claims)
        self.nr_claims = len(list_claim_dicts)
        for id in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_dir_database_claims, str(id) + '.json')
            dict_claim_id = list_claim_dicts[id]
            dict_claim_id['verifiable'] = unicodedata.normalize('NFD', normalise_text(dict_claim_id['verifiable']))
            dict_claim_id['claim'] = unicodedata.normalize('NFD', normalise_text(dict_claim_id['claim']))
            for interpreter in range(len(dict_claim_id['evidence'])):
                for proof in range(len(dict_claim_id['evidence'][interpreter])):
                    if dict_claim_id['evidence'][interpreter][proof][2] != None:
                        dict_claim_id['evidence'][interpreter][proof][2] = unicodedata.normalize('NFD', normalise_text(dict_claim_id['evidence'][interpreter][proof][2]))
            
            dict_save_json(dict_claim_id, path_claim)
        
        if os.path.isfile(self.path_settings):
            raise ValueError('settings file should not exist', self.path_settings)
            # settings = dict_load_json(self.path_settings)
        else:
            settings = {}
            settings['nr_claims'] = None
            settings['scoring_flag'] = config.FALSE
        
        settings['nr_claims'] = self.nr_claims
        dict_save_json(settings, self.path_settings)
    
    def get_claim_from_id(self, id):
        path_claim = os.path.join(self.path_dir_database_claims, str(id) + '.json')
        dict_claim_id = dict_load_json(path_claim)
        return dict_claim_id

    def write_claim_2_db(self, id, dict_claim_id):
        path_claim = os.path.join(self.path_dir_database_claims, str(id) + '.json')
        with HiddenPrints():
            dict_save_json(dict_claim_id, path_claim)

    def add_score(self, score, method, id):
        results_key_name = 'results'
        dict_claim = self.get_claim_from_id(id)

        if results_key_name not in dict_claim:
            dict_claim[results_key_name] = {}

        dict_claim[results_key_name][method] = score

        self.write_claim_2_db(id, dict_claim)    

    def set_scoring_flag(self, scoring_flag):
        print('convert scoring flag from', self.settings['scoring_flag'], ' to ', scoring_flag)
        if scoring_flag == self.settings['scoring_flag']:
            print('no need to save, because the flags are the same')
        elif scoring_flag in [config.FALSE, config.TRUE]:
            self.settings['scoring_flag'] = scoring_flag
            with HiddenPrints():
                dict_save_json(self.settings, self.path_settings)

        else:
            raise ValueError('scoring flag is not True or False', scoring_flag)