import os, sys
from tqdm import tqdm
from sqlitedict import SqliteDict
import numpy as np
import torch
import unicodedata

from wiki_database import WikiDatabaseSqlite, Text
from utils_doc_results_db import get_empty_tag_dict, get_tag_2_id_dict_unigrams, get_tag_2_id_dict_bigrams, get_tag_dict, get_tf_idf_from_exp, get_tf_idf_name, get_vocab_tf_idf_from_exp
from utils_doc_results_db import get_dict_from_n_gram, get_list_properties, get_value_if_exists, label_2_num
from utils_db import dict_load_json, dict_save_json, HiddenPrints, load_jsonl, mkdir_if_not_exist
from utils_doc_results import Claim, ClaimDocTokenizer, get_tag_word_from_wordtag, ClaimDatabase
from doc_results import PerformanceTFIDF

import config

class ClaimFile:
    """A sample Employee class"""
    def __init__(self, id, path_dir_files):
        # description: dictionary of a claim id with all the information from the different experiments
        # input:
        # - id : claim id number
        # - path_dir_files : directory at which it needs to be saved

        self.path_claim = os.path.join(path_dir_files, str(id) + '.json')
        if os.path.isfile(self.path_claim):
            self.claim_dict = dict_load_json(self.path_claim)
        else:
            self.claim_dict = {}
            self.claim_dict['claim'] = {}
            self.claim_dict['claim']['1_gram'] = {}
            self.claim_dict['claim']['1_gram']['nr_words'] = None
            self.claim_dict['claim']['1_gram']['nr_words_per_pos'] = get_empty_tag_dict(n_gram = 1)
            self.claim_dict['title'] = {}
            self.claim_dict['title']['1_gram'] = {}
            self.claim_dict['text'] = {}
            self.claim_dict['text']['1_gram'] = {}
            self.save_claim()
    
    def process_claims_selected(self, claim_dictionary, wiki_database):
        # description: add ids to selected dictionary which are the proof
        # input: 
        # - claim_dictionary: dictionary of the claim
        claim = Claim(claim_dictionary)
        if 'ids_selected' not in self.claim_dict:
            interpreter_list = claim.evidence
            id_list = []
            for interpreter in interpreter_list:
                for proof in interpreter:
                    title = proof[2]
                    if title is not None:
                        id = wiki_database.get_id_from_title(title)
                        id_list.append(id)
            self.claim_dict['ids_correct_docs'] = id_list
            self.claim_dict['ids_selected'] = id_list
        
        # === add from selected_ids in claim_dictionary === #    

        if 'docs_selected' in claim_dictionary:
            self.claim_dict['ids_selected'] += claim_dictionary['docs_selected']

        # === save === #
        self.save_claim()
        
    def process_claim(self, claim):
        # description: adds text and label of claim to claim_dict
        # input
        # - claim: claim [class Claim]

        self.claim_dict['claim']['text'] = claim.claim
        self.claim_dict['claim']['label'] = claim.label
        self.save_claim()

    def process_tags(self, tag_list, n_gram):
        # description: add tags of unigrams and bigrams to claim_dict
        # input:
        # - n_gram: n-gram [int]
        # - tag_list: list of tags [list of str]

        if n_gram == 1:
            self.claim_dict['claim'][str(n_gram) +'_gram']['tag_list'] = tag_list
        else:
            raise ValueError('written for n_gram == 1')
        self.save_claim()
    
    def process_tf_idf_experiment(self, experiment_nr, tf_idf_db, mydict_ids, mydict_tf_idf):
        # description:
        # input:
        # - tf_idf_db
        # - mydict_ids : key: word, value: list of document ids [Sqlite database]
        # - mydict_tf_idf: key: word, value: list of document tf_idfs [Sqlite database]

        tf_idf_name = get_tf_idf_name(experiment_nr)
        source = tf_idf_db.source # 'text', 'title'
        method_tokenization = tf_idf_db.vocab.method_tokenization[0]

        if tf_idf_db.n_gram == 1:
            tag_2_id_dict = get_tag_2_id_dict_unigrams()
        elif tf_idf_db.n_gram == 2:
            tag_2_id_dict = get_tag_2_id_dict_bigrams(tf_idf_db.vocab.tag_list_selected)
        else:
            raise ValueError('Code is only written for unigrams and bigrams', tf_idf_db.n_gram)

        if tf_idf_db.n_gram == 1:
            doc = tf_idf_db.vocab.wiki_database.nlp(self.claim_dict['claim']['text'])

            tag_list = [word.pos_ for word in doc]      
            nr_words_claim = len(tag_list)

            claim_text = Text(doc)
            tokenized_claim_list = claim_text.process(tf_idf_db.vocab.method_tokenization)
            
            for i in range(nr_words_claim):
                tag = tag_list[i]
                word = tokenized_claim_list[i]
            
                pos_id = tag_2_id_dict[tag]
                
                with HiddenPrints():
                    dictionary = get_dict_from_n_gram([word], mydict_ids, mydict_tf_idf, tf_idf_db)

                for id, tf_idf_value in dictionary.items():
                    # only save the tf idf of ids in the selected id list
                    if id in self.claim_dict['ids_selected']:
                        # save number of words claim/title/text and total tf idf
                        if str(id) not in self.claim_dict[source]['1_gram']:
                            self.claim_dict[source]['1_gram'][str(id)] = {}

                            if source == 'title':
                                text = tf_idf_db.vocab.wiki_database.get_title_from_id(id)
                            elif source == 'text':
                                text = tf_idf_db.vocab.wiki_database.get_text_from_id(id)
                            else:
                                raise ValueError('source not in options', source)

                            doc = tf_idf_db.vocab.wiki_database.nlp(text)
                            claim_doc_tokenizer = ClaimDocTokenizer(doc, tf_idf_db.vocab.delimiter_words)
                            _, nr_words_text_source = claim_doc_tokenizer.get_n_grams(tf_idf_db.vocab.method_tokenization, tf_idf_db.vocab.n_gram)

                            self.claim_dict[source]['1_gram'][str(id)]['nr_words'] = nr_words_text_source

                        # create empty tag dictionary for method for id if does not exist
                        if method_tokenization not in self.claim_dict[source]['1_gram'][str(id)].keys():
                            self.claim_dict[source]['1_gram'][str(id)][method_tokenization] = {}
                            if tf_idf_name not in self.claim_dict[source]['1_gram'][str(id)][method_tokenization].keys():
                                self.claim_dict[source]['1_gram'][str(id)][method_tokenization][tf_idf_name] = get_empty_tag_dict(n_gram = 1)

                        # enter total tf_idf if not in dictionary 
                        if 'total_tf_idf' not in self.claim_dict[source]['1_gram'][str(id)][method_tokenization][tf_idf_name]:
                            self.claim_dict[source]['1_gram'][str(id)][method_tokenization][tf_idf_name]['total_tf_idf'] = tf_idf_db.id_2_total_tf_idf[str(id)]

                        # save tf_idf value 
                        self.claim_dict[source]['1_gram'][str(id)][method_tokenization][tf_idf_name][str(pos_id)] += tf_idf_value   
        
        elif tf_idf_db.n_gram == 2:
            doc = tf_idf_db.vocab.wiki_database.nlp(self.claim_dict['claim']['text'])

            tag_list = [word.pos_ for word in doc]      
            nr_words_claim = len(tag_list)

            claim_text = Text(doc)
            tokenized_claim_list = claim_text.process(tf_idf_db.vocab.method_tokenization)
            
            for i in range(nr_words_claim-1):
                tag_1 = tag_list[i]
                tag_2 = tag_list[i+1]

                word1 = tokenized_claim_list[i]
                word2 = tokenized_claim_list[i+1]

                pos_id = tag_2_id_dict[tag1 + tag2]
                
                word = tf_idf_db.vocab.delimiter_words.join([word1, word2])

                with HiddenPrints():
                    dictionary = get_dict_from_n_gram([word], mydict_ids, mydict_tf_idf, tf_idf_db)

                for id, tf_idf_value in dictionary.items():
                    # only save the tf idf of ids in the selected id list
                    if id in self.claim_dict['ids_selected']:
                        # create dictionary if id not in dictionary
                        if str(id) not in self.claim_dict[source]['2_gram']:
                            self.claim_dict[source]['2_gram'][str(id)] = {}

                        # create empty tag dictionary for method for id if does not exist
                        if method_tokenization not in self.claim_dict[source]['2_gram'][str(id)].keys():
                            self.claim_dict[source]['2_gram'][str(id)][method_tokenization] = {}
                            if tf_idf_name not in self.claim_dict[source]['2_gram'][str(id)][method_tokenization].keys():
                                self.claim_dict[source]['2_gram'][str(id)][method_tokenization][tf_idf_name] = get_empty_tag_dict(n_gram = 2)

                        # enter total tf_idf if not in dictionary 
                        if 'total_tf_idf' not in self.claim_dict[source]['2_gram'][str(id)][method_tokenization][tf_idf_name]:
                            self.claim_dict[source]['2_gram'][str(id)][method_tokenization][tf_idf_name]['total_tf_idf'] = tf_idf_db.id_2_total_tf_idf[str(id)]

                        # save tf_idf value 
                        self.claim_dict[source]['2_gram'][str(id)][method_tokenization][tf_idf_name][str(pos_id)] += tf_idf_value   
        
        else:
            raise ValueError('Function only written for unigrams and bigrams')
        
        self.save_claim()
    
    def process_nr_words_per_pos(self, tf_idf_db, tag_2_id_dict):
        if tf_idf_db.n_gram == 1:
            doc = tf_idf_db.vocab.wiki_database.nlp(self.claim_dict['claim']['text'])
            claim_doc_tokenizer = ClaimDocTokenizer(doc, tf_idf_db.vocab.delimiter_words)
            n_grams_dict, nr_words = claim_doc_tokenizer.get_n_grams(tf_idf_db.vocab.method_tokenization, tf_idf_db.vocab.n_gram)

            self.claim_dict['claim']['1_gram']['nr_words'] = sum(n_grams_dict.values())
            
            for key, count in n_grams_dict.items():
                tag, word = get_tag_word_from_wordtag(key, tf_idf_db.vocab.delimiter_tag_word)
                pos_id = tag_2_id_dict[tag]
                self.claim_dict['claim']['1_gram']['nr_words_per_pos'][str(pos_id)] += count
            self.save_claim()

        elif tf_idf_db.n_gram == 2:
            doc = tf_idf_db.vocab.wiki_database.nlp(self.claim_dict['claim']['text'])
            claim_doc_tokenizer = ClaimDocTokenizer(doc, tf_idf_db.vocab.delimiter_words)
            n_grams_dict, nr_words = claim_doc_tokenizer.get_n_grams(tf_idf_db.vocab.method_tokenization, tf_idf_db.vocab.n_gram)

            self.claim_dict['claim']['1_gram']['nr_words'] = sum(n_grams_dict.values())
            
            delimiter_position_tag = '\z'

            for key, count in n_grams_dict.items():
                splitted_key = key.split(tf_idf_db.delimiter_words)
                key1_splitted = splitted_key[0].split(delimiter_position_tag)
                key2_splitted = splitted_key[1].split(delimiter_position_tag)
                tag1 = splitted_key[0].split(delimiter_position_tag)[0]
                tag2 = splitted_key[1].split(delimiter_position_tag)[0]
                word1 = splitted_key[0].split(delimiter_position_tag)[1]
                word2 = splitted_key[1].split(delimiter_position_tag)[1]

                pos_id = tag_2_id_dict[tag1 + tag2]
                self.claim_dict['claim']['1_gram']['nr_words_per_pos'][str(pos_id)] += count
            self.save_claim()
            
        else:
            raise ValueError('Adapt function for bigrams')

    def save_claim(self):
        with HiddenPrints():
            dict_save_json(self.claim_dict, self.path_claim)

class ClaimTensorDatabase():
    def __init__(self, path_wiki_pages, path_wiki_database_dir, setup):
        # === process inputs === #
        self.claim_data_set = claim_data_set

        # === variables === #
        if setup == 1:
            self.claim_data_set = 'dev'
            self.experiment_list = [31, 37] # [31,32,33,34,35,36,37]
        elif setup == 2:
            self.claim_data_set = 'dev'
            self.experiment_list = [31, 37, 41]

        # === process === #
        self.path_dir_claim_database = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
        self.path_raw_claim_data = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR)
        self.path_results_dir = os.path.join(config.ROOT, config.RESULTS_DIR, config.SCORE_COMBINATION_DIR)
        self.path_setup_dir = os.path.join(self.path_results_dir, 'setup_' + str(setup))
        self.path_tags_unigram = os.path.join(self.path_setup_dir, 'tags_' + self.claim_data_set + '_n_gram_' + str(1) + '.json')
        self.path_claims_dir = os.path.join(self.path_setup_dir, 'claims')
        self.path_label_correct_evidence_false_dir = os.path.join(self.path_setup_dir, self.claim_data_set + '_correct_false_tensor')
        self.path_label_correct_evidence_true_dir = os.path.join(self.path_setup_dir, self.claim_data_set + '_correct_true_tensor')
        self.path_label_refuted_evidence_false_dir = os.path.join(self.path_setup_dir, self.claim_data_set + '_refuted_false_tensor')
        self.path_label_refuted_evidence_true_dir = os.path.join(self.path_setup_dir, self.claim_data_set + '_refuted_true_tensor')
        self.path_settings_dict = os.path.join(self.path_setup_dir, 'settings.json')
        self.tag_list_selected = ["INTJ", "NOUN", "NUM", "PROPN", "SYM", "X", "ADJ"]

        if not os.path.isdir(self.path_setup_dir):
            self.settings = {}
            mkdir_if_not_exist(self.path_setup_dir)
            mkdir_if_not_exist(self.path_claims_dir)
            mkdir_if_not_exist(self.path_label_correct_evidence_false_dir)
            mkdir_if_not_exist(self.path_label_correct_evidence_true_dir)
            mkdir_if_not_exist(self.path_label_refuted_evidence_false_dir)
            mkdir_if_not_exist(self.path_label_refuted_evidence_true_dir)
            self.get_results()
        else:
            self.settings = dict_load_json(self.path_settings_dict)
        

    def get_results(self):
        self.wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)
        self.claim_database = ClaimDatabase(path_dir_database = self.path_dir_claim_database, path_raw_data = self.path_raw_claim_data, claim_data_set = self.claim_data_set)

        self.claim_database.nr_claims = 100
        self.nr_claims = self.claim_database.nr_claims

        self.tag_2_id_unigram_dict = get_tag_2_id_dict_unigrams()
        self.tag_2_id_bigram_dict = get_tag_2_id_dict_bigrams(self.tag_list_selected)
        self.tag_dict_unigrams = get_tag_dict(self.claim_database, 1, self.path_tags_unigram, self.wiki_database)
        self.process_claim_tag_list()
        self.process_nr_words_per_tag(n_gram = 1)
        # self.process_nr_words_per_tag(n_gram = 2)
        self.process_claims_in_selection_list(experiment_nr = 37, K = 5, score_method = 'f_score', title_tf_idf_normalise_flag = False)
        self.process_selected_ids()
        self.save_2_tensor()


    def process_claims_in_selection_list(self, experiment_nr, K, score_method, title_tf_idf_normalise_flag):
        # description: add documents which are close to the claim to the dataset
        # input:
        # - K : [int]
        # - score_method : 
        # - title_tf_idf_normalise_flag : 

        performance_tf_idf = PerformanceTFIDF(self.wiki_database, experiment_nr, 
            self.claim_data_set, K, score_method, title_tf_idf_normalise_flag)

        for id in tqdm(range(self.nr_claims), total = self.nr_claims, desc = 'process_claims_selected'):
            if id < self.nr_claims:
                file = ClaimFile(id = id, path_dir_files = self.path_claims_dir)
                claim_dict = performance_tf_idf.claim_database.get_claim_from_id(id)
                file.process_claims_selected(claim_dict, self.wiki_database)

    def process_claim_tag_list(self):
        print('claim database: insert claim\'s text and claim\'s tag_list')

        n_gram = 1
        for str_id, tag_list in tqdm(self.tag_dict_unigrams.items(), total = len(self.tag_dict_unigrams), desc = 'tag'):
            id = int(str_id)
            if id < self.nr_claims:
                file = ClaimFile(id = id, path_dir_files = self.path_claims_dir)
                file.process_tags(tag_list, n_gram)
                claim_dict = self.claim_database.get_claim_from_id(id)
                claim = Claim(claim_dict)
                file.process_claim(claim)
                file.process_claims_selected(claim_dict, self.wiki_database)

    def process_nr_words_per_tag(self, n_gram):
        print('claim database: insert nr words per tag for claim')
        if n_gram == 1:
            experiment_nr = 37
            tag_2_id_dict = self.tag_2_id_unigram_dict
        elif n_gram == 2:
            experiment_nr = 41
            tag_2_id_dict = self.tag_2_id_bigram_dict
        else:
            raise ValueError('only written for unigrams and bigrams', n_gram)

        with HiddenPrints():
            tf_idf_db = get_tf_idf_from_exp(experiment_nr, self.wiki_database)
            
        for id in tqdm(range(self.nr_claims), desc = 'nr words per pos'):
            file = ClaimFile(id = id, path_dir_files = self.path_claims_dir)
            file.process_nr_words_per_pos(tf_idf_db, self.tag_2_id_unigram_dict)

    def process_selected_ids(self):
        print('claim database: insert selected ids')

        for experiment_nr in self.experiment_list:
            print('experiment:', experiment_nr)
            with HiddenPrints():
                tf_idf_db = get_tf_idf_from_exp(experiment_nr, self.wiki_database)

            mydict_ids = SqliteDict(tf_idf_db.path_ids_dict)
            mydict_tf_idf = SqliteDict(tf_idf_db.path_tf_idf_dict)

            for id in tqdm(range(self.nr_claims), desc = 'nr words per pos'):
                file = ClaimFile(id = id, path_dir_files = self.path_claims_dir)
                file.process_tf_idf_experiment(experiment_nr, tf_idf_db, mydict_ids, mydict_tf_idf)

    def save_2_tensor(self):
        print('claim database: save results to folder with tensors')

        settings_dict = {}

        id = 5
        file = ClaimFile(id = id, path_dir_files = self.path_claims_dir)

        id_list_title = list(file.claim_dict['title']['1_gram'].keys())
        id_list_text = list(file.claim_dict['text']['1_gram'].keys())

        observation_key_list_claim, _ = get_list_properties(file.claim_dict['claim']['1_gram'], [], [], [])

        if len(id_list_title) > 0:
            observation_key_list_title , _ = get_list_properties(file.claim_dict['title']['1_gram'][id_list_title[0]], [], [], []) 
        else:
            observation_key_list_title = []

        if len(id_list_text) > 0:
            observation_key_list_text,  _ = get_list_properties(file.claim_dict['text']['1_gram'][id_list_text[0]], [], [], [])
        else:
            observation_key_list_text = []

        settings_dict['observation_key_list_claim'] = observation_key_list_claim
        settings_dict['observation_key_list_title'] = observation_key_list_title
        settings_dict['observation_key_list_text'] = observation_key_list_text

        idx = 0

        nr_correct_false = 0
        nr_correct_true = 0
        nr_refuted_false = 0
        nr_refuted_true = 0

        for id in tqdm(range(self.claim_database.nr_claims), desc = 'save_2_tensor'):
            file = ClaimFile(id = id, path_dir_files = self.path_claims_dir)

            label = file.claim_dict['claim']['label']

            if label != 'NOT ENOUGH INFO':
                label_nr = label_2_num(label)

                for id_document in file.claim_dict['ids_selected']:

                    if label == 'SUPPORTS':
                        if id_document in file.claim_dict['ids_correct_docs']:
                            file_name_variables = os.path.join(self.path_label_correct_evidence_true_dir, 'variable_' + str(nr_correct_true) + '.pt')
                            file_name_label = os.path.join(self.path_label_correct_evidence_true_dir, 'label_' + str(nr_correct_true) + '.pt')
                            nr_correct_true += 1
                        else:
                            file_name_variables = os.path.join(self.path_label_correct_evidence_false_dir, 'variable_' + str(nr_correct_false) + '.pt')
                            file_name_label = os.path.join(self.path_label_correct_evidence_false_dir, 'label_' + str(nr_correct_false) + '.pt')
                            nr_correct_false += 1

                    elif label == 'REFUTES':
                        if id_document in file.claim_dict['ids_correct_docs']:
                            file_name_variables = os.path.join(self.path_label_refuted_evidence_true_dir, 'variable_' + str(nr_refuted_true) + '.pt')
                            file_name_label = os.path.join(self.path_label_refuted_evidence_true_dir, 'label_' + str(nr_refuted_true) + '.pt')
                            nr_refuted_true += 1
                        else:
                            file_name_variables = os.path.join(self.path_label_refuted_evidence_false_dir, 'variable_' + str(nr_refuted_false) + '.pt')
                            file_name_label = os.path.join(self.path_label_refuted_evidence_false_dir, 'label_' + str(nr_refuted_false) + '.pt')
                            nr_refuted_false += 1
                    else:
                        raise ValueError('label not correct', label)

                    list_variables = []

                    id_list = list(file.claim_dict['title']['1_gram'].keys())

                    _, values_claim = get_list_properties(file.claim_dict['claim']['1_gram'], [], [], [])
                    list_variables += values_claim

                    if id_document in file.claim_dict['title']['1_gram']:
                        _, values_title = get_list_properties(file.claim_dict['title']['1_gram'][id_document], [], [], [])
                        list_variables += values_title

                    if id_document in file.claim_dict['text']['1_gram']:
                        _, values_text  = get_list_properties(file.claim_dict['text']['1_gram'][id_document], [], [], [])
                        list_variables += values_text


                    numpy_array = np.array(list_variables)
                    tensor_variable = torch.from_numpy(numpy_array)      
                    tensor_label = torch.tensor([label_nr])

                    torch.save(tensor_variable, file_name_variables)
                    torch.save(tensor_label, file_name_label)
                    
                    idx += 1

        settings_dict['nr_total'] = idx
        settings_dict['nr_correct_false'] = nr_correct_false
        settings_dict['nr_correct_true'] = nr_correct_true
        settings_dict['nr_refuted_false'] = nr_refuted_false
        settings_dict['nr_refuted_true'] = nr_refuted_true

        settings_dict['nr_claims'] = self.claim_database.nr_claims

        dict_save_json(settings_dict, self.path_settings_dict)

if __name__ == '__main__':

    claim_data_set = 'dev'
    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)

    setup = 1

    ClaimTensorDatabase(path_wiki_pages, path_wiki_database_dir, setup)
    
    
