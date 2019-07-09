import os
from tqdm import tqdm
import json
import unicodedata

from utils_db import dict_save_json, dict_load_json, load_jsonl, HiddenPrints
from vocabulary import VocabularySqlite
from tfidf_database import TFIDFDatabaseSqlite
from wiki_database import WikiDatabaseSqlite

from utils_doc_results import Claim, ClaimDocTokenizer

import config

def get_tag_word_from_wordtag(key, delimiter):
    phrase = key.split(delimiter)
    tag = phrase[0]
    word = phrase[1]
    return tag, word

def get_tag_dict(claim_data_set, n_gram, path_tags, wiki_database):

    if os.path.isfile(path_tags):
        print('tags file already exists')
        dictionary_tags = dict_load_json(path_tags)
    else:
        
        path_dev_set = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR, claim_data_set + ".jsonl")
        results = load_jsonl(path_dev_set)

        tag_2_id_dict = get_tag_2_id_dict()
        if n_gram == 1:
            experiment_nr = 37
        else:
            raise ValueError('train model with tags for n_grams = 2')

        with HiddenPrints():
            vocab, tf_idf_db = get_vocab_tf_idf_from_exp(experiment_nr, wiki_database)


        dictionary_tags = {}
        for claim_nr in tqdm(range(len(results)), desc = 'tags claims'):
            claim = Claim(results[claim_nr])

            doc = vocab.wiki_database.nlp(claim.claim)
            claim_doc_tokenizer = ClaimDocTokenizer(doc, vocab.delimiter_words)
            n_grams, nr_words = claim_doc_tokenizer.get_n_grams(vocab.method_tokenization, tf_idf_db.vocab.n_gram)
            tag_list = []
            for key, count in n_grams.items():
                tag, word = get_tag_word_from_wordtag(key, vocab.delimiter_tag_word)
                tag_list.append(tag)

            dictionary_tags[str(claim_nr)] = tag_list
        dict_save_json(dictionary_tags, path_tags)
    return dictionary_tags

def get_empty_tag_dict():
    tag_2_id_dict = get_tag_2_id_dict()
    empty_tag_dict = {}
    for pos_id in range(len(tag_2_id_dict)):
        empty_tag_dict[str(pos_id)] = 0
    return empty_tag_dict

def get_tag_2_id_dict():
    tag_2_id_dict = {}
    tag_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']

    for i in range(len(tag_list)):
        tag = tag_list[i]
        tag_2_id_dict[tag] = i
    return tag_2_id_dict

def get_vocab_tf_idf_from_exp(experiment_nr, wiki_database):
    file_name = 'experiment_%.2d.json'%(experiment_nr)
    path_experiment = os.path.join(config.ROOT, config.CONFIG_DIR, file_name)

    with open(path_experiment) as json_data_file:
        data = json.load(json_data_file)

    vocab = VocabularySqlite(wiki_database = wiki_database, n_gram = data['n_gram'],
        method_tokenization = data['method_tokenization'], tags_in_db_flag = data['tags_in_db_flag'], 
        source = data['vocabulary_source'], tag_list_selected = data['tag_list_selected'])

    tf_idf_db = TFIDFDatabaseSqlite(vocabulary = vocab, method_tf = data['method_tf'], method_df = data['method_df'],
        delimiter = data['delimiter'], threshold = data['threshold'], source = data['tf_idf_source'])
    return vocab, tf_idf_db

def get_tf_idf_name(experiment_nr):
    if experiment_nr in [31,34, 37]:
        return 'tf_idf'
    elif experiment_nr in [32,35, 38]:
        return 'raw_count_idf'
    elif experiment_nr in [33,36]:
        return 'idf'
    else:
        raise ValueError('experiment_nr not in selection', experiment_nr)

def get_dict_from_n_gram(n_gram_list, mydict_ids, mydict_tf_idf, tf_idf_db):
    
    dictionary = {}

    for word in n_gram_list:
        try:
            word_id_list = mydict_ids[word].split(tf_idf_db.delimiter)[1:]
            word_tf_idf_list = mydict_tf_idf[word].split(tf_idf_db.delimiter)[1:]
        except KeyError:
            print('KeyError', word)
            word_id_list = []
            word_tf_idf_list = []
        for j in range(len(word_id_list)):
            id = int(word_id_list[j])       
            tf_idf = float(word_tf_idf_list[j])
            try:
                dictionary[id] = dictionary[id] + tf_idf
            except KeyError:
                dictionary[id] = tf_idf
    return dictionary

class ClaimDatabase:
    def __init__(self, path_dir_database, path_raw_data, claim_data_set):
        self.path_dir_database = path_dir_database
        self.path_raw_data = path_raw_data
        self.claim_data_set = claim_data_set
        
        self.path_dir_database_claims = os.path.join(self.path_dir_database, 'claims_' + str(self.claim_data_set))
        self.path_raw_claims = os.path.join(path_raw_data, str(self.claim_data_set) + '.jsonl')
        self.path_settings = os.path.join(path_dir_database_claims, 'settings.json')
        
        if not os.path.isdir(self.path_dir_database_claims):
            print('create claim database')
            os.makedirs(self.path_dir_database_claims)
            self.create_database()
        else:
            print('claim database already exists')
        
        if os.path.isfile(self.path_settings):
            self.settings = dict_load_json(self.path_settings)
            self.nr_claims = settings['nr_claims']
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
            settings = dict_load_json(self.path_settings)
            
        else:
            settings = {}
        
        settings['nr_claims'] = self.nr_claims
        dict_save_json(settings, self.path_settings)
    
    def get_claim_from_id(self, id):
        path_claim = os.path.join(self.path_dir_database_claims, str(id) + '.json')
        dict_claim_id = dict_load_json(path_claim)
        return dict_claim_id