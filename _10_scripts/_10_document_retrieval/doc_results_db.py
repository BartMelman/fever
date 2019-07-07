import os
import shutil
import json
from sqlitedict import SqliteDict

from utils_db import load_jsonl
from vocabulary import VocabularySqlite
from tfidf_database import TFIDFDatabaseSqlite
from wiki_database import WikiDatabaseSqlite
# from text_database import TextDatabaseSqlite
from utils_doc_results import Claim, ClaimDocTokenizer

import config

def get_vocab_tf_idf_from_exp(experiment_nr):
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

def get_empty_dict_claim():
    empty_dict = {}
    empty_dict['nr_words_title'] = None
    empty_dict['nr_words_per_pos'] = None
    empty_dict['tokenize'] = {}
    empty_dict['tokenize']['matches_per_pos'] = get_empty_tag_dict()
    empty_dict['tokenize']['tf_idf'] = get_empty_tag_dict()
    empty_dict['tokenize']['raw_count_idf'] = get_empty_tag_dict()
    empty_dict['tokenize']['idf'] = get_empty_tag_dict()
#     empty_dict['tokenize']['raw_count_sum_per_pos'] = get_empty_tag_dict()
#     empty_dict['tokenize']['max_sum_idf'] = 0
    empty_dict['tokenize_lemma'] = {}
    empty_dict['tokenize_lemma']['matches_per_pos'] = get_empty_tag_dict()
    empty_dict['tokenize_lemma']['tf_idf_sum_per_pos'] = get_empty_tag_dict()
    empty_dict['tokenize_lemma']['raw_count_idf_sum_per_pos'] = get_empty_tag_dict()
    empty_dict['tokenize_lemma']['idf_sum_per_pos'] = get_empty_tag_dict()
    empty_dict['tokenize_lemma']['raw_count_sum_per_pos'] = get_empty_tag_dict()
    empty_dict['tokenize_lemma']['max_sum_idf'] = 0
    return empty_dict
       
def get_empty_tag_dict():
    tag_2_id_dict = get_tag_2_id_dict()
    empty_tag_dict = {}
    for pos_id in range(len(tag_2_id_dict)):
        empty_tag_dict[pos_id] = 0
    return empty_tag_dict

def get_tag_2_id_dict():
    tag_2_id_dict = {}
    tag_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']

    for i in range(len(tag_list)):
        tag = tag_list[i]
        tag_2_id_dict[tag] = i
    return tag_2_id_dict

def get_tf_idf_name(experiment_nr):
    if experiment_nr in [31,34, 37]:
        return 'tf_idf'
    elif experiment_nr in [32,35, 38]:
        return 'raw_count_idf'
    elif experiment_nr in [33,36]:
        return 'idf'
    else:
        raise ValueError('experiment_nr not in selection', experiment_nr)

if __name__ == '__main__':

    claim_data_set_names = ['dev']
    claim_data_set = claim_data_set_names[0]
    path_dev_set = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR, claim_data_set + ".jsonl")
    results = load_jsonl(path_dev_set)
    results
    i = 0
    claim = Claim(results[i])


    tag_2_id_dict = get_tag_2_id_dict()

    # === process claim === #
    experiment_nr = 37

    with HiddenPrints():
        vocab, tf_idf_db = get_vocab_tf_idf_from_exp(experiment_nr)

    doc = vocab.wiki_database.nlp(claim.claim)
    claim_doc_tokenizer = ClaimDocTokenizer(doc, vocab.delimiter_words)
    n_grams, nr_words = claim_doc_tokenizer.get_n_grams(vocab.method_tokenization, tf_idf_db.vocab.n_gram)

    claim_dict = {}

    claim_dict['claim'] = {}
    claim_dict['claim']['nr_words'] = sum(n_grams.values())
    claim_dict['claim']['nr_words_per_pos'] = get_empty_tag_dict()
    claim_dict['title'] = {}
    claim_dict['title']['ids'] = {}
    tag_list = []
    for key, count in n_grams.items():
        tag, word = get_tag_word_from_wordtag(key, vocab.delimiter_tag_word)
        tag_list.append(tag)
        pos_id = tag_2_id_dict[tag]
        claim_dict['claim']['nr_words_per_pos'][pos_id] += count

    # === process titles === #
    experiment_nr = 31

    with HiddenPrints():
        vocab, tf_idf_db = get_vocab_tf_idf_from_exp(experiment_nr)

    doc = vocab.wiki_database.nlp(claim.claim)
    claim_doc_tokenizer = ClaimDocTokenizer(doc, vocab.delimiter_words)
    n_grams, nr_words = claim_doc_tokenizer.get_n_grams(vocab.method_tokenization, tf_idf_db.vocab.n_gram)

    path_title_ids = tf_idf_db.path_ids_dict
    path_title_tf_idf = tf_idf_db.path_tf_idf_dict
    mydict_ids = SqliteDict(path_title_ids)
    mydict_tf_idf = SqliteDict(path_title_tf_idf)

    idx = 0
    for key, count in n_grams.items():
        tag = tag_list[idx]

        dictionary = get_dict_from_n_gram([key], mydict_ids, mydict_tf_idf, tf_idf_db)

        tf_idf_name = get_tf_idf_name(experiment_nr)
        tag_nr = tag_2_id_dict[tag]

        for id, tf_idf_value in dictionary.items():
            if id in claim_dict['title']['ids'].keys():
                claim_dict['title']['ids'][id][vocab.method_tokenization[0]][tf_idf_name][tag_nr] += tf_idf_value
            else:
                claim_dict['title']['ids'][id] = get_empty_dict_claim()
                claim_dict['title']['ids'][id][vocab.method_tokenization[0]][tf_idf_name][tag_nr] += tf_idf_value

        idx += 1





    



# === claim ttile 