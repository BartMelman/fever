import os, sys
from tqdm import tqdm
from sqlitedict import SqliteDict

from wiki_database import WikiDatabaseSqlite, Text
from doc_results_db_utils import get_empty_tag_dict, get_tag_2_id_dict, get_tag_dict, get_vocab_tf_idf_from_exp, get_tf_idf_name, get_vocab_tf_idf_from_exp
from doc_results_db_utils import get_dict_from_n_gram
from utils_db import dict_load_json, dict_save_json, HiddenPrints, load_jsonl
from utils_doc_results import Claim, ClaimDocTokenizer, get_tag_word_from_wordtag

import config




# def get_empty_dict_claim():
#     empty_dict = {}
#     empty_dict['nr_words_title'] = None
#     empty_dict['nr_words_per_pos'] = None
#     empty_dict['tokenize'] = {}
#     empty_dict['tokenize']['matches_per_pos'] = get_empty_tag_dict()
#     empty_dict['tokenize']['tf_idf'] = get_empty_tag_dict()
#     empty_dict['tokenize']['raw_count_idf'] = get_empty_tag_dict()
#     empty_dict['tokenize']['idf'] = get_empty_tag_dict()
# #     empty_dict['tokenize']['raw_count_sum_per_pos'] = get_empty_tag_dict()
# #     empty_dict['tokenize']['max_sum_idf'] = 0
#     empty_dict['tokenize_lemma'] = {}
#     empty_dict['tokenize_lemma']['matches_per_pos'] = get_empty_tag_dict()
#     empty_dict['tokenize_lemma']['tf_idf_sum_per_pos'] = get_empty_tag_dict()
#     empty_dict['tokenize_lemma']['raw_count_idf_sum_per_pos'] = get_empty_tag_dict()
#     empty_dict['tokenize_lemma']['idf_sum_per_pos'] = get_empty_tag_dict()
#     empty_dict['tokenize_lemma']['raw_count_sum_per_pos'] = get_empty_tag_dict()
#     empty_dict['tokenize_lemma']['max_sum_idf'] = 0
#     return empty_dict


class ClaimFile:
    """A sample Employee class"""
    def __init__(self, id, path_dir_files):
        self.path_claim = os.path.join(path_dir_files, str(id) + '.json')
        if os.path.isfile(self.path_claim):
            self.claim_dict = dict_load_json(self.path_claim)
        else:
            self.claim_dict = {}
            self.claim_dict['claim'] = {}
            self.claim_dict['claim']['1_gram'] = {}
            self.claim_dict['claim']['1_gram']['nr_words'] = None
            self.claim_dict['claim']['1_gram']['nr_words_per_pos'] = get_empty_tag_dict()
            self.claim_dict['title'] = {}
            self.claim_dict['title']['1_gram'] = {}
#             self.claim_dict['title']['1_gram']['nr_words'] = None
#             self.claim_dict['title']['1_gram']['nr_words_per_pos'] = get_empty_tag_dict()
#             self.claim_dict['title']['ids'] = {}
            self.save_claim()

    def process_claim(self, claim):
        self.claim_dict['claim']['text'] = claim
        self.save_claim()

    def process_tags(self, tag_list, n_gram):
#         self.claim_dict['claim'][str(n_gram) +'_gram'] = {}
        if n_gram == 1:
            self.claim_dict['claim'][str(n_gram) +'_gram']['tag_list'] = tag_list
        else:
            raise ValueError('written for n_gram == 1')
        self.save_claim()
    
    def process_tf_idf_experiment(self, tag_2_id_dict, tf_idf_db, mydict_ids, mydict_tf_idf):
        tf_idf_name = get_tf_idf_name(experiment_nr)
        if tf_idf_db.n_gram == 1:
            doc = tf_idf_db.vocab.wiki_database.nlp(self.claim_dict['claim']['text'])
            
            tag_list = [word.pos_ for word in doc]        
#             claim_doc_tokenizer = ClaimDocTokenizer(doc, tf_idf_db.vocab.delimiter_words)
#             n_grams_dict, nr_words = claim_doc_tokenizer.get_n_grams(tf_idf_db.vocab.method_tokenization, tf_idf_db.vocab.n_gram)
            # === write tf-idf values === #
            claim_text = Text(doc)
            tokenized_claim_list = claim_text.process(tf_idf_db.vocab.method_tokenization)
#             print(tag_list, tokenized_claim_list)
            idx = 0
            for i in range(len(tag_list)):
                tag = tag_list[i]
                word = tokenized_claim_list[i]
            
                pos_id = tag_2_id_dict[tag]
                
                with HiddenPrints():
                    dictionary = get_dict_from_n_gram([word], mydict_ids, mydict_tf_idf, tf_idf_db)
#                 print(len(dictionary))
                if len(dictionary) < 2000:
                    for id, tf_idf_value in dictionary.items():
                        # === create dictionary if does not exist === #
                        if str(id) not in self.claim_dict['title']['1_gram']:
                            self.claim_dict['title']['1_gram'][str(id)] = {}
                            title = tf_idf_db.vocab.wiki_database.get_title_from_id(id)

                            doc = tf_idf_db.vocab.wiki_database.nlp(title)
                            claim_doc_tokenizer = ClaimDocTokenizer(doc, tf_idf_db.vocab.delimiter_words)
                            n_grams_dict_title, nr_words_title = claim_doc_tokenizer.get_n_grams(tf_idf_db.vocab.method_tokenization, tf_idf_db.vocab.n_gram)

                            self.claim_dict['title']['1_gram'][str(id)]['nr_words'] = nr_words_title

                        if vocab.method_tokenization[0] not in self.claim_dict['title']['1_gram'][str(id)].keys():
                            self.claim_dict['title']['1_gram'][str(id)][vocab.method_tokenization[0]] = {}
                            if tf_idf_name not in self.claim_dict['title']['1_gram'][str(id)][vocab.method_tokenization[0]].keys():
                                self.claim_dict['title']['1_gram'][str(id)][vocab.method_tokenization[0]][tf_idf_name] = get_empty_tag_dict()

                        self.claim_dict['title']['1_gram'][str(id)][vocab.method_tokenization[0]][tf_idf_name][str(pos_id)] += tf_idf_value   
                idx += 1
            self.save_claim()
        else:
            raise ValueError('Adapt function for bigrams')
        
        self.save_claim()
    
    def process_nr_words_per_pos(self, tf_idf_db, tag_2_id_dict):
        if tf_idf_db.n_gram == 1:
            doc = tf_idf_db.vocab.wiki_database.nlp(self.claim_dict['claim']['text'])
            claim_doc_tokenizer = ClaimDocTokenizer(doc, tf_idf_db.vocab.delimiter_words)
            n_grams_dict, nr_words = claim_doc_tokenizer.get_n_grams(tf_idf_db.vocab.method_tokenization, tf_idf_db.vocab.n_gram)

            self.claim_dict['claim']['1_gram']['nr_words'] = sum(n_grams_dict.values())
            
            for key, count in n_grams_dict.items():
                tag, word = get_tag_word_from_wordtag(key, vocab.delimiter_tag_word)
                pos_id = tag_2_id_dict[tag]
                self.claim_dict['claim']['1_gram']['nr_words_per_pos'][str(pos_id)] += count
            self.save_claim()
        else:
            raise ValueError('Adapt function for bigrams')

    def save_claim(self):
        with HiddenPrints():
            dict_save_json(self.claim_dict, self.path_claim)

if __name__ == '__main__':

    # === variables === #
    n_gram = 1
    claim_data_set = 'dev'
    folder_name_score_combination = 'score_combination'

    path_dir_results = os.path.join(config.ROOT, config.RESULTS_DIR, folder_name_score_combination)
    path_tags = os.path.join(path_dir_results, 'tags_' + claim_data_set + '_n_gram_' + str(n_gram) + '.json')
    # path_dir_files = os.path.join(path_dir_results, claim_data_set)
    path_dir_claims = os.path.join(path_dir_results, 'claims_' + claim_data_set)
    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
    path_claim_data_set = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR, claim_data_set + ".jsonl")

    try:
        os.makedirs(path_dir_results, exist_ok=True)
    except FileExistsError:
        print('folder already exists:', path_dir_results)

    try:
        os.makedirs(path_dir_claims, exist_ok=True)
    except FileExistsError:
        print('folder already exists:', path_dir_claim)
    
    results = load_jsonl(path_claim_data_set)
    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)
    tag_2_id_dict = get_tag_2_id_dict()

    tag_dict = get_tag_dict(claim_data_set, n_gram, path_tags, wiki_database)
    nr_claims = 1000 # len(results)

    print('claim database: insert claim\'s text and claim\'s tag_list')

    for str_id, tag_list in tqdm(tag_dict.items(), total = len(tag_dict), desc = 'tag'):
        id = int(str_id)
        if id < nr_claims:
            file = ClaimFile(id = id, path_dir_files = path_dir_claims)
            file.process_tags(tag_list, n_gram)
            claim = Claim(results[id])
            file.process_claim(claim.claim)

    print('claim database: insert nr words per tag for claim')

    experiment_nr = 37
    with HiddenPrints():
        vocab, tf_idf_db = get_vocab_tf_idf_from_exp(experiment_nr, wiki_database)
        
    for id in tqdm(range(nr_claims), desc = 'nr words per pos'):
        file = ClaimFile(id = id, path_dir_files = path_dir_claims)
        file.process_nr_words_per_pos(tf_idf_db, tag_2_id_dict)

    print('claim database: ')
    # === selected ids === #

    # experiment_nr = 31

    # print('load tf_idf nr_words_pos')
    # with HiddenPrints():
    #     vocab, tf_idf_db = get_vocab_tf_idf_from_exp(experiment_nr, wiki_database)

    # mydict_ids = SqliteDict(tf_idf_db.path_ids_dict)
    # mydict_tf_idf = SqliteDict(tf_idf_db.path_tf_idf_dict)
    
