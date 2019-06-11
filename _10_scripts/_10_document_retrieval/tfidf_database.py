import os
from sqlitedict import SqliteDict
import sqlite3
from tqdm import tqdm
from tqdm import tnrange
import shutil
import math

from document import Document

from utils_db import dict_save_json, dict_load_json
# from tf_idf import count_n_grams
from dictionary_batch_idf import DictionaryBatchIDF
from vocabulary import Vocabulary, count_n_grams
from _10_scripts._01_database.wiki_database import WikiDatabase



class TFIDFDatabase:
    """A sample Employee class"""
    def __init__(self, vocabulary, method_tf, method_df, delimiter, threshold, source):
        self.default_first_value = 'F'
        self.vocab = vocabulary
        self.delimiter = delimiter
        self.batch_size = 50000
        # self.get_base_dir()
        self.method_tf = method_tf
        self.method_df = method_df
        self.threshold = threshold
        self.nr_wiki_pages = self.vocab.nr_wiki_pages
        self.source = source 

        self.path_threshold_folder = None
        self.base_dir = None
        self.path_vocabulary_selected_dict = None
        self.path_tf_idf_dict_empty = None
        self.path_ids_dict_empty = None
        self.path_tf_idf_dict = None
        self.path_ids_dict = None
        self.get_paths()
        
        
        if os.path.isfile(self.path_vocabulary_selected_dict):
            print('selected vocabulary dictionary already exists')
        else:
            self.get_vocabulary_selected_dictionary()

        if not(os.path.isfile(self.path_tf_idf_dict_empty) and os.path.isfile(self.path_ids_dict_empty)):
            print('construct empty database')
            self.construct_empty_database()
        else:
            print('empty database already exists')
            
        if not(os.path.isfile(self.path_tf_idf_dict) and os.path.isfile(self.path_ids_dict)):
            print('construct database')
            self.fill_database() 
        else:
            print('database already filled')
            
    def get_paths(self):       
        base_dir = self.vocab.base_dir
        threshold_folder_name = 'thr_' + str(self.threshold) + '_'
        folder_name = 'ex_' + self.method_tf + '_' + self.method_df + '_' + self.source
        
        self.path_threshold_folder = os.path.join(base_dir, threshold_folder_name)
        self.base_dir = os.path.join(base_dir, threshold_folder_name, folder_name)
        self.path_vocabulary_selected_dict = os.path.join(base_dir, threshold_folder_name, 'vocabulary_selected.sqlite')
        
        self.path_tf_idf_dict_empty = os.path.join(self.base_dir, 'tf_idf_dict_empty.sqlite')
        self.path_ids_dict_empty = os.path.join(self.base_dir, 'ids_dict_empty.sqlite')
        self.path_tf_idf_dict = os.path.join(self.base_dir, 'tf_idf_dict.sqlite')
        self.path_ids_dict = os.path.join(self.base_dir, 'ids_dict.sqlite')
        
        try:
            os.makedirs(self.path_threshold_folder)    
        except FileExistsError:
            print("Directory " , self.path_threshold_folder ,  " already exists") 
        
        try:
            os.makedirs(self.base_dir)    
        except FileExistsError:
            print("Directory " , self.base_dir ,  " already exists") 
            
    def get_vocabulary_selected_dictionary(self):
        
        with SqliteDict(self.vocab.path_document_count) as document_count_dict:
            with SqliteDict(self.path_vocabulary_selected_dict) as vocabulary_selected_dict:
                for word, value in tqdm(document_count_dict.items(), desc='vocabulary selected', total = self.vocab.nr_words):
                    if (value/float(self.nr_wiki_pages)) < self.threshold:
                        vocabulary_selected_dict[word] = 1
                    else: 
                        vocabulary_selected_dict[word] = 0
                    
                vocabulary_selected_dict.commit()                
        
    def construct_empty_database(self):
        # description: construct the database with as rows all the possible words. We only enter the 'F' as first element.
                
        with SqliteDict(self.path_tf_idf_dict_empty) as mydict_tf_idf:
            with SqliteDict(self.path_ids_dict_empty) as mydict_ids:
                with SqliteDict(self.vocab.path_document_count) as document_count_dict:
                    for key, value in tqdm(document_count_dict.items(), desc='empty database', total = self.vocab.nr_words):
                        mydict_tf_idf[key] = self.default_first_value
                        mydict_ids[key] = self.default_first_value
                mydict_ids.commit()
            mydict_tf_idf.commit()
                
    def fill_database(self):
        
        shutil.copyfile(self.path_tf_idf_dict_empty, self.path_tf_idf_dict)  
        shutil.copyfile(self.path_ids_dict_empty, self.path_ids_dict)
        
        scorer = TFIDF(self.method_tf, self.method_df)

        batch_dictionary = DictionaryBatchIDF(self.delimiter)
        
        nr_wiki_pages = self.vocab.nr_wiki_pages
        method_tokenization = self.vocab.method_tokenization
        n_gram = self.vocab.n_gram
        
        with SqliteDict(self.vocab.path_document_count) as dict_document_count:
            with SqliteDict(self.path_vocabulary_selected_dict) as vocabulary_selected_dict:
                for i in tqdm(range(nr_wiki_pages), desc='fill database'):
                    id_wiki_page = i + 1
        #             title = vocab.text_database.wiki_database.get_title_from_id(id_nr)
        #             id_wiki_page = vocab.title_2_id_dict[title]
                    if self.source == 'text':
                        tokenized_text = self.vocab.text_database.get_tokenized_text_from_id(id_wiki_page, method_tokenization)
                    elif self.source == 'title':
                        tokenized_text = self.vocab.text_database.get_tokenized_title_from_id(id_wiki_page, method_tokenization)
                    else:
                        raise ValueError('source not in options', self.source)

                    tf_dict, nr_words_doc = count_n_grams(tokenized_text, n_gram, 'str')
                    
                    total_count_doc = self.vocab.nr_wiki_pages
                    total_count_tf = nr_words_doc
                    
                    for word in tf_dict.keys():
                        count_tf = tf_dict[word]
                        count_doc = dict_document_count[word]
                        
                        tf_idf_value  = scorer.get_tf_idf(count_tf, total_count_tf, count_doc, total_count_doc)

                        batch_dictionary.update(word, id_wiki_page, tf_idf_value)

                    if (i%self.batch_size == 0) or (i == self.vocab.nr_wiki_pages - 1):
                        # === write to table === #
                        with SqliteDict(self.path_tf_idf_dict) as mydict_tf_idf:
                            with SqliteDict(self.path_ids_dict) as mydict_ids:
                                list_keys = list(batch_dictionary.dictionary_tf_idf.keys())
                                for i in tqdm(range(len(list_keys)), desc='batch'):
                                    word = list_keys[i] 
                                    if vocabulary_selected_dict[word] == 1:
                                        tf_idf_value = batch_dictionary.dictionary_tf_idf[word]['values']
                                        id_word = batch_dictionary.dictionary_tf_idf[word]['ids']
                                        mydict_tf_idf[word] = mydict_tf_idf[word] + self.delimiter + tf_idf_value
                                        mydict_ids[word] = mydict_ids[word] + self.delimiter + id_word
                                mydict_ids.commit()
                            mydict_tf_idf.commit()
                        # === reset dictionary === #
                        batch_dictionary.reset()

class TFIDF(object):
    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    def __init__(self, method_tf, method_idf):
        self.method_tf = method_tf
        self.method_idf = method_idf
        
        method_tf_list = ['raw_count','term_frequency','constant_one']
        method_idf_list = ['inverse_document_frequency', 'constant_one']
        
        if self.method_tf not in method_tf_list:
            raise ValueError('method_tf not in method_tf_list', self.method_tf, method_tf_list)
            
        if self.method_idf not in method_idf_list:
            raise ValueError('method_tf not in method_tf_list', self.method_idf, method_idf_list)
            
    def get_tf_idf(self, count_tf, total_count_tf, count_doc, total_count_doc):
        """Dispatch method"""
        self.count_tf = count_tf
        self.total_count_tf = total_count_tf
        self.count_doc = count_doc
        self.total_count_doc = total_count_doc
        
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, self.method_tf, lambda: "Invalid tf method")
        tf = method()
        method = getattr(self, self.method_idf, lambda: "Invalid idf method")
        idf = method()
        
        tf_idf = tf * idf
        return tf_idf

    def constant_one(self):
        return 1.0

    def raw_count(self):
        return self.count_tf
 
    def term_frequency(self):
        return float(self.count_tf)/self.total_count_tf
 
    def inverse_document_frequency(self):
        if self.count_doc == 0:
            idf = 0
#             raise ValueError('word does not exist in texts', method_tf, method_tf_list)
        else:
            idf = math.log10(float(self.total_count_doc) / self.count_doc)
        
        return idf
