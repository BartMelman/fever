import nltk
from nltk import FreqDist
from nltk.util import ngrams
import math
import os
from tqdm import tqdm
from tqdm import tnrange
from sqlitedict import SqliteDict

from text_database import TextDatabase
from utils_db import dict_save_json, dict_load_json
from _10_scripts._01_database.wiki_database import WikiDatabase

import config

class Vocabulary:
    """A sample Employee class"""
    def __init__(self, path_wiki_database, table_name_wiki, n_gram, method_tokenization, source):
        self.text_database = TextDatabase(path_wiki_database, table_name_wiki)
        self.nr_wiki_pages = self.text_database.nr_rows
        self.vocabulary_dict = None
        self.word_count = None
        self.document_count = None
        self.tf_idf_value_list = None
        self.tf_idf_id_list = None
        self.title_2_id_dict = None 
        self.id_2_title_dict = None 
        self.n_gram = n_gram
        self.method_tokenization = method_tokenization 
        
        source_options = ['title','text']
        if source not in source_options:
            raise ValueError('source not in source_options', source, source_options)
        self.source = source

        self.base_dir = None
        self.get_base_dir()
        
        self.path_title_2_id = os.path.join(self.base_dir, 'title_2_id.json')
        self.path_id_2_title = os.path.join(self.base_dir, 'id_2_title.json')
        self.path_vocabulary = os.path.join(self.base_dir, 'vocabulary.json')
        self.path_word_count = os.path.join(self.base_dir, 'word_count.sqlite')
        self.path_document_count = os.path.join(self.base_dir, 'document_count.sqlite')
        self.path_vocabulary_selected = os.path.join(self.base_dir, 'vocabulary_selected.json')
        self.nr_words = None
        
        if os.path.isfile(self.path_word_count):
            print('Word count count dictionary already exists')
        else:
            self.get_word_count()
        
        print('count total number of words')
        with SqliteDict(self.path_word_count) as dict_word_count:
            # self.nr_words = sum(list(dict_word_count.values()))
            self.nr_words = len(dict_word_count)
        
        if os.path.isfile(self.path_document_count):
            print('Document count dictionary already exists')
        else:
            self.get_document_count()
              
#         if os.path.isfile(self.path_vocabulary):
#             print('Load vocabulary dictionary')
#             self.vocabulary_dict = dict_load_json(self.path_vocabulary)
#         else:
#             print('Construct vocabulary dictionary')
#             self.vocabulary_dict = {}
#             for i, word in enumerate(self.document_count.keys()):
#                 self.vocabulary_dict[word] = i
#             dict_save_json(self.vocabulary_dict, self.path_vocabulary)
        
        if os.path.isfile(self.path_title_2_id) and os.path.isfile(self.path_id_2_title):
            print('Load title_2_id and id_2_title dictionaries')
            self.title_2_id_dict = dict_load_json(self.path_title_2_id)
            self.id_2_title_dict = dict_load_json(self.path_id_2_title)
        else:
            self.get_title_dictionary()
        
    def get_base_dir(self):
        method_options = ['tokenize', 'remove_space', 'remove_bracket_and_word_between', 'make_lower_case', 'lemmatization', 'lemmatization_get_nouns']
        folder_name = 'vocab'
        folder_name = folder_name + '_' + self.source + '_' + str(self.n_gram)

        for method in self.method_tokenization:
            if method == 'tokenize':
                folder_name = folder_name + '_t'
            elif method == 'remove_space':
                folder_name = folder_name + '_rs'
            elif method == 'remove_bracket_and_word_between':
                folder_name = folder_name + '_rmb'
            elif method == 'make_lower_case':
                folder_name = folder_name + '_mlc'
            elif method == 'lemmatization':
                folder_name = folder_name + '_lm'
            elif method == 'lemmatization_get_nouns':
                folder_name = folder_name + '_lmgn'
            else:
                raise ValueError('method not in method_options', method, method_options)
        
        self.base_dir = os.path.join(config.ROOT, config.RESULTS_DIR, folder_name) 
        print(self.base_dir)
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except FileExistsError:
            print('folder already exists:', self.base_dir)
        
    def get_title_dictionary(self):
        if os.path.isfile(self.path_title_2_id) and os.path.isfile(self.path_id_2_title):
            print('Load title dictionary')
            self.title_2_id_dict = dict_load_json(self.path_title_2_id)
            self.id_2_title_dict = dict_load_json(self.path_id_2_title)
        else:
            self.title_2_id_dict  = {}
            self.id_2_title_dict  = {}
            
            for i in tqdm(range(self.nr_wiki_pages), desc='title-id-dictionary'):
                id_nr = i + 1
#                 tokenized_text = vocab.text_database.get_tokenized_text_from_id(id_nr)
                title = self.text_database.wiki_database.get_title_from_id(id_nr)
                self.title_2_id_dict[title] = id_nr
                self.id_2_title_dict[id_nr] = title
            
            dict_save_json(self.title_2_id_dict, self.path_title_2_id)
            dict_save_json(self.id_2_title_dict, self.path_id_2_title)
            
#     def get_list_titles(self):

    def get_word_count(self):
        # description: 
        # input
        # - tokenized_text: tokenized and splitted text
        #  - n_gram: n_gram [int]
        # output
        # - [dict] : dictionary of counts for n_grams
        separator = ' '
        batch_size = 1000000

        if os.path.isfile(self.path_word_count):
            print('Word count dictionary already exists')
        else:  
            n_gramfdist = FreqDist()

            for id_nr in tqdm(range(1, self.text_database.nr_rows+1), desc='word count'): #
                if self.source == 'text':
                    tokenized_text = self.text_database.get_tokenized_text_from_id(id_nr, self.method_tokenization)
                elif self.source == 'title':
                    tokenized_text = self.text_database.get_tokenized_title_from_id(id_nr, self.method_tokenization)
                else:
                    raise ValueError('source not in options', self.source)
                
                n_gramfdist.update(ngrams(tokenized_text, self.n_gram))
                
                if (id_nr%batch_size==0) or (id_nr == self.text_database.nr_rows):
                    with SqliteDict(self.path_word_count) as dict_words:
                        for key, value in tqdm(n_gramfdist.items(), desc='word count dictionary'):
                            word = ' '.join(key)
                            if word in dict_words:
                                dict_words[word] = dict_words[word] + value
                            else:
                                dict_words[word] = value
                        dict_words.commit()
                    n_gramfdist = FreqDist()
            
        return self.nr_words
    
    def get_document_count(self):
        # description: 
        # input
        # - tokenized_text: tokenized and splitted text
        #  - n_gram: n_gram [int]
        # output
        # - [dict] : dictionary of counts for n_grams
        
        separator = ' '
        batch_size = 1000000
        if os.path.isfile(self.path_document_count):
            print('Document count dictionary already exists')
        else:     
            n_gramfdist = FreqDist()
            for id_nr in tqdm(range(1, self.text_database.nr_rows+1), desc='document count'):
                tokenized_text = self.text_database.get_tokenized_text_from_id(id_nr, self.method_tokenization)
                tokenized_text_reduced = list(set(tokenized_text))
                n_gramfdist.update(ngrams(tokenized_text_reduced, self.n_gram))

                if (id_nr%batch_size==0) or (id_nr == self.text_database.nr_rows):
                    with SqliteDict(self.path_document_count) as dict_document_count:
                        for key, value in tqdm(n_gramfdist.items(), desc='document count dictionary'):
                            word = ' '.join(key)
                            if word in dict_document_count:
                                dict_document_count[word] = dict_document_count[word] + value
                            else:
                                dict_document_count[' '.join(key)] = value
                        dict_document_count.commit()
                    n_gramfdist = FreqDist()
                 
def count_n_grams(tokenized_text, n_gram, output_format):
    # description: 
    # input
    # - tokenized_text: tokenized and splitted text
    #  - n_gram: n_gram [int]
    # output
    # - [dict] : dictionary of counts for n_grams
    
    output_format_options = ['tuple','str']
    separator = ' '
    n_gramfdist = FreqDist()
    n_gramfdist.update(ngrams(tokenized_text, n_gram))
    dictionary = dict(n_gramfdist)
    if output_format not in output_format_options:
        raise ValueError('output_format not in output_format_options', output_format, output_format_options)
    
    if output_format == 'str':
        new_dictionary = {}
        for key in dictionary.keys():
            new_dictionary[' '.join(key)] = dictionary[key]
        dictionary = new_dictionary
    
    nr_words = sum(list(dictionary.values()))
#     dictionary['nr_words_document'] = nr_words
    return dictionary, nr_words

