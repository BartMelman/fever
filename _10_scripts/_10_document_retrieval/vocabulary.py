import nltk
from nltk import FreqDist
from nltk.util import ngrams
import math
import os
from tqdm import tqdm
from tqdm import tnrange
from sqlitedict import SqliteDict
import spacy

from text_database import TextDatabase, Text
from utils_db import dict_save_json, dict_load_json
from _10_scripts._01_database.wiki_database import WikiDatabase

import config

class Vocabulary:
    """A sample Employee class"""
    def __init__(self, path_wiki_database, table_name_wiki, n_gram, method_tokenization, tags_in_db_flag, source, tag_list_selected):
        self.delimiter_title = '_'
        self.delimiter_text = ' '
        self.nlp = spacy.load('en', disable=["parser", "ner"])

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
        self.delimiter_words = '\q'
        self.tags_in_db_flag = tags_in_db_flag
        self.tag_list_selected = tag_list_selected

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
        self.path_settings = os.path.join(self.base_dir, 'settings.json')
        self.nr_words = None
        
        if os.path.isfile(self.path_settings):
            print('Load existing settings file')
            self.settings = dict_load_json(self.path_settings)
        else:
            self.settings = {}

        # if os.path.isfile(self.path_word_count):
        #     print('Word count count dictionary already exists')
        # else:
        #     self.get_word_count()
        
        # print('count total number of words')
        # with SqliteDict(self.path_word_count) as dict_word_count:
        #     # self.nr_words = sum(list(dict_word_count.values()))
        #     self.nr_words = len(dict_word_count)
        
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
        method_options = ['tokenize', 'tokenize_lemma', 'tokenize_lemma_list_accepted', 'tokenize_lemma_nouns', 
        'tokenize_lemma_prop_nouns', 'tokenize_lemma_number', 'tokenize_lemma_pos']
        folder_name = 'vocab'
        folder_name = folder_name + '_' + self.source + '_' + str(self.n_gram)

        for method in self.method_tokenization:
            if method == 'tokenize':
                folder_name = folder_name + '_t'
            elif method == 'tokenize_lemma':
                folder_name = folder_name + '_tl'
            elif method == 'tokenize_lemma_list_accepted':
                folder_name = folder_name + '_tlla'
            elif method == 'tokenize_lemma_nouns':
                folder_name = folder_name + '_tln'
            elif method == 'tokenize_lemma_prop_nouns':
                folder_name = folder_name + '_tlpn'
            elif method == 'tokenize_lemma_number':
                folder_name = folder_name + '_tln'
            elif method == 'tokenize_lemma_pos':
                folder_name = folder_name + '_lp'
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
        batch_size = 3500000
        list_exception_tokens = [' ', '', ',', '.']

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
                            if stop_word_in_key(key) == False:
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
        batch_size_pipeline = 100000
        batch_size_sqlite =  100000
        list_pos_tokenization = ['tokenize_lemma_pos', 'tokenize_text_pos', 'tokenize_lower_pos']
        # list_exception_tokens = ['is', 'a', 'of', ',', 'and', '.', 'in', 'by', 'their', 'has', 'been', 'to', 'from', 'the', 'that', 'as', "'s", 'are', 'for', 'who', '', 'it', 'known', 'its', 'was', 'first', 'with', 'be', 'but', 'an', 'on', '-lrb-', '-rrb-', 'which', 'district', ':', '``', "''", 'one', 'national', 'united', 'states', 'at', 'this', 'county', ';', 'may', 'new', 'his', 'american', 'she', 'born', 'film', 'he', 'also', 'or', 'were', '--', 'two', 'had', 'after']
        # list_exception_tokens = [' ', '', ',', '.']

        if os.path.isfile(self.path_document_count):
            print('Document count dictionary already exists')
        else:     
            nr_n_grams = 0 # nr of different n-grams
            n_gramfdist = FreqDist()
            text_list = []

            for id_nr in tqdm(range(1, self.text_database.nr_rows+1), desc='document count'):
                # iterate through id_list
                if self.source == 'text':
                    text = self.text_database.wiki_database.get_text_from_id(id_nr)
                elif self.source == 'title':
                    text = self.text_database.wiki_database.get_title_from_id(id_nr)
                    text = text.replace(self.delimiter_title, self.delimiter_text)
                else:
                    raise ValueError('source not in options', self.source)
                if text != '':
                    text_list.append(text)

                # iterate through 
                if (id_nr%batch_size_pipeline == 0) or (id_nr == self.nr_wiki_pages):
                    for doc in tqdm(self.nlp.pipe(iter_phrases(text_list)), desc='pipeline', total = len(text_list)):
                        text_class = Text(doc)
                        tokenized_text = text_class.process(self.method_tokenization)

                        n_gram_text = ngrams(tokenized_text, self.n_gram)
                        n_gram_text_unique = unique_values(sorted(n_gram_text))

                        n_gramfdist.update(n_gram_text_unique)
                    text_list = []

                if (id_nr%batch_size_sqlite == 0) or (id_nr == self.nr_wiki_pages):
                    with SqliteDict(self.path_document_count) as dict_document_count:
                        for key, value in tqdm(n_gramfdist.items(), desc='document count dictionary'):
                            if self.n_gram == 1:
                                if self.method_tokenization in list_pos_tokenization:
                                    tag = key.split(self.delimiter_tag_word)[0]
                                    word = key.split(self.delimiter_tag_word)[1]
                                    if self.tags_in_db_flag == True:
                                        if stop_word_in_key(word) == False:
                                            if key in dict_document_count:
                                                dict_document_count[key] = dict_document_count[key] + value
                                            else:
                                                dict_document_count[key] = value
                                                nr_n_grams += 1
                                    else:
                                        raise ValueError('use lemma instead')
                                else:
                                    if stop_word_in_key(key) == False:
                                        word = self.delimiter_words.join(key)
                                        if word in dict_document_count:
                                            dict_document_count[word] = dict_document_count[word] + value
                                        else:
                                            dict_document_count[word] = value
                                            nr_n_grams += 1
                            elif self.n_gram == 2:
                                if self.method_tokenization in list_pos_tokenization: 
                                    if self.tags_in_db_flag == True:
                                        phrase = key.split(self.delimiter_words)
                                        tag1 = phrase[0].split(self.delimiter_tag_word)[0]
                                        tag2 = phrase[1].split(self.delimiter_tag_word)[0]
                                        word1 = phrase[0].split(self.delimiter_tag_word)[1]
                                        word2 = phrase[1].split(self.delimiter_tag_word)[1]

                                        if (tag1 in self.tag_list_selected) and (tag2 in self.tag_list_selected):
                                            word = self.delimiter_words.join([word1, word2])
                                            if word in dict_document_count:
                                                dict_document_count[word] = dict_document_count[word] + value
                                            else:
                                                dict_document_count[word] = value
                                                nr_n_grams += 1
                                    else:
                                        raise ValueError('no code for tag_selected == False')
                                else:
                                    raise ValueError('code not designed for no pos for bigrams')
                            else:
                                raise ValueError('code not designed for n_gram > 2', self.n_gram)
                        dict_document_count.commit()
                    n_gramfdist = FreqDist()

            self.settings['nr_n_grams'] = nr_n_grams
            dict_save_json(self.settings, self.path_settings)

def stop_word_in_key(key):
    list_exception_tokens = ['is', 'a', 'of', ',', 'and', '.', 'in', 'by', 'their', 'has', 'been', 'to', 'from', 'the', 'that', 'as', "'s", 'are', 'for', 'who', '', 'it', 'known', 'its', 'was', 'first', 'with', 'be', 'but', 'an', 'on', '-lrb-', '-rrb-', 'which', 'district', ':', '``', "''", 'one', 'national', 'united', 'states', 'at', 'this', 'county', ';', 'may', 'new', 'his', 'american', 'she', 'born', 'film', 'he', 'also', 'or', 'were', '--', 'two', 'had', 'after']
    flag_exception_token = False
    for exception_token in list_exception_tokens:
        if exception_token in key:
            flag_exception_token = True
            break
    return flag_exception_token

def iter_phrases(text_list):
    for text in text_list:
        yield text


def unique_values(iterable):
    it = iter(iterable)
    previous = next(it)
    yield previous
    for item in it:
        if item != previous:
            previous = item
            yield item

def count_n_grams(tokenized_text, n_gram, output_format, separator_words):
    # description: 
    # input
    # - tokenized_text: tokenized and splitted text
    #  - n_gram: n_gram [int]
    # output
    # - [dict] : dictionary of counts for n_grams
    
    output_format_options = ['tuple','str']
    # separator = ' '

    n_gramfdist = FreqDist()
    n_gramfdist.update(ngrams(tokenized_text, n_gram))
    dictionary = dict(n_gramfdist)

    if output_format not in output_format_options:
        raise ValueError('output_format not in output_format_options', output_format, output_format_options)
    
    if output_format == 'str':
        new_dictionary = {}
        for key in dictionary.keys():
            if stop_word_in_key(key) == False:
                new_dictionary[separator.join(key)] = dictionary[key]
        dictionary = new_dictionary
    
    nr_words = sum(list(dictionary.values()))
#     dictionary['nr_words_document'] = nr_words
    return dictionary, nr_words

