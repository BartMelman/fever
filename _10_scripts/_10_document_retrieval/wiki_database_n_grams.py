import os
import spacy
from tqdm import tqdm

from utils_db import get_file_name_from_variable_list, mkdir_if_not_exist, create_path_dictionary
from template import Settings
from database import Database
from wiki_database import WikiDatabaseSqlite

import config


class WikiDatabaseNgrams:

    def __init__(self, path_dir_database, wiki_database, method_tokenization, n_gram, delimiter_option=False):
        # === save input(s) ===#
        self.wiki_database = wiki_database
        self.method_tokenization = method_tokenization
        self.n_gram = n_gram
        #         self.output_type = output_type
        self.delimiter_option = delimiter_option
        self.path_dir_database = os.path.join(path_dir_database, 'WikiNgram')

        # === variables === #

        # === process === #
        print('WikiDatabaseNgrams')

        mkdir_if_not_exist(self.path_dir_database)

        self.settings = Settings(self.path_dir_database)

        self.nlp = spacy.load('en', disable=["parser", "ner"])

        method_tokenization_list = ['tokenize', 'tag'] # ['tokenize', 'lemma', 'tag', 'lower']
        n_gram_list = [1, 2]
        delimiter_options_list = [True, False]
        doc_type_list = ['lines', 'title', 'text'] # ['lines', 'title', 'text']

        # === create database === #
        self.flag_function_call(function_name='create_database', arg_list=[method_tokenization_list,
                                                                           n_gram_list,
                                                                           delimiter_options_list,
                                                                           doc_type_list])

        # self.create_database(method_tokenization_list,
        #                      n_gram_list,
        #                      delimiter_options_list,
        #                      doc_type_list)

        self.id_2_title_db = Database(path_database_dir=self.path_dir_database,
                                      database_name=get_database_name_from_options(doc_type='title',
                                                                                   method_tokenization=self.method_tokenization,
                                                                                   n_gram=self.n_gram,
                                                                                   delimiter_option=self.delimiter_option),
                                      database_method='lsm',
                                      input_type='int',
                                      output_type='list_str',
                                      checks_flag=True)
        self.id_2_text_db = Database(path_database_dir=self.path_dir_database,
                                     database_name=get_database_name_from_options(doc_type='text',
                                                                                  method_tokenization=self.method_tokenization,
                                                                                  n_gram=self.n_gram,
                                                                                  delimiter_option=self.delimiter_option),
                                     database_method='lsm',
                                     input_type='int',
                                     output_type='list_str',
                                     checks_flag=True)
        self.id_2_lines_db = Database(path_database_dir=self.path_dir_database,
                                      database_name=get_database_name_from_options(doc_type='lines',
                                                                                   method_tokenization=self.method_tokenization,
                                                                                   n_gram=self.n_gram,
                                                                                   delimiter_option=self.delimiter_option),
                                      database_method='lsm',
                                      input_type='int',
                                      output_type='list_str',
                                      checks_flag=True)

        print('***finished***')

    def get_item(self, input_type, input_value, output_type):
        # description:
        # input:
        # - input_type: options: 'id'
        # - input_value: value that is the key
        # - output_type: 'title', 'text', 'lines'

        if input_type == 'id':
            if output_type == 'title':
                return self.id_2_title_db.get_item(input_value)
            elif output_type == 'text':
                return self.id_2_text_db.get_item(input_value)
            elif output_type == 'lines':
                return self.id_2_lines_db.get_item(input_value)
            else:
                raise ValueError('output_type not in options', output_type)
        else:
            raise ValueError('input_type not in options', input_type)

    def create_database(self, method_tokenization_list, n_gram_list, delimiter_options_list, doc_type_list):
        # description :
        # input :
        # -
        # output :
        # -

        database_dict = {}

        batch_size = 50000

        experiment_settings_list = []
        for method_tokenization in method_tokenization_list:
            for n_gram in n_gram_list:
                if n_gram == 1:
                    delimiter_option = True
                    experiment_settings_list.append([method_tokenization, n_gram, delimiter_option])
                else:
                    for delimiter_option in delimiter_options_list:
                        experiment_settings_list.append([method_tokenization, n_gram, delimiter_option])

        # === create databases === #
        for doc_type in doc_type_list:
            for experiment in experiment_settings_list:
                method_tokenization, n_gram, delimiter_option = experiment
                database_dict = create_path_dictionary(
                    [doc_type, method_tokenization, n_gram, delimiter_option],
                    database_dict)
                database_dict[doc_type][method_tokenization][n_gram][delimiter_option] = Database(
                    path_database_dir=self.path_dir_database,
                    database_name=get_database_name_from_options(doc_type=doc_type,
                                                                 method_tokenization=method_tokenization,
                                                                 n_gram=n_gram,
                                                                 delimiter_option=delimiter_option),
                    database_method='lsm',
                    input_type='int',
                    output_type='list_str',
                    checks_flag=True)

        for doc_type in doc_type_list:
            if doc_type in ['title', 'text']:
                text_list = []
                id_list = []

                for doc_nr in tqdm(range(self.wiki_database.nr_wikipedia_pages), desc='n_gram_wiki_database_' + doc_type):
                    text = self.wiki_database.get_item(input_type='id', input_value=doc_nr, output_type=doc_type)
                    text_list.append(text)
                    id_list.append(doc_nr)

                    if doc_nr % batch_size == 0 or doc_nr == self.wiki_database.nr_wikipedia_pages - 1:
                        idx = 0
                        for doc in tqdm(self.nlp.pipe(iter_phrases(text_list)), desc='pipeline', total=len(text_list)):
                            text_class = Text(doc)

                            doc_nr_batch = id_list[idx]

                            for experiment in experiment_settings_list:
                                method_tokenization_tmp, n_gram_tmp, delimiter_option_tmp = experiment

                                tokenized_text = text_class.process(method_tokenization=method_tokenization_tmp,
                                                                    n_gram=n_gram_tmp,
                                                                    delimiter_flag=delimiter_option_tmp)
                                print('store', doc_nr_batch, tokenized_text)
                                database_dict[doc_type][method_tokenization_tmp][n_gram_tmp][
                                    delimiter_option_tmp].store_item(key=doc_nr_batch,
                                                                 value=tokenized_text)

                            idx += 1
                        text_list = []
                        id_list = []

            elif doc_type == 'lines':

                # === empty dictionary === #
                empty_list_dictionary = {}
                for experiment in experiment_settings_list:
                    method_tokenization_tmp, n_gram_tmp, delimiter_option_tmp = experiment
                    empty_list_dictionary = create_path_dictionary(
                        [method_tokenization_tmp, n_gram_tmp, delimiter_option_tmp],
                        empty_list_dictionary)
                    empty_list_dictionary[method_tokenization_tmp][n_gram_tmp][delimiter_option_tmp] = []

                text_list = []
                doc_id_list = []
                last_element_list = []

                for doc_nr in tqdm(range(self.wiki_database.nr_wikipedia_pages), desc='n_gram_wiki_database_' + doc_type):
                    text_wiki_list = self.wiki_database.get_item(input_type='id', input_value=doc_nr, output_type=doc_type)
                    for i in range(len(text_wiki_list)):
                        line = text_wiki_list[i]
                        if i == len(text_wiki_list) - 1:
                            last_element_list.append('last_line')
                        else:
                            last_element_list.append('not_last_line')
                        text_list.append(line)
                        doc_id_list.append(doc_nr)

                    if doc_nr % batch_size == 0 or doc_nr == self.wiki_database.nr_wikipedia_pages - 1:
                        idx = 0
                        dict_list_lines = empty_list_dictionary

                        for doc in tqdm(self.nlp.pipe(iter_phrases(text_list)), desc='pipeline', total=len(text_list)):
                            text_class = Text(doc)

                            for experiment in experiment_settings_list:
                                method_tokenization, n_gram, delimiter_option = experiment
                                tokenized_text = text_class.process(method_tokenization=method_tokenization,
                                                                    n_gram=n_gram,
                                                                    delimiter_flag=delimiter_option)
                                dict_list_lines[method_tokenization][n_gram][delimiter_option].append(
                                    tokenized_text)

                            if last_element_list[idx] == 'last_line':
                                doc_idx = doc_id_list[idx]
                                for experiment in experiment_settings_list:
                                    method_tokenization, n_gram, delimiter_option = experiment
                                    tokenized_text = dict_list_lines[method_tokenization][n_gram][
                                        delimiter_option]
                                    database_dict[doc_type][method_tokenization][n_gram][
                                        delimiter_option].store_item(key=doc_idx,
                                                                     value=tokenized_text)
                                    dict_list_lines[method_tokenization][n_gram][delimiter_option] = []

                                dict_list_lines = empty_list_dictionary

                            idx += 1

                        text_list = []
                        doc_id_list = []
                        last_element_list = []
            else:
                raise ValueError('doc_type not in options', doc_type)

    # === recurrent functions == #
    def flag_function_call(self, function_name, arg_list):
        check_flag = self.settings.check_function_flag(function_name, 'check')

        if check_flag == 'finished_correctly':
            return True

        elif check_flag == 'not_started_yet':
            self.settings.check_function_flag(function_name, 'start')

            values = getattr(self, function_name)(*arg_list)

            self.settings.check_function_flag(function_name, 'finish')
        else:
            raise ValueError('check_flag not in options', check_flag)

class Text:
    """A sample Employee class"""

    def __init__(self, text, delimiter=config.DelimiterWords):
        self.delimiter = delimiter
        self.text = text

    def process(self, method_tokenization, n_gram, delimiter_flag):
        """Dispatch method"""
        method_options = ['tokenize', 'lower', 'lemma', 'tag']

        if method_tokenization not in method_options:
            raise ValueError('method not in method_options', method_tokenization, method_options)

        unigram_text = getattr(self, method_tokenization)()

        n_gram_list = n_gram_from_unigram(unigram_list=unigram_text,
                                          n_gram=n_gram,
                                          delimiter_flag=delimiter_flag,
                                          delimiter=self.delimiter)

        return n_gram_list

    def tag(self):
        return [word.pos_ for word in self.text]

    def tokenize(self):
        return [word.text for word in self.text]

    def lower(self):
        return [(word.text).lower() for word in self.text]

    def lemma(self):
        return [word.lemma_ for word in self.text]


def n_gram_from_unigram(unigram_list, n_gram, delimiter_flag=False, delimiter='/k'):
    if n_gram < 1 or type(n_gram) is not int:
        raise ValueError('n_gram is too small or no int', n_gram)

    if delimiter_flag not in [True, False]:
        raise ValueError('delimiter_flag not in options', delimiter_flag)

    n_gram_list = []

    for i in range(len(unigram_list) - n_gram + 1):
        n_gram_list.append(unigram_list[i:i + n_gram])

    if delimiter_flag:
        return [delimiter.join(element) for element in n_gram_list]
    else:
        return n_gram_list


def iter_phrases(text_list):
    for text in text_list:
        yield text

def get_database_name_from_options(doc_type, method_tokenization, n_gram, delimiter_option):
    if n_gram == 1:
        return get_file_name_from_variable_list(['db',
                                                  doc_type,
                                                  method_tokenization,
                                                  str(n_gram)])
    else:
        return get_file_name_from_variable_list(['db',
                                                 doc_type,
                                                 method_tokenization,
                                                 str(n_gram),
                                                 delimiter_option])

if __name__ == '__main__':
    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)

    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)

    method_tokenization = 'tokenize'
    n_gram = 1
    delimiter_option = True
    wiki_database.nr_wikipedia_pages = 200
    wiki_database_n_grams = WikiDatabaseNgrams(path_dir_database=path_wiki_database_dir,
                                                wiki_database=wiki_database,
                                                method_tokenization=method_tokenization,
                                                n_gram=n_gram,
                                                delimiter_option=delimiter_option)