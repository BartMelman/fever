import os
import spacy
from tqdm import tqdm

from utils_db import mkdir_if_not_exist, create_path_dictionary
from template import Settings
from database import Database
from wiki_database_n_grams import get_database_name_from_options, iter_phrases, Text
from claim_database import ClaimDatabase
from wiki_database import WikiDatabaseSqlite

import config

class ClaimDatabaseNgrams:

    def __init__(self, path_dir_database, claim_database, method_tokenization, n_gram, delimiter_option=False):
        # === save input(s) ===#
        self.claim_database = claim_database
        self.method_tokenization = method_tokenization
        self.n_gram = n_gram
        #         self.output_type = output_type
        self.delimiter_option = delimiter_option
        self.path_dir_database = os.path.join(path_dir_database,
                                              'claim_database_n_gram_' + self.claim_database.claim_data_set)
        # === variables === #

        # === process === #
        print('ClaimDatabaseNgrams')

        mkdir_if_not_exist(self.path_dir_database)

        self.settings = Settings(self.path_dir_database)

        self.nlp = spacy.load('en', disable=["parser", "ner"])

        method_tokenization_list = ['tokenize', 'tag']  # ['tokenize', 'lemma', 'tag', 'lower']
        n_gram_list = [1]
        delimiter_options_list = [True, False]
        doc_type_list = ['claim']

        # === create database === #
        self.flag_function_call(function_name='create_database', arg_list=[method_tokenization_list,
                                                                           n_gram_list,
                                                                           delimiter_options_list,
                                                                           doc_type_list])

        self.id_2_claim_db = Database(path_database_dir=self.path_dir_database,
                                      database_name=get_database_name_from_options(doc_type='claim',
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
            if output_type == 'claim':
                return self.id_2_claim_db.get_item(input_value)
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

        batch_size = 500000

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
            if doc_type in ['claim']:
                text_list = []
                id_list = []

                for doc_nr in tqdm(range(claim_database.nr_claims), desc='n_gram_claim_database_' + doc_type):
                    text = claim_database.get_item(input_type='id', input_value=doc_nr, output_type=doc_type)
                    text_list.append(text)
                    id_list.append(doc_nr)

                    if doc_nr % batch_size == 0 or doc_nr == claim_database.nr_claims - 1:
                        idx = 0
                        for doc in tqdm(self.nlp.pipe(iter_phrases(text_list)), desc='pipeline', total=len(text_list)):
                            text_class = Text(doc)

                            doc_nr_batch = id_list[idx]

                            for experiment in experiment_settings_list:
                                method_tokenization, n_gram, delimiter_option = experiment

                                tokenized_text = text_class.process(method_tokenization=method_tokenization,
                                                                    n_gram=n_gram,
                                                                    delimiter_flag=delimiter_option)

                                database_dict[doc_type][method_tokenization][n_gram][
                                    delimiter_option].store_item(key=doc_nr_batch,
                                                                 value=tokenized_text)

                            idx += 1
                        text_list = []
                        id_list = []
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


if __name__ == '__main__':
    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)

    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)

    path_raw_data = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR)
    path_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
    claim_data_set = 'dev'
    claim_database = ClaimDatabase(path_dir_database=path_database_dir,
                                   path_raw_data_dir=path_raw_data,
                                   claim_data_set=claim_data_set,
                                   wiki_database=wiki_database)

    method_tokenization = 'tokenize'
    n_gram = 1
    delimiter_option = True
    claim_database_n_grams = ClaimDatabaseNgrams(path_dir_database=path_database_dir,
                                                 claim_database=claim_database,
                                                 method_tokenization=method_tokenization,
                                                 n_gram=n_gram,
                                                 delimiter_option=delimiter_option)