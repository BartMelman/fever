import os
from tqdm import tqdm
import unicodedata

from wiki_database import WikiDatabaseSqlite
from utils_wiki_database import normalise_text
from utils_db import mkdir_if_not_exist, load_jsonl
from template import Settings
from database import Database

import config

class Evidence:
    def __init__(self, evidence):
        self.evidence = evidence
        self.nr_annotators = len(evidence)

    def get_nr_annotators(self):
        return self.nr_annotators

    def get_nr_evidence_sentences(self, annotator_nr):
        return len(self.evidence[annotator_nr])

    def get_evidence(self, annotator_nr, sentence_nr):
        evidence_line = self.evidence[annotator_nr][sentence_nr]
        doc_nr = evidence_line[2]
        sentence_nr = evidence_line[3]
        return doc_nr, sentence_nr

class ClaimDatabase:
    def __init__(self, path_dir_database, path_raw_data_dir, claim_data_set, wiki_database=None):
        # === save input(s) ===#
        self.path_raw_data_dir = path_raw_data_dir
        self.claim_data_set = claim_data_set
        self.path_dir_database = os.path.join(path_dir_database, 'claim_database_' + self.claim_data_set)

        # === variables === #
        self.path_raw_claims = os.path.join(path_raw_data_dir, self.claim_data_set + '.jsonl')

        self.verifiable_2_int = {}
        self.verifiable_2_int['NOT VERIFIABLE'] = 0
        self.verifiable_2_int['VERIFIABLE'] = 1
        self.int_2_verifiable = {}
        self.int_2_verifiable[0] = 'NOT VERIFIABLE'
        self.int_2_verifiable[1] = 'VERIFIABLE'

        self.label_2_int = {}
        self.label_2_int['REFUTES'] = 0
        self.label_2_int['SUPPORTS'] = 1
        self.label_2_int['NOT ENOUGH INFO'] = 2
        self.int_2_label = {}
        self.int_2_label[0] = 'REFUTES'
        self.int_2_label[1] = 'SUPPORTS'
        self.int_2_label[2] = 'NOT ENOUGH INFO'

        # === process === #
        print('ClaimDatabase')

        mkdir_if_not_exist(self.path_dir_database)

        self.settings = Settings(path_settings_dir=self.path_dir_database)

        self.id_2_id_number_db = Database(path_database_dir=self.path_dir_database,
                                          database_name='id_2_id_number',
                                          database_method='lsm',
                                          input_type='int',
                                          output_type='int',
                                          checks_flag=True)
        self.id_2_verifiable_db = Database(path_database_dir=self.path_dir_database,
                                           database_name='id_2_verifiable',
                                           database_method='lsm',
                                           input_type='int',
                                           output_type='int',
                                           checks_flag=True)
        self.id_2_label_db = Database(path_database_dir=self.path_dir_database,
                                      database_name='id_2_label',
                                      database_method='lsm',
                                      input_type='int',
                                      output_type='int',
                                      checks_flag=True)
        self.id_2_claim_db = Database(path_database_dir=self.path_dir_database,
                                      database_name='id_2_claim',
                                      database_method='lsm',
                                      input_type='int',
                                      output_type='string',
                                      checks_flag=True)
        self.id_2_evidence_db = Database(path_database_dir=self.path_dir_database,
                                         database_name='id_2_evidence',
                                         database_method='lsm',
                                         input_type='int',
                                         output_type='list_str',
                                         checks_flag=True)

        # === create database === #
        self.flag_function_call(function_name='create_database', arg_list=[wiki_database])

        self.nr_claims = self.settings.get_item(key='nr_claims')

        print('***finished***')

    def get_item(self, input_type, input_value, output_type):
        if input_type == 'id':
            if output_type == 'id_number':
                # return the id number as specified in the raw data
                return self.id_2_id_number_db.get_item(input_value)
            elif output_type == 'verifiable_int':
                # return the verifiable flag in integer (int) format (0 or 1)
                return self.id_2_verifiable_db.get_item(input_value)
            elif output_type == 'verifiable_str':
                # return the verifiable flag in string(str) format ('NOT VERIFIABLE', 'VERIFIABLE')
                return self.int_2_verifiable[self.id_2_verifiable_db.get_item(input_value)]
            elif output_type == 'label_int':
                # return the label in integer (int) format (0, 1, 2)
                return self.id_2_label_db.get_item(input_value)
            elif output_type == 'label_str':
                # return the label in string (str) format ('REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO')
                return self.int_2_label[self.id_2_label_db.get_item(input_value)]
            elif output_type == 'claim':
                return self.id_2_claim_db.get_item(input_value)
            elif output_type == 'evidence':
                return self.id_2_evidence_db.get_item(input_value)
            elif output_type == 'evidence_class':
                return Evidence(self.id_2_evidence_db.get_item(input_value))
            else:
                raise ValueError('output_type not in options', output_type)
        else:
            raise ValueError('input_type not in options', input_type)

    def create_database(self, wiki_database):
        list_claim_dicts = load_jsonl(self.path_raw_claims)
        nr_claims = len(list_claim_dicts)

        for id in tqdm(range(nr_claims), desc='claims'):
            dict_claim_id = list_claim_dicts[id]
            #             print(dict_claim_id['claim'], type(dict_claim_id['claim']))
            dict_claim_id['verifiable'] = unicodedata.normalize('NFD', normalise_text(dict_claim_id['verifiable']))
            dict_claim_id['claim'] = unicodedata.normalize('NFD', normalise_text(dict_claim_id['claim']))
            for interpreter in range(len(dict_claim_id['evidence'])):
                for proof in range(len(dict_claim_id['evidence'][interpreter])):
                    if dict_claim_id['evidence'][interpreter][proof][2] != None:
                        title = unicodedata.normalize('NFD',
                                                      normalise_text(dict_claim_id['evidence'][interpreter][proof][2]))
                        dict_claim_id['evidence'][interpreter][proof][2] = wiki_database.get_item(input_type='title',
                                                                                                  input_value=title,
                                                                                                  output_type='id')

            self.id_2_id_number_db.store_item(key=id, value=dict_claim_id['id'])

            self.id_2_verifiable_db.store_item(key=id, value=self.verifiable_2_int[dict_claim_id['verifiable']])

            self.id_2_label_db.store_item(key=id, value=self.label_2_int[dict_claim_id['label']])

            #             print(dict_claim_id['claim'], type(dict_claim_id['claim']))
            self.id_2_claim_db.store_item(key=id, value=dict_claim_id['claim'])

            self.id_2_evidence_db.store_item(key=id, value=dict_claim_id['evidence'])

        self.settings.add_item(key='nr_claims', value=nr_claims)

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