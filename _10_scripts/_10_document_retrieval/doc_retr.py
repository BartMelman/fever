import os
import shutil
import json

from vocabulary import VocabularySqlite
from tfidf_database import TFIDFDatabaseSqlite
from wiki_database import WikiDatabaseSqlite
# from text_database import TextDatabaseSqlite

import config

if __name__ == '__main__':

    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)

    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)

    experiment_list = [31]#[31, 32, 33, 34, 35, 36, 37,38, 39]

    for experiment_nr in experiment_list:
        # experiment_nr = 11
        file_name = 'experiment_%.2d.json'%(experiment_nr)
        path_experiment = os.path.join(config.ROOT, config.CONFIG_DIR, file_name)

        with open(path_experiment) as json_data_file:
            data = json.load(json_data_file)

        # === run === #
        vocab = VocabularySqlite(wiki_database = wiki_database, n_gram = data['n_gram'],
            method_tokenization = data['method_tokenization'], tags_in_db_flag = data['tags_in_db_flag'], 
            source = data['vocabulary_source'], tag_list_selected = data['tag_list_selected'])

        tf_idf_db = TFIDFDatabaseSqlite(vocabulary = vocab, method_tf = data['method_tf'], method_df = data['method_df'],
            delimiter = data['delimiter'], threshold = data['threshold'], source = data['tf_idf_source'])


    
