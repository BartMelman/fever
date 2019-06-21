import os
# from sqlitedict import SqliteDict
# import sqlite3
# import tqdm
# from tqdm import tnrange
import shutil
import json

# from document import Document

# from utils_db import dict_save_json, dict_load_json
# from tf_idf import count_n_grams
# from dictionary_batch_idf import DictionaryBatchIDF
from vocabulary import Vocabulary
# from vocabulary import count_n_grams

from tfidf_database import TFIDFDatabase

# from _10_scripts._01_database.wiki_database import WikiDatabase

import config

if __name__ == '__main__':
    # === file name === #
    experiment_nr = 4
    file_name = 'experiment_%.2d.json'%(experiment_nr)
    path_experiment = os.path.join(config.ROOT, config.CONFIG_DIR, file_name)
    print(path_experiment)
    with open(path_experiment) as json_data_file:
        data = json.load(json_data_file)

    # === run === #
    vocab = Vocabulary(path_wiki_database = os.path.join(config.ROOT, data['path_large_wiki_database']), 
        table_name_wiki = data['table_name_wiki'], n_gram = data['n_gram'],
        method_tokenization = data['method_tokenization'], source = data['vocabulary_source'])

    tf_idf_db = TFIDFDatabase(vocabulary = vocab, method_tf = data['method_tf'], method_df = data['method_df'],
        delimiter = data['delimiter'], threshold = data['threshold'], source = data['tf_idf_source'])


    
