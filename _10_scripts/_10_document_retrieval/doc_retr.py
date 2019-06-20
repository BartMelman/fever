import os
# from sqlitedict import SqliteDict
# import sqlite3
import tqdm
from tqdm import tnrange
import shutil

# from document import Document

# from utils_db import dict_save_json, dict_load_json
# from tf_idf import count_n_grams
# from dictionary_batch_idf import DictionaryBatchIDF
from vocabulary import Vocabulary, count_n_grams
from tfidf_database import TFIDFDatabase

# from _10_scripts._01_database.wiki_database import WikiDatabase

import config


if __name__ == '__main__':
    # === constants === #
    path_large_wiki_database = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR,'wiki.db')#'/home/bmelman/C_disk/02_university/06_thesis/01_code/fever/_01_data/_03_database/wiki.db' 
    # path_wiki_database = 'wiki.db'
    table_name_wiki = 'wikipages'
    # table_name_tf_idf = 'tf_idf'
    # path_tf_idf_database = 'tf_idf.db'
    # path_mydict_tf_idf = 'mydict_tf_idf.sqlite'
    # path_mydict_ids = 'mydict_ids.sqlite'

    # === settings experiment === #
    n_gram = 1
    # method_tokenization = ['tokenize', 'remove_space', 'make_lower_case', 'lemmatization_get_nouns']
    method_tokenization = ['tokenize', 'make_lower_case']
    threshold = 0.005
    method_tf = 'term_frequency' # raw_count term_frequency
    method_df = 'inverse_document_frequency' # 
    delimiter = '\k'

    vocab = Vocabulary(path_large_wiki_database, table_name_wiki, n_gram, method_tokenization, 'text')
    tf_idf_db = TFIDFDatabase(vocab, method_tf, method_df, delimiter, threshold, 'text')


    
