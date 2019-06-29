import os
import shutil
import json

from vocabulary import Vocabulary
from tfidf_database import TFIDFDatabase

import config

if __name__ == '__main__':
    # === file name === #
    experiment_list = [6,7,8,9,10]
    for experiment_nr in experiment_list:
        # experiment_nr = 11
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


    
