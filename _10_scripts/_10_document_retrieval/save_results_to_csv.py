import os
from tqdm import tqdm
import csv

from utils_db import HiddenPrints
from wiki_database import WikiDatabaseSqlite
from utils_doc_results_db import get_tf_idf_from_exp
from doc_results import PerformanceTFIDF
from utils_db import mkdir_if_not_exist

import config

if __name__ == '__main__':

    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
       
    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)

    claim_data_set = 'dev' # 
    experiment_nr_list = [31,32,33,34,35,36,37,38,39, 41,51]
    list_K = [5, 10, 20, 40, 100]
    score_list = ['e_score', 'f_score', 'e_score_labelled', 'f_score_labelled']
    title_tf_idf_flag_normalise_list = [True, False]

    score_method = 'e_score'
    list_results = []
    for experiment_nr in tqdm(experiment_nr_list):
        for K in list_K:
            for title_tf_idf_normalise_flag in title_tf_idf_flag_normalise_list:
                results_exp = PerformanceTFIDF(wiki_database, experiment_nr, claim_data_set, K, score_method, title_tf_idf_normalise_flag)
                dictionary = results_exp.score_dict
                list_results.append([claim_data_set, experiment_nr, K, 
                                     title_tf_idf_normalise_flag, 
                                     results_exp.tf_idf_db.method_tokenization,
                                     results_exp.tf_idf_db.method_tf,
                                     results_exp.tf_idf_db.method_df,
                                     dictionary['e_score'], 
                                     dictionary['f_score'],
                                     dictionary['e_score_labelled']['SUPPORTS'],
                                     dictionary['e_score_labelled']['NOT ENOUGH INFO'],
                                     dictionary['e_score_labelled']['REFUTES'],
                                     dictionary['f_score_labelled']['SUPPORTS'],
                                     dictionary['f_score_labelled']['NOT ENOUGH INFO'],
                                     dictionary['f_score_labelled']['REFUTES']
                                    ])
                results_exp = None
    path_csv_dir = os.path.join(config.ROOT, config.RESULTS_DIR, 'csv_results')
    path_csv = os.path.join(path_csv_dir, "experiments.csv")
    mkdir_if_not_exist(path_csv_dir)
    with open(path_csv, 'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(list_results)