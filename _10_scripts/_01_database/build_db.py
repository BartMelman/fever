import sqlite3
import os
from tqdm import tqdm

from tqdm import tnrange
from document import Document
from wiki_database import WikiDatabase

# from tf_idf import count_n_grams, vocabulary_text, tf_idf
from utils_db import num_files_in_directory, load_jsonl, save_dict_pickle, load_dict_pickle, dict_load_json, dict_save_json
# from tqdm.auto import tqdm
import nltk
# nltk.download('averaged_perceptron_tagger')
import sys

import config

path_text_database = 'text.db'
path_wiki_database = 'wiki.db'
table_name_text = 'text_table'
table_name_wiki = 'wikipages'


def create_wiki_database(path_wiki_database, table_name_wiki, path_wiki_pages):
    # description: iterate over all wikipedia pages and store the result in a database
    # input
    #  - path_wiki_database: path where the database needs to be stored
    #  - table_name_wiki: table name within the database where all the wikipages are stored
    #  - path_wiki_pages: path where the wikipages are stored
    # output
    #  - wiki_database: the database
    if os.path.isfile(path_wiki_database):
        raise ValueError('The database already exists.', path_wiki_database)

    wiki_database = WikiDatabase(path_wiki_database, table_name_wiki)
    wiki_database.remove_all_docs()
    id_loc = 0
    text_loc = 1
    lines_loc = 2

    nr_wikipedia_files = num_files_in_directory(path_wiki_pages)
    for wiki_page_nr in tqdm(range(1, nr_wikipedia_files + 1), desc='wiki_page_nr'):
        wiki_page_path = os.path.join(
            path_wiki_pages, 'wiki-%.3d.jsonl' % (wiki_page_nr))
        list_dict = load_jsonl(wiki_page_path)

        nr_lines = len(list_dict)

        for i in tqdm(range(nr_lines), desc='lines'):
            if list_dict[i]['id'] != '':
                doc = Document(list_dict[i]['id'], list_dict[i][
                               'text'], list_dict[i]['lines'])
        #         wiki_database.insert_doc_from_list([list_dict[i]['id'], list_dict[i]['text'], list_dict[i]['lines']])
                wiki_database.insert_doc(doc)
    return wiki_database


def update_dictionary_df(dictionary_df, dictionary_new):
    # description: for all keys in dictionary_new add one count
    # to the corresponding keys in dictionary_df or create a new key if the
    # key does not exist yet and initialise it with one.

    # input 
    #  - dictionary_df: 
    #  - dictionary_new: 
    # output
    #  - dictionary_df: 
    for key in dictionary_new.keys():
        if key in dictionary_df.keys():
            dictionary_df[key] += 1
        else:
            dictionary_df[key] = 1

    if 'nr_documents' in dictionary_df.keys():
        dictionary_df['nr_documents'] += 1
    else:
        dictionary_df['nr_documents'] = 1
    return dictionary_df

if __name__ == '__main__':

    path_wiki_pages = os.path.join(
        config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database = os.path.join(
        config.ROOT, config.DATA_DIR, config.DATABASE_DIR, 'wiki.db')

    # path_tf_dict = os.path.join(
    #     __root__, __jupyter_notebook_dir__, 'dictionary_tf')
    # path_idf_dict = os.path.join(
    #     __root__, __jupyter_notebook_dir__, 'dictionary_idf')

    wiki_database = create_wiki_database(
        path_wiki_database, table_name_wiki, path_wiki_pages)
    wiki_database.update_nr_rows()

    # text_database = TextDatabase(
    #     path_text_database, path_wiki_database, table_name_text, table_name_wiki)
