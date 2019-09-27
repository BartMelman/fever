import os
import spacy
from tqdm import tqdm

from utils_wiki_database import normalise_text
from utils_db import num_files_in_directory
from utils_db import dict_save_json, dict_load_json, load_jsonl
from utils_db import mkdir_if_not_exist

from template import Settings
from database import Database

import config

import re

def get_list_lines(input_wiki_page_dict):
    string = input_wiki_page_dict['lines']
    if string == '':
        return ['']
    else:
        string = '\n' + string
        lines_list = re.split(r'(\n\d+\t)', string)[1:]

        skip = 0
        line_nr = 0
        list_sentences = []
        for i in range(len(lines_list)):
            if i % 2 == 0:
                # string
                line_nr = int(re.split('\n|\t', lines_list[i])[1])
            else:
                # line nr
                if int((i - skip) / 2.0) == line_nr:
                    string = lines_list[i].split('\t')[0]
                    list_sentences.append(string)
                else:
                    print(i, int(i / 2.0), 'should be equal to', line_nr)
                    skip += 2
                    # raise ValueError(i, int(i / 2.0), 'should be equal to', line_nr)

        return list_sentences

def get_dict_lines(input_wiki_page_dict):
    output_wiki_page_dict = {}
    line_nr = 0
    for lines_list in input_wiki_page_dict['lines'].split('\n'):
        splitted_lines_list = lines_list.split('\t')
        if len(splitted_lines_list) > 1:
            line_nr = int(splitted_lines_list[0])
            line_text = splitted_lines_list[1]
            output_wiki_page_dict[line_nr] = normalise_text(line_text)
    output_wiki_page_dict['nr_lines'] = int(line_nr)
    return output_wiki_page_dict

# def get_list_lines(input_wiki_page_dict):
#     dict_lines = get_dict_lines(input_wiki_page_dict)
#     list_lines = []
#     for line_nr in range(dict_lines['nr_lines']+1):
#         if line_nr in dict_lines:
#             list_lines.append(dict_lines[line_nr])
#         else:
#             list_lines.append('')
#     return list_lines

class WikiDatabaseSqlite:

    def __init__(self, path_dir_database, path_wiki_pages):
        # === save input(s) ===#
        self.path_dir_database = os.path.join(path_dir_database, 'wiki_database')
        self.path_wiki_pages = path_wiki_pages

        # === variables === #

        # === process === #
        print('WikiDatabase')

        mkdir_if_not_exist(self.path_dir_database)

        self.settings = Settings(self.path_dir_database)

        self.title_2_id_db = Database(path_database_dir=self.path_dir_database,
                                      database_name='title_2_id',
                                      database_method='lsm',
                                      input_type='string',
                                      output_type='int',
                                      checks_flag=True)
        self.id_2_title_db = Database(path_database_dir=self.path_dir_database,
                                      database_name='id_2_title',
                                      database_method='lsm',
                                      input_type='int',
                                      output_type='string',
                                      checks_flag=True)
        self.id_2_text_db = Database(path_database_dir=self.path_dir_database,
                                     database_name='id_2_text',
                                     database_method='lsm',
                                     input_type='int',
                                     output_type='string',
                                     checks_flag=True)
        self.id_2_lines_db = Database(path_database_dir=self.path_dir_database,
                                      database_name='id_2_lines',
                                      database_method='lsm',
                                      input_type='int',
                                      output_type='list_str',
                                      checks_flag=True)

        # === create database === #
        self.flag_function_call(function_name='create_database', arg_list=[])

        self.nr_wikipedia_pages = self.settings.get_item(key='nr_wikipedia_pages')

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

        elif input_type == 'title':
            if output_type == 'id':
                return self.title_2_id_db.get_item(input_value)
            else:
                raise ValueError('output_type not in options', output_type)
        else:
            raise ValueError('input_type not in options', input_type)

    def create_database(self):
        # description :
        # input :
        # -
        # output :
        # -

        nr_wikipedia_files = num_files_in_directory(self.path_wiki_pages)

        self.settings.add_item(key = 'nr_wikipedia_files', value = nr_wikipedia_files)

        id_cnt = 0

        for wiki_page_nr in tqdm(range(1, nr_wikipedia_files + 1), desc='wiki_page_nr'):
            # load json wikipedia dump file
            wiki_page_path = os.path.join(self.path_wiki_pages, 'wiki-%.3d.jsonl' % (wiki_page_nr))
            list_dict = load_jsonl(wiki_page_path)

            # iterate over pages
            for page in list_dict:
                title = normalise_text(page['id'])
                if title != '':
                    text = normalise_text(page['text'])
                    self.title_2_id_db.store_item(key = title, value = id_cnt)
                    self.id_2_title_db.store_item(key = id_cnt, value = title)
                    self.id_2_text_db.store_item(key = id_cnt, value = text)
                    self.id_2_lines_db.store_item(key = id_cnt, value = get_list_lines(page))

                    id_cnt += 1

        self.settings.add_item(key = 'nr_wikipedia_pages', value = id_cnt)

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