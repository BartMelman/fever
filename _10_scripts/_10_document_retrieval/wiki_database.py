import os
from sqlitedict import SqliteDict
import sqlite3
import spacy
from tqdm import tqdm

from utils_wiki_database import normalise_text
from utils_db import num_files_in_directory
from utils_db import dict_save_json, dict_load_json, load_jsonl
from utils_db import mkdir_if_not_exist

class WikiDatabaseSqlite:

    def __init__(self, path_dir_database, path_wiki_pages):
        # === save input(s) ===#
        self.path_dir_database = path_dir_database
        self.path_wiki_pages = path_wiki_pages

        # === constants === #

        # === variables === #
        self.id_title_dict_flag = True

        self.path_settings = os.path.join(path_dir_database, 'settings.json')

        self.path_id_2_title = os.path.join(path_dir_database, 'id_2_title.sqlite')
        self.path_title_2_id = os.path.join(path_dir_database, 'title_2_id.sqlite')
        self.path_id_2_text = os.path.join(path_dir_database, 'id_2_text.sqlite')
        self.path_title_2_text = os.path.join(path_dir_database, 'title_2_text.sqlite')
        self.path_id_2_lines = os.path.join(path_dir_database, 'id_2_lines.sqlite')
        self.path_title_2_lines = os.path.join(path_dir_database, 'title_2_lines.sqlite')

        self.path_id_2_title_dict = os.path.join(path_dir_database, 'id_2_title.json')
        self.path_title_2_id_dict = os.path.join(path_dir_database, 'title_2_id.json')
        
        # === process === #
        print('wiki_database')
        self.nlp = spacy.load('en', disable=["parser", "ner"])
        print(self.path_dir_database)
        mkdir_if_not_exist(self.path_dir_database)
        if not (os.path.isfile(self.path_id_2_title) and os.path.isfile(self.path_title_2_id) and os.path.isfile(self.path_id_2_text) and os.path.isfile(self.path_title_2_text)):
            self.create_databases()

        if os.path.isfile(self.path_settings):
            print('- Load existing settings file')
            self.settings = dict_load_json(self.path_settings)
            self.nr_wiki_pages = self.settings['nr_wikipedia_pages']
        else:
            raise ValueError('the settings dictionary should exist')

        # databases
        self.id_2_title_db = SqliteDict(self.path_id_2_title) # if autocommit: SqliteDict('./my_db.sqlite', autocommit=True)
        self.title_2_id_db = SqliteDict(self.path_title_2_id)
        self.id_2_text_db = SqliteDict(self.path_id_2_text)
        self.title_2_text_db = SqliteDict(self.path_title_2_text)
        self.id_2_lines_db = SqliteDict(self.path_id_2_lines)
        self.title_2_lines_db = SqliteDict(self.path_title_2_lines)

        self.title_2_id_dict  = None
        self.id_2_title_dict  = None

        self.get_title_dictionary()

    def get_title_dictionary(self):
        if os.path.isfile(self.path_title_2_id_dict) and os.path.isfile(self.path_id_2_title_dict):
            if self.id_title_dict_flag == True:
                print('- Load title dictionary')
                self.title_2_id_dict = dict_load_json(self.path_title_2_id_dict)
                self.id_2_title_dict = dict_load_json(self.path_id_2_title_dict)
        else:
            self.title_2_id_dict  = {}
            self.id_2_title_dict  = {}
            self.tmp_id_title_dict_flag = self.id_title_dict_flag
            self.id_title_dict_flag = False
            for id_nr in tqdm(range(self.nr_wiki_pages), desc='title-id-dictionary'):
                title = self.get_title_from_id(id_nr)
                self.title_2_id_dict[title] = id_nr
                self.id_2_title_dict[str(id_nr)] = title
            
            dict_save_json(self.title_2_id_dict, self.path_title_2_id_dict)
            dict_save_json(self.id_2_title_dict, self.path_id_2_title_dict)

            self.id_title_dict_flag = self.tmp_id_title_dict_flag

    def create_databases(self):
        batch_size = 10
        
        if os.path.isfile(self.path_settings):
            print('- Load existing settings file')
            settings = dict_load_json(self.path_settings)
        else:
            settings = {}
        nr_wikipedia_files = num_files_in_directory(self.path_wiki_pages)
        id_cnt = 0

        with SqliteDict(self.path_id_2_title) as dict_id_2_title:
            with SqliteDict(self.path_title_2_id) as dict_title_2_id:
                with SqliteDict(self.path_id_2_text) as dict_id_2_text:
                    with SqliteDict(self.path_title_2_text) as dict_title_2_text:
                        with SqliteDict(self.path_id_2_lines) as dict_id_2_lines:
                            with SqliteDict(self.path_title_2_lines) as dict_title_2_lines:
                                for wiki_page_nr in tqdm(range(1, nr_wikipedia_files + 1), desc='wiki_page_nr'):
                                    # load json wikipedia dump file
                                    wiki_page_path = os.path.join(self.path_wiki_pages, 'wiki-%.3d.jsonl' % (wiki_page_nr))
                                    list_dict = load_jsonl(wiki_page_path)
                                    
                                    # iterate over pages
                                    for page in list_dict:
                                        title = normalise_text(page['id'])
                                        if title != '':
                                            text = normalise_text(page['text'])
                                            
                                            dict_id_2_title[id_cnt] = title
                                            dict_title_2_id[title] = id_cnt
                                            dict_id_2_text[id_cnt] = text
                                            dict_title_2_text[title] = text
                                            dict_id_2_lines[id_cnt] = get_dict_lines(page)
                                            dict_title_2_lines[title] = get_dict_lines(page)

                                            id_cnt += 1
                                    # commit every batch_size'th document
                                    if (wiki_page_nr%batch_size == 0) or (wiki_page_nr == nr_wikipedia_files):
                                        dict_id_2_title.commit()
                                        dict_title_2_id.commit()
                                        dict_id_2_text.commit()
                                        dict_title_2_text.commit()
                                        dict_id_2_lines.commit()
                                        dict_title_2_lines.commit()
                                dict_title_2_lines.commit()
                            dict_id_2_lines.commit()
                        dict_title_2_text.commit()
                    dict_id_2_text.commit()
                dict_title_2_id.commit()
            dict_id_2_title.commit()


        settings['nr_wikipedia_pages'] = id_cnt
        dict_save_json(settings, self.path_settings)

    def get_lines_list_from_title(self, title):
        dict_wiki_page = self.title_2_lines_db[title]
        nr_lines = dict_wiki_page['nr_lines']
        lines_list = []
        for line_nr in range(nr_lines):
            if str(line_nr) in dict_wiki_page:
                line_text = dict_wiki_page[str(line_nr)]
                lines_list.append(line_text)
            else:
                lines_list.append('')
        return lines_list

    def get_line_from_title(self, title, line_nr):
        dict_wiki_page = self.title_2_lines_db[title]
        # print(dict_wiki_page)
        return dict_wiki_page[str(line_nr)]

    def get_title_from_id(self, id_nr):
        if self.id_title_dict_flag == True:
            return self.id_2_title_dict[str(id_nr)]
        else:
            return self.id_2_title_db[id_nr]

    def get_id_from_title(self, title):
        if self.id_title_dict_flag == True:
            return self.title_2_id_dict[title]
        else:
            return self.title_2_id_db[title]

    def get_text_from_id(self, id_nr):
        return self.id_2_text_db[id_nr]

    def get_text_from_title(self, title):
        return self.title_2_text_db[title]

    def get_tokenized_text_from_id(self, id_nr, method_list = []):
        # method: 
        text = Text(self.get_text_from_id(id_nr), 'text', self.nlp)
        tokenized_text = text.process(method_list)
        return tokenized_text
    
    def get_tokenized_title_from_id(self, id_nr, method_list = []):
        # recover title from wiki_database and return the tokonized title
        title = Text(self.get_title_from_id(id_nr), 'title', self.nlp)
        tokenized_title = title.process(method_list)
        return tokenized_title

    
class Text:
    """A sample Employee class"""
    def __init__(self, text):
        self.list_accepted_tags = ['INTJ', 'NOUN', 'NUM', 'PART', 'PROPN', 'SYM', 'X']
        self.list_nouns = ['NOUN', 'PROPN']
        self.list_symb = ['SYM']
        self.list_num = ['NUM']
        self.list_prop_noun = ['PROPN']

        self.text = text
        self.delimiter_position_tag = '\z'

    def process(self, method_list):
        """Dispatch method"""
        method_options = ['tokenize', 'tokenize_lemma', 'tokenize_lemma_list_accepted', 
        'tokenize_lemma_nouns', 'tokenize_lemma_prop_nouns', 'tokenize_lemma_number', 'tokenize_lemma_pos', 'tokenize_text_pos'] 

        for method in method_list:
            if method not in method_options:
                raise ValueError('method not in method_options', method, method_options)
            method = getattr(self, method, lambda: "Method Options")
            text = method()
            
        return text
    
    def tokenize(self):
        return [word.text for word in self.text]

    def tokenize_lower(self):
        return [(word.text).lower() for word in self.text]

    def tokenize_lemma(self):
        return [word.lemma_ for word in self.text]

    def tokenize_lemma_list_accepted(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_accepted_tags]

    def tokenize_lemma_nouns(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_nouns]

    def tokenize_lemma_prop_nouns(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_prop_noun]

    def tokenize_lemma_number(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_num]

    def tokenize_lemma_pos(self):
        return [word.pos_ + self.delimiter_position_tag + word.lemma_ for word in self.text]

    def tokenize_text_pos(self):
        return [word.pos_ + self.delimiter_position_tag + word.text for word in self.text]

    def tokenize_lower_pos(self):
        return [word.pos_ + self.delimiter_position_tag + (word.text).lower() for word in self.text]

    def tokenize_tag(self):
        return [word.pos_ for word in self.text]

def get_dict_lines(input_wiki_page_dict):
    output_wiki_page_dict = {}
    line_nr = 0
    for lines_list in input_wiki_page_dict['lines'].split('\n'):
        splitted_lines_list = lines_list.split('\t')
        if len(splitted_lines_list) > 1:
            line_nr = splitted_lines_list[0]
            line_text = splitted_lines_list[1]
            output_wiki_page_dict[str(line_nr)] = normalise_text(line_text)
    output_wiki_page_dict['nr_lines'] = int(line_nr)
    return output_wiki_page_dict

class WikipagesLines:
    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.base_dir = os.path.join(path_dir, 'wiki_database_lines')
        mkdir_if_not_exist(self.base_dir)

    def add_page_2_database(self, input_wiki_page_dict, title_id):
        file_name = str(title_id) + '.json'
        path_wiki_page_save = os.path.join(self.base_dir, file_name)
        if os.path.isfile(path_wiki_page_save):
            raise ValueError('file already exists', path_wiki_page_save)
        output_wiki_page_dict = {}
        line_nr = 0
        for lines_list in input_wiki_page_dict['lines'].split('\n'):
            splitted_lines_list = lines_list.split('\t')
            if len(splitted_lines_list) > 1:
                line_nr = splitted_lines_list[0]
                line_text = splitted_lines_list[1]
                output_wiki_page_dict[str(line_nr)] = normalise_text(line_text)
        output_wiki_page_dict['nr_lines'] = int(line_nr)
        dict_save_json(output_wiki_page_dict, path_wiki_page_save)
    
    def get_line_from_title(self, title, line_nr):
        title_id = self.get_id_from_title(title)
        return self.get_line_from_title_id(self, title_id, line_nr)

    def get_line_from_title_id(self, title_id, line_nr):
        file_name = str(title_id) + '.json'
        path_wiki_page = os.path.join(self.base_dir, file_name)
        dict_wiki_page = dict_load_json(path_wiki_page)
        return dict_wiki_page[str(line_nr)]

    def get_line_list_from_title_id(self, title_id, line_nr_list):
        file_name = str(title_id) + '.json'
        path_wiki_page = os.path.join(self.base_dir, file_name)
        dict_wiki_page = dict_load_json(path_wiki_page)
        return [dict_wiki_page[str(line_nr)] for line_nr in line_nr_list]

    def get_line_list(self, title_id):
        file_name = str(title_id) + '.json'
        path_wiki_page = os.path.join(self.base_dir, file_name)
        dict_wiki_page = dict_load_json(path_wiki_page)
        nr_lines = dict_wiki_page['nr_lines']
        lines_list = []
        for line_nr in range(nr_lines):
            if str(line_nr) in dict_wiki_page:
                line_text = dict_wiki_page[str(line_nr)]
                lines_list.append(line_text)
        return lines_list