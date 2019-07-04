import os
from sqlitedict import SqliteDict
import sqlite3
import spacy
from tqdm import tqdm

from utils_db import num_files_in_directory
from utils_db import dict_save_json, dict_load_json, load_jsonl


class WikiDatabaseSqlite:

    def __init__(self, path_dir_database, path_wiki_pages):
        # === save input(s) ===#
        self.path_dir_database = path_dir_database
        self.path_wiki_pages = path_wiki_pages

        # === constants === #
        # id -> text, id -> title, title -> text, title ->id

        # === variables === #
        self.id_title_dict_flag = True

        self.path_settings = os.path.join(path_dir_database, 'settings.json')

        self.path_id_2_title = os.path.join(path_dir_database, 'id_2_title.sqlite')
        self.path_title_2_id = os.path.join(path_dir_database, 'title_2_id.sqlite')
        self.path_id_2_text = os.path.join(path_dir_database, 'id_2_text.sqlite')
        self.path_title_2_text = os.path.join(path_dir_database, 'title_2_text.sqlite')

        self.path_id_2_title_dict = os.path.join(path_dir_database, 'id_2_title.json')
        self.path_title_2_id_dict = os.path.join(path_dir_database, 'title_2_id.json')
        
        # === process === #
        self.nlp = spacy.load('en', disable=["parser", "ner"])

        if not (os.path.isfile(self.path_id_2_title) and os.path.isfile(self.path_title_2_id) and os.path.isfile(self.path_id_2_text) and os.path.isfile(self.path_title_2_text)):
            self.create_databases()

        if os.path.isfile(self.path_settings):
            print('Load existing settings file')
            self.settings = dict_load_json(self.path_settings)
            self.nr_wiki_pages = self.settings['nr_wikipedia_pages']
        else:
            raise ValueError('the settings dictionary should exist')

        # databases
        self.id_2_title_db = SqliteDict(self.path_id_2_title) # if autocommit: SqliteDict('./my_db.sqlite', autocommit=True)
        self.title_2_id_db = SqliteDict(self.path_title_2_id)
        self.id_2_text_db = SqliteDict(self.path_id_2_text)
        self.title_2_text_db = SqliteDict(self.path_title_2_text)

        self.title_2_id_dict  = None
        self.id_2_title_dict  = None

        self.get_title_dictionary()

    def get_title_dictionary(self):
        if os.path.isfile(self.path_title_2_id_dict) and os.path.isfile(self.path_id_2_title_dict):
            if self.id_title_dict_flag == True:
                print('Load title dictionary')
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
                self.id_2_title_dict[id_nr] = title
            
            dict_save_json(self.title_2_id_dict, self.path_title_2_id_dict)
            dict_save_json(self.id_2_title_dict, self.path_id_2_title_dict)

            self.id_title_dict_flag = self.tmp_id_title_dict_flag

    def create_databases(self):
        batch_size = 10
        
        if os.path.isfile(self.path_settings):
            print('Load existing settings file')
            settings = dict_load_json(self.path_settings)
        else:
            settings = {}

        nr_wikipedia_files = num_files_in_directory(self.path_wiki_pages)
        id_cnt = 0

        with SqliteDict(self.path_id_2_title) as dict_id_2_title:
            with SqliteDict(self.path_title_2_id) as dict_title_2_id:
                with SqliteDict(self.path_id_2_text) as dict_id_2_text:
                    with SqliteDict(self.path_title_2_text) as dict_title_2_text:
                        for wiki_page_nr in tqdm(range(1, nr_wikipedia_files + 1), desc='wiki_page_nr'):
                            # load json wikipedia dump file
                            wiki_page_path = os.path.join(self.path_wiki_pages, 'wiki-%.3d.jsonl' % (wiki_page_nr))
                            list_dict = load_jsonl(wiki_page_path)
                            
                            # iterate over pages
                            for page in list_dict:
                                title = page['id']
                                if title != '':
                                    text = page['text']
                                    
                                    dict_id_2_title[id_cnt] = title
                                    dict_title_2_id[title] = id_cnt
                                    dict_id_2_text[id_cnt] = text
                                    dict_title_2_text[title] = text
                                    id_cnt += 1
                            
                            # commit every batch_size'th document
                            if (wiki_page_nr%batch_size == 0) or (wiki_page_nr == nr_wikipedia_files):
                                dict_id_2_title.commit()
                                dict_title_2_id.commit()
                                dict_id_2_text.commit()
                                dict_title_2_text.commit()
                        dict_title_2_text.commit()
                    dict_id_2_text.commit()
                dict_title_2_id.commit()
            dict_id_2_title.commit()


        settings['nr_wikipedia_pages'] = id_cnt
        dict_save_json(settings, self.path_settings)

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


class WikiDatabase:
    """A sample Employee class"""
    def __init__(self, path_database, table_name):
        self.path_database = path_database
        self.table_name_data = table_name
        self.table_name_general_stats = 'general_statistics'
        
        self.conn = None
        self.c = None
        self.connect()
        self.initialise_tables()
        self.nr_rows = self.get_nr_rows()
        
    
    def connect(self):
        # description: connect to the database if it already exists and otherwise create a table
        print(self.path_database)
        self.conn = sqlite3.connect(self.path_database)
        self.c = self.conn.cursor()

    def initialise_tables(self):
        if os.path.isfile(self.path_database):
            self.c.execute("""CREATE TABLE if not exists %s (
                id integer primary key autoincrement,
                title text,
                text text,
                lines text
                )"""%(self.table_name_data))
            self.c.execute("""CREATE TABLE if not exists %s (
                id integer primary key autoincrement,
                variable text,
                text text,
                value real
                )"""%(self.table_name_general_stats))

    # === used === #
    def remove_all_docs(self):
        # remove all rows
        with self.conn:
            self.c.execute("DELETE from %s"%(self.table_name_data))

    def insert_doc(self, doc):
        with self.conn:
            self.c.execute("INSERT INTO %s (title, text, lines) VALUES (:title, :text, :lines)"%(self.table_name_data), {'title': doc.title, 'text': doc.text, 'lines': doc.lines})
    
    def update_nr_rows(self):
        row_name_nr_rows = 'nr_rows'
        self.nr_rows = int(self.c.execute("SELECT COUNT(*) FROM %s"%(self.table_name_data)).fetchone()[0])

        self.c.execute("SELECT id FROM %s WHERE variable=:variable"%(self.table_name_general_stats), {'variable': row_name_nr_rows})
        if self.c.fetchone():
            # if exists
            with self.conn:
                self.c.execute("UPDATE %s SET value=:value WHERE variable=:variable"%(self.table_name_general_stats),{'value': self.nr_rows, 'variable': row_name_nr_rows})
        else:
            # if not exists
            self.c.execute("INSERT INTO %s (variable, text, value) VALUES (:variable, :text, :value)"%(self.table_name_general_stats), 
                           {'variable': 'nr_rows', 'text': '', 'value': self.nr_rows})
        return self.nr_rows    

    def get_title_from_id(self, id_nr):
        self.c.execute("SELECT title FROM %s WHERE id=:id"%(self.table_name_data), {'id': id_nr})
        return self.c.fetchone()[0]

    def get_text_from_id(self, id_nr):
        self.c.execute("SELECT text FROM %s WHERE id=:id"%(self.table_name_data), {'id': id_nr})
        return self.c.fetchone()[0]


    # === not used === #
    def remove_doc(self, doc):
        with self.conn:
            self.c.execute("DELETE from %s WHERE title = :title"%(self.table_name_data),
                      {'title': doc.title,})
    
    def get_doc(self, method, value):
        method_list = ['id','title']
        if method not in method_list:
            raise ValueError('method not in method_list', method, method_list)
        self.c.execute("SELECT * FROM %s WHERE %s=:value"%(self.table_name, method), {'value': str(value)})
        return self.c.fetchone()
    
    def get_all_docs(self):
        self.c.execute("SELECT * FROM %s"%(self.table_name_data))
        return self.c.fetchall() 
    
    
            
    def get_nr_rows(self):
        row_name_nr_rows = 'nr_rows'

        self.c.execute("SELECT id FROM %s WHERE variable=:variable"%(self.table_name_general_stats), {'variable': row_name_nr_rows})
        if self.c.fetchone():
            # if exists
            self.c.execute("SELECT value FROM %s WHERE variable=:variable"%(self.table_name_general_stats), {'variable': row_name_nr_rows})
            self.nr_rows = int(self.c.fetchone()[0])
            return self.nr_rows
        else:
            # if not exists
            return self.update_nr_rows()
    
        
    
    def insert_doc_from_list(self, input_list):
        with self.conn:
            self.c.execute("INSERT INTO %s (title, text, lines) VALUES (:title, :text, :lines)"%(self.table_name_data), 
                           {'title': input_list[0], 'text': input_list[1], 'lines': input_list[2]})
    
    def get_text_from_title(self, title):
        self.c.execute("SELECT text FROM %s WHERE title=:title"%(self.table_name_data), {'title': title})
        return self.c.fetchone()
    
    def get_lines(self, title):
        self.c.execute("SELECT lines FROM %s WHERE title=:title"%(self.table_name_data), {'title': title})
        return self.c.fetchone()
    def get_all_titles(self):
        self.c.execute("SELECT title FROM %s"%(self.table_name_data))
        return self.c.fetchall() 
    
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
        'tokenize_lemma_nouns', 'tokenize_lemma_prop_nouns', 'tokenize_lemma_number', 'tokenize_lemma_pos'] 

        for method in method_list:
            if method not in method_options:
                raise ValueError('method not in method_options', method, method_options)
            method = getattr(self, method, lambda: "Method Options")
            text = method()
            
        return text
    
    def tokenize(self):
        return [word.text for word in self.text]

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
        return [(word.pos_ + self.delimiter_position_tag + word.text).lower() for word in self.text]