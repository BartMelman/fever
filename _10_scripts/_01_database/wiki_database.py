import os
import sqlite3

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
            
    def insert_doc(self, doc):
        with self.conn:
            self.c.execute("INSERT INTO %s (title, text, lines) VALUES (:title, :text, :lines)"%(self.table_name_data), {'title': doc.title, 'text': doc.text, 'lines': doc.lines})
    
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
    
    def remove_all_docs(self):
        # remove all rows
        with self.conn:
            self.c.execute("DELETE from %s"%(self.table_name_data))
            
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
    
    def insert_doc_from_list(self, input_list):
        with self.conn:
            self.c.execute("INSERT INTO %s (title, text, lines) VALUES (:title, :text, :lines)"%(self.table_name_data), 
                           {'title': input_list[0], 'text': input_list[1], 'lines': input_list[2]})
    def get_title_from_id(self, id_nr):
        self.c.execute("SELECT title FROM %s WHERE id=:id"%(self.table_name_data), {'id': id_nr})
        return self.c.fetchone()[0]
    
    def get_text_from_id(self, id_nr):
        self.c.execute("SELECT text FROM %s WHERE id=:id"%(self.table_name_data), {'id': id_nr})
        return self.c.fetchone()[0]
    
    def get_text_from_title(self, title):
        self.c.execute("SELECT text FROM %s WHERE title=:title"%(self.table_name_data), {'title': title})
        return self.c.fetchone()
    
    def get_lines(self, title):
        self.c.execute("SELECT lines FROM %s WHERE title=:title"%(self.table_name_data), {'title': title})
        return self.c.fetchone()
    def get_all_titles(self):
        self.c.execute("SELECT title FROM %s"%(self.table_name_data))
        return self.c.fetchall() 
    
    