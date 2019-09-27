import os
import json
from lsm import LSM

from utils_db import mkdir_if_not_exist, dict_save_json, dict_load_json, database_save_json, database_load_json


class Database:
    def __init__(self, path_database_dir, database_name, database_method,
                 input_type, output_type, encoding='utf-8', checks_flag=True):
        # input:
        # - path_database_dir : path of the directory of the database
        # - database_name : name of database without extension
        # - database_method : package/method used to construct database
        # - input_type : format of key in dictionary that should always
        #     be used for the database (e.g.'str', 'int', 'float', etc, )
        # - output_type : format of value of dictionary that is
        #     fixed for the database (e.g. 'str', 'int', 'float', etc)
        # - encoding : encoding for characters
        # - check_flags :

        self.path_database_dir = path_database_dir
        self.database_name = database_name
        self.database_method = database_method
        self.input_type = input_type
        self.output_type = output_type
        self.encoding = encoding
        self.checks_flag = checks_flag

        self.path_settings = os.path.join(path_database_dir,
                                          'settings_' + database_name + '_' + self.database_method + '.json')

        mkdir_if_not_exist(path_database_dir)

        self.settings_keys = ['database_method', 'input_type', 'output_type', 'encoding']
        self.settings_values = [self.database_method, self.input_type, self.output_type, self.encoding]

        if os.path.isfile(self.path_settings):
            settings = dict_load_json(self.path_settings)
            if len(self.settings_keys) == len(settings['settings_keys']):
                for i in range(len(self.settings_keys)):
                    if settings['settings_values'][i] != self.settings_values[i] or settings['settings_keys'][i] != \
                            self.settings_keys[i]:
                        raise ValueError(
                            'saved settings dictionary does not correspond to the settings passed for this database')
            else:
                raise ValueError(
                    'saved settings dictionary does not correspond to the settings passed for this database')
        else:
            self.settings = {}

            for i in range(len(self.settings_keys)):
                key = self.settings_keys[i]
                value = self.settings_values[i]
                self.settings[key] = value

            self.settings['settings_keys'] = self.settings_keys
            self.settings['settings_values'] = self.settings_values

            self.save_settings()

        if self.database_method == 'lsm':
            self.path_database = os.path.join(path_database_dir, database_name + '.ldb')
            self.db = LSM(self.path_database)

        elif self.database_method == 'json':
            # only allows data types that can be converted to string
            self.path_database = os.path.join(path_database_dir, database_name + '.json')
            list_allowed_types = ['string', 'int', 'float']
            if self.input_type not in list_allowed_types:
                raise ValueError('input type not in allowed list', self.input_type)

            if self.output_type not in list_allowed_types:
                raise ValueError('output type not in allowed list', self.output_type)

            if os.path.isfile(self.path_database):
                print('load database at: ' + self.path_database)
                self.db = database_load_json(self.path_database, self.encoding)
            else:
                self.db = {}
        else:
            raise ValueError('database_method is not in options', self.database_method)

    def store_item(self, key, value):
        if self.checks_flag:
            if not check_variable_type(key, self.input_type):
                raise ValueError('type of key does not correspond to database', type(key), key)

            if not check_variable_type(value, self.output_type):
                raise ValueError('type of value does not correspond to database', type(value), value)

        if self.database_method == 'lsm':
            if self.output_type == 'list_str':
                self.db[key] = json.dumps(value)
            else:
                self.db[key] = value
        elif self.database_method == 'json':
            self.db[str(key)] = value
        else:
            raise ValueError('database_method is not in options', self.database_method)

    def save_db(self):
        if self.database_method == 'json':
            database_save_json(self.db, self.path_database, self.encoding)
            self.db = database_load_json(self.path_database, self.encoding)
        else:
            raise ValueError('cannot save database for database method', self.database_method)

    def get_item(self, key):
        if self.checks_flag:
            if not check_variable_type(key, self.input_type):
                raise ValueError('type of key does not correspond to database', type(key), key)

        if self.database_method == 'lsm':
            if self.output_type == 'list_str':
                return json.loads(self.db[key])
            else:
                value = self.db[key]
                return self.convert_type(value, self.output_type)
        elif self.database_method == 'json':
            value = self.db[str(key)]
            return value
        else:
            raise ValueError('database_method is not in options', self.database_method)

    def convert_type(self, value, type_value):
        if type_value == 'string':
            return value.decode(self.encoding)
        elif type_value == 'int':
            return int(value)
        elif type_value == 'float':
            return float(value)
        else:
            raise ValueError('type_value not in options', type_value)

    def save_settings(self):
        dict_save_json(self.settings, self.path_settings)

def check_variable_type(variable, type_reference):
    if type_reference == 'string':
        if type(variable) == str:
            return True
        else:
            return False
    elif type_reference == 'int':
        if type(variable) == int:
            return True
        else:
            return False
    elif type_reference == 'float':
        if type(variable) == float:
            return True
        else:
            return False
    elif type_reference == 'list_str':
        if type(variable) == list:
            return True
        else:
            return False
    else:
        raise ValueError('Type is not in options', type(variable), variable)