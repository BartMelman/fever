import os

from utils_db import dict_load_json, dict_save_json

class Settings:
    def __init__(self, path_settings_dir, file_name = 'settings'):
        
        self.path_settings = os.path.join(path_settings_dir, file_name + '.json')
        
        if os.path.isfile(self.path_settings):
            self.settings_dict = dict_load_json(self.path_settings)
        else:
            self.settings_dict = {}
            self.save_settings()
    
    def check_function_flag(self, function_name, status):
        # description: 
        # input:
        # - function_name : name of function that we are going to run
        # - status: 'start' (a flag is saved which indicates that a function has started),
        #           'check' (check if a function has started or finished correctly, else an error is raised)
        #           'finish' (set a flag which indicates that a function has finished correctly)
        
        if status == 'start':
            if 'flag_function_status' not in self.settings_dict:
                self.settings_dict['flag_function_status'] = {}
                
            if function_name in self.settings_dict['flag_function_status']:
                raise ValueError('cannot start a flag if it already exists', self.settings_dict['flag_function_status'][function_name])
            else:
                self.settings_dict['flag_function_status'][function_name] = False
                self.save_settings()

        elif status == 'check':
            # the three options are: (1) return 'not_started_yet', (2) return 'finished_correctly', (3) raise ValueError
            if 'flag_function_status' not in self.settings_dict:
                return 'not_started_yet'
            else:
                if function_name not in self.settings_dict['flag_function_status']:
                    return 'not_started_yet'
                else:
                    if self.settings_dict['flag_function_status'][function_name] == True:
                        return 'finished_correctly'
                    elif self.settings_dict['flag_function_status'][function_name] == False:
                        raise ValueError('function was terminated too early', function_name)
                    else:
                        raise ValueError('the stored value should be True or False', self.settings_dict['flag_function_status'][function_name])

        elif status == 'finish':
            if 'flag_function_status' not in self.settings_dict:
                raise ValueError('cannot call the finish status if a function has not started yet')
            else:
                if function_name not in self.settings_dict['flag_function_status']:
                    raise ValueError('cannot call the finish status if a function has not started yet')
                else:
                    if self.settings_dict['flag_function_status'][function_name] == False:
                        self.settings_dict['flag_function_status'][function_name] = True
                        self.save_settings()
                    else:
                        raise ValueError('flag status should be started (False), but that is not the case', self.settings_dict['flag_function_status'][function_name])
        else:
            raise ValueError('status not in options', status)
    
    def add_item(self, key, value):
        self.settings_dict[key] = value
        self.save_settings()
    
    def get_item(self, key):
        return self.settings_dict[key]
    
    def save_settings(self):
        dict_save_json(self.settings_dict, self.path_settings)     
        
