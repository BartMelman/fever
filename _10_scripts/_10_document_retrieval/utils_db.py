import os
import json
import pickle 
import sys


def create_path_dictionary(list_elements, dictionary):
    first_element = list_elements[0]
    if first_element not in dictionary:
        dictionary[first_element] = {}

    if len(list_elements) <= 1:
        return dictionary
    else:
        dictionary[first_element] = create_path_dictionary(list_elements[1:], dictionary[first_element])
        return dictionary


def get_file_name_from_variable_list(variable_list, delimiter = '_'):
    file_name = ''
    for i in range(len(variable_list)):
        variable = variable_list[i]
        if i != len(variable_list)-1:
            file_name += str(variable) + delimiter
        else:
            file_name += str(variable)
    return file_name

def mkdir_if_not_exist(path_dir):        
    try:
        os.makedirs(path_dir, exist_ok=True)
    except FileExistsError:
        print('folder already exists:', path_dir)
        
def num_files_in_directory(path_directory):
    # description: find the number of files in directory
    # input: path to directory
    # output: number of files in directory

    number_of_files = len([item for item in os.listdir(path_directory) if os.path.isfile(os.path.join(path_directory, item))])

    return number_of_files
        
def write_jsonl(filename, file):
    # description: write a list of dictionaries in utf-8 format
    with open(filename, encoding='utf-8', mode='w') as outfile:
        for line in file:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write("\n")

def load_jsonl(filename):
    # description: only use for wikipedia dump
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
#         print("Load Jsonl:", filename)
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list

def save_dict_pickle(dictionary, path_file):
    if os.path.isfile(path_file):
        print('overwriting file: %s'%(path_file))
    with open(path_file, "wb") as file:
        pickle.dump(dictionary, file)
        
def load_dict_pickle(path_file):
    if os.path.isfile(path_file):
        with open("test", "rb") as file:
            dictionary=pickle.load(file)
    else:
        raise ValueError('json file does not exist', path_file)
    return dictionary

def dict_save_json(dictionary, path_file):
    if os.path.isfile(path_file):
        print('overwriting file: %s'%(path_file))
    with open(path_file, "w") as f:
        json.dump(dictionary, f)

def dict_load_json(path_file):
    if os.path.isfile(path_file):
        with open(path_file, "r") as f:
            dictionary = json.load(f)
    else:
        raise ValueError('json file does not exist', path_file)
    return dictionary

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def database_save_json(dictionary, path_file, encoding):
    if os.path.isfile(path_file):
        print('overwriting file: %s'%(path_file))
    with open(path_file, encoding=encoding, mode="w") as f:
        json.dump(dictionary, f, ensure_ascii=False)

def database_load_json(path_file, encoding):
    if os.path.isfile(path_file):
        with open(path_file, encoding=encoding, mode='r') as f:
            dictionary = json.load(f)
    else:
        raise ValueError('json file does not exist', path_file)
    return dictionary
        
# def save_dict_2_json(dictionary, path_file):
#     if os.path.isfile(path_file):
#         print('overwriting file: %s'%(path_file))
#     with open(path_file, "w") as f:
#         k = dictionary.keys() 
#         v = dictionary.values() 
#         k1 = [str(i) for i in k]
#         json.dump(json.dumps(dict(zip(*[k1, v]))),f) 
        
# def load_dict_from_json(path_file):
#     if os.path.isfile(path_file):
#         with open(path_file, 'r') as f:
#             data = json.load(f)
#             dic = json.loads(data)
#             k = dic.keys() 
#             v = dic.values() 
#             k1 = [eval(i) for i in k] 
#             return dict(zip(*[k1,v]))  
#     else:
#         raise ValueError('json file does not exist', path_file)
      