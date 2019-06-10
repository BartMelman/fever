#!/usr/bin/env python3


import os
import json
import pickle 

def num_files_in_directory(path_directory):
    # description: find the number of files in directory
    # input: path to directory
    # output: number of files in directory

    number_of_files = len([item for item in os.listdir(path_directory) if os.path.isfile(os.path.join(path_directory, item))])

    return number_of_files

def load_jsonl(filename):
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
      