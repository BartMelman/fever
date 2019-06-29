#!/usr/bin/env python3


import os
import json
import pickle 

class Claim:
    def __init__(self, claim_dictionary):
        self.id = claim_dictionary['id']
        self.verifiable = claim_dictionary['verifiable']
        self.label = claim_dictionary['label']
        self.claim = claim_dictionary['claim']
        self.claim_without_dot = self.claim[:-1]
        self.evidence = claim_dictionary['evidence']
        # self.nlp = nlp

        # if 'docs_selected' in claim_dictionary:
        #     self.docs_selected = claim_dictionary['docs_selected']
    
class ClaimDocTokenizer:
    def __init__(self, doc):
        self.doc = doc
    def get_tokenized_claim(self, method_tokenization):
        # claim_without_dot = self.claim[:-1]  # remove . at the end
        # doc = self.nlp(claim_without_dot)
        text = Text(self.doc)
        tokenized_claim = text.process(method_tokenization)
        return tokenized_claim
    def get_n_grams(self, method_tokenization, n_gram):
        return count_n_grams(self.get_tokenized_claim(method_tokenization), n_gram, 'str')
        
def num_files_in_directory(path_directory):
    # description: find the number of files in directory
    # input: path to directory
    # output: number of files in directory

    number_of_files = len([item for item in os.listdir(path_directory) if os.path.isfile(os.path.join(path_directory, item))])

    return number_of_files

# def write_jsonl(filename, dic_list):
#     # description: only use for wikipedia dump
#     output_file = open(filename, 'w', encoding='utf-8')
#     for dic in dic_list:
#         json.dump(dic, output_file) 
#         output_file.write("\n")
        
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
      