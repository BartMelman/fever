from wiki_database import WikiDatabaseSqlite
from utils_db import mkdir_if_not_exist
import config
import os

import os

from utils_db import load_jsonl

import config
from tqdm import tqdm


from sqlitedict import SqliteDict
import sqlite3
from utils_db import mkdir_if_not_exist

def get_claim_dict_stage_3(claim_dict, wiki_database, claim_nr):
    method_tokenization = 'tokenize_text_pos'
    
    sentence_dict_list = []
    sentence_dict_total = {}
    list_old_proofs = []
    
    for interpreter in claim_dict['evidence']:
        sentence_dict = {}
        tmp_proof_list = []
        for proof in interpreter:
            title = proof[2]
            if title is not None:
                normalised_title = normalise_text(title)
                line_nr = proof[3]
                tmp_proof_list.append(title + str(line_nr))
                evidence_sentence = wiki_database.get_line_from_title(normalised_title, line_nr)
                if normalised_title in sentence_dict:
                    sentence_dict[normalised_title].append(line_nr)
                else:
                    sentence_dict[normalised_title] = [line_nr]
                if normalised_title in sentence_dict_total:
                    sentence_dict_total[normalised_title].append(line_nr)
                else:
                    sentence_dict_total[normalised_title] = [line_nr]
        proof_str =  '' + ' '.join(sorted(tmp_proof_list))           
        if proof_str not in list_old_proofs:
            sentence_dict_list.append(sentence_dict)
            list_old_proofs.append(proof_str)
           
    text_claim = normalise_text(claim_dict['claim'])
    tag_list_claim, word_list_claim = get_word_tag_list_from_text(text_str = text_claim, nlp = claim_database.nlp, method_tokenization_str = method_tokenization)
    
    list_correct_observations = []
    list_nei_observations = []
    
    # --- iterate over interpreters --- #
    interpreters_nr = 0
    for sentence_dict in sentence_dict_list:
        
        correct_dict_list = []
        potential_dict_list = []
        old_processed_claims = []
        
        # --- iterate over different documents --- #
        for title, sentences_correct_list in sentence_dict.items():
            for line_nr in sentences_correct_list:
                # get the sentences and a list of 5 alternatives for every document
                # create selection of proof + random select other sentences
                # correct
                dict_tmp = {}
                text_hypothesis = wiki_database.get_line_from_title(title, line_nr)
                tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis, nlp = claim_database.nlp, method_tokenization_str = method_tokenization)
                dict_tmp['hypotheses'] = word_list_hypotheses
                dict_tmp['hypotheses_tags'] = tag_list_hypotheses
                correct_dict_list.append(dict_tmp)
            
            # incorrect
            lines_file = wiki_database.get_lines_list_from_title(title)
            cosine_distance_list = []
            for line_nr in range(len(lines_file)):
                text_line = lines_file[line_nr]
            
                if ( len(text_line)>4 ) and ( line_nr not in sentence_dict_total[title] ):
                    cosine_distance = get_cosine(text_claim, text_line)
                else:
                    cosine_distance = 0 
                cosine_distance_list.append(cosine_distance)
            index_list = get_indices_top_K_values_list(cosine_distance_list, min(9, len(lines_file)))
            for index in index_list:
                dict_tmp = {}
                text_hypothesis_incorrect = lines_file[index]
                tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis_incorrect, nlp = claim_database.nlp, method_tokenization_str = method_tokenization)
                dict_tmp['hypotheses'] = word_list_hypotheses
                dict_tmp['hypotheses_tags'] = tag_list_hypotheses
                potential_dict_list.append(dict_tmp)
        
        nr_correct_sentences = len(correct_dict_list)
        nr_random_sentences = len(potential_dict_list)
        # --- only add correct claim if enough random sentences --- #
        if nr_correct_sentences + nr_random_sentences >= 5:
            correct_generated_observation_flag = True
            indices_selected_list = random.sample(range(nr_random_sentences), 5-nr_correct_sentences)
            
            combination_list_correct = correct_dict_list
            
            for index in indices_selected_list:
                combination_list_correct.append(potential_dict_list[index])
                
            shuffle(combination_list_correct)
        else:
            correct_generated_observation_flag = False
            
        # --- only add not enough info if enough random sentences --- #
        if nr_random_sentences >= 5:
            random_generated_observation_flag = True
            indices_selected_list = random.sample(range(nr_random_sentences), 5)
            
            combination_list_random = []
            for index in indices_selected_list:
                combination_list_random.append(potential_dict_list[index])
                
            shuffle(combination_list_random)
        else:
            random_generated_observation_flag = False
        # --- process --- #
        # --- correct --- #
        if correct_generated_observation_flag == True:
            dict_out = {}
            dict_out['premises'] = word_list_claim
            dict_out['ids'] = str(claim_dict['id']) + '_' + str(claim_nr) + '_' + str(interpreters_nr) + '_correct'
            dict_out['labels'] = claim_dict['label']
            dict_out['primises_tags'] = tag_list_claim
            dict_out['hypotheses'] = []
            dict_out['hypotheses_tags'] = []

            for tmp_dict in combination_list_correct:           
                dict_out['hypotheses'] += dict_tmp['hypotheses']
                dict_out['hypotheses_tags'] += dict_tmp['hypotheses_tags']
            list_correct_observations.append(dict_out)
    
        # --- random --- #
        if random_generated_observation_flag == True:
            dict_out = {}
            dict_out['premises'] = word_list_claim
            dict_out['ids'] = str(claim_dict['id']) + '_' + str(claim_nr) + '_' + str(interpreters_nr) + '_random'
            dict_out['labels'] = claim_dict['label']
            dict_out['primises_tags'] = tag_list_claim
            dict_out['hypotheses'] = []
            dict_out['hypotheses_tags'] = []

            for tmp_dict in combination_list_correct:           
                dict_out['hypotheses'] += dict_tmp['hypotheses']
                dict_out['hypotheses_tags'] += dict_tmp['hypotheses_tags']    
            list_nei_observations.append(dict_out)
            
        interpreters_nr += 1
        
    return list_correct_observations, list_nei_observations

def get_indices_top_K_values_list(input_list, K):
    # description: return the indices of the highest K values of a list
    tmp = [value for value in input_list]
    input_list.sort()
    return [tmp.index(input_list[-i]) for i in range(1, K+1) if input_list[-i]>0]

# save observation, n correct, n closest
from utils_wiki_database import normalise_text

def get_claim_dict_stage_2(claim_dict, wiki_database, claim_nr):
    method_tokenization = 'tokenize_text_pos'
    
    sentence_dict = {}
    print(claim_dict['evidence'])
    for interpreter in claim_dict['evidence']:
        for proof in interpreter:
            title = proof[2]
            if title is not None:
                normalised_title = normalise_text(title)
                line_nr = proof[3]
                evidence_sentence = wiki_database.get_line_from_title(normalised_title, line_nr)
                if normalised_title in sentence_dict:
                    sentence_dict[normalised_title].append(line_nr)
                else:
                    sentence_dict[normalised_title] = [line_nr]
                    
    correct_evidence_list = []
    incorrect_evidence_list = []
    interpreters_nr = 0
    for title, sentences_correct_list in sentence_dict.items():
        sentences_correct_list = list(set(sentences_correct_list)) # remove duplicates
        for line_nr in sentences_correct_list:
            # correct
            dict_out = {}
            text_claim = normalise_text(claim_dict['claim'])
            tag_list_claim, word_list_claim = get_word_tag_list_from_text(text_str = text_claim, nlp = claim_database.nlp, method_tokenization_str = method_tokenization)
            dict_out['premises'] = word_list_claim
            dict_out['ids'] = str(claim_dict['id']) + '_' + str(claim_nr) + '_' + str(interpreters_nr) + '_correct'
            dict_out['labels'] = claim_dict['label']
            dict_out['primises_tags'] = tag_list_claim
            text_hypothesis = wiki_database.get_line_from_title(title, line_nr)
            tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis, nlp = claim_database.nlp, method_tokenization_str = method_tokenization)
            dict_out['hypotheses'] = word_list_hypotheses
            dict_out['hypotheses_tags'] = tag_list_hypotheses
            
            correct_evidence_list.append(dict_out)
            # incorrect
            lines_file = wiki_database.get_lines_list_from_title(title)
            cosine_distance_list = []
            for i in range(len(lines_file)):
                text_line = lines_file[i]
                if len(text_line)>5 and (i not in sentences_correct_list):
                    cosine_distance = get_cosine(text_claim, text_line)
                else:
                    cosine_distance = 0 
                cosine_distance_list.append(cosine_distance)
            index_largest_cosine_distance = cosine_distance_list.index(max(cosine_distance_list))
            text_hypothesis_incorrect = lines_file[index_largest_cosine_distance]
            
            dict_out['premises'] = word_list_claim
            dict_out['ids'] = str(claim_dict['id']) + '_' + str(claim_nr) + '_' + str(interpreters_nr) + '_random'
            dict_out['labels'] = claim_dict['label']
            dict_out['primises_tags'] = tag_list_claim
            tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis_incorrect, nlp = claim_database.nlp, method_tokenization_str = method_tokenization)
            dict_out['hypotheses'] = word_list_hypotheses
            dict_out['hypotheses_tags'] = tag_list_hypotheses
            incorrect_evidence_list.append(dict_out)
        interpreters_nr += 1
    return correct_evidence_list, incorrect_evidence_list

import random
from random import shuffle
import spacy

from utils_db import mkdir_if_not_exist, dict_load_json, dict_save_json

class ClaimDatabaseStage2:
    def __init__(self, path_database_dir, path_raw_data, method_false_statements, fraction_validation):
        # description: create a database in which the 
        
        # folder layout:
        # base_folder_name
        # - settings.json
        # - train
            # - evidence_correct
            # - evidence_incorrect
            # - combined
        # - validation
            # - evidence_correct
            # - evidence_incorrect
            # - combined 
        # - dev
            # - evidence_correct
            # - evidence_incorrect
            # - combined
        
        self.path_database_dir = path_database_dir
        self.path_raw_data = path_raw_data
        self.method_false_statements = method_false_statements
        self.fraction_validation = fraction_validation
        
        self.total_nr_claims_train_set = self.get_nr_claims(data_set_type = 'train')
        self.total_nr_claims_dev_set = self.get_nr_claims(data_set_type = 'dev')
#         self.nr_claims_training = 
#         self.nr_claims_validation = 
#         self.nr_claims_dev = 
        self.random_seed_nr = 1

        base_folder_name = 'claim_database'
        self.path_base_folder_dir = os.path.join(self.path_database_dir, base_folder_name)
        self.path_settings = os.path.join(self.path_base_folder_dir, 'settings.json')
        
        self.nlp = spacy.load('en', disable=["parser", "ner"])
        self.partition = self.get_partition_list(fraction_validation = fraction_validation)
        
#         if not os.path.isdir(self.path_base_folder_dir):
#             mkdir_if_not_exist(self.path_base_folder_dir)
#             self.settings = {}
            
            
#             data_set_type_list = ['training', 'validation', 'dev']
            
#             for data_set_type in data_set_type_list:
#                 data = self.get_data_dict(data_set_type)
#                 self.save_data_dict(self, data)
#                 data = None
                
#             dict_save_json(self.settings, self.path_settings)
#         else:
#             self.settings = dict_load_json(self.path_settings)

    def get_nr_claims(self, data_set_type):
        path_data_set = os.path.join(self.path_raw_data, data_set_type + '.jsonl')
        claim_dict_list = load_jsonl(path_data_set)
        nr_claims = len(claim_dict_list)
        return nr_claims
    
    def get_partition_list(self, fraction_validation):
        list_total_shuffled = list(range(self.total_nr_claims_train_set))
        random.seed(self.random_seed_nr)
        shuffle(list_total_shuffled)
        
        partition = {}
        partition['train'] = list_total_shuffled[int(self.total_nr_claims_train_set*fraction_validation):self.total_nr_claims_train_set]
        partition['validation'] = list_total_shuffled[0:int(self.total_nr_claims_train_set*fraction_validation)]
        partition['dev'] = list(range(self.total_nr_claims_dev_set))

        return partition
        
#     def create_database(self):
        
#     def get_data_dict(self, data_set_type):
#         # description
#         # input:
#         # - data_set_type: 'train', 'validation' or 'dev' [str]
        
#         path_data_type_dir = os.path.join(self.path_base_folder_dir, data_set_type)
        
#         return data
        
#     def save_data_dict(self, data):


import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

from wiki_database import Text

def get_word_tag_list_from_text(text_str, nlp, method_tokenization_str):
    doc = nlp(text_str)
    text = Text(doc)
    delimiter = text.delimiter_position_tag
    tokenized_text_list = text.process([method_tokenization_str])
    word_list = []
    tag_list = []

    for i in range(len(tokenized_text_list)):
        key = tokenized_text_list[i]
        tag, word = key.split(delimiter)
        if i == 0:
            if not(tag == 'PROPN'):
                word = word.lower()
        word_list.append(word)
        tag_list.append(tag)
    
    return tag_list, word_list

if __name__ == '__main__':
	path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
	path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
	# mkdir_if_not_exist(path_wiki_database_dir)
	wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)

	path_dir_database = 'claim_db'
	path_raw_data = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR)
	claim_data_set = 'train'
	fraction_validation = 0.1

	claim_database = ClaimDatabaseStage2(path_dir_database, path_raw_data, claim_data_set, fraction_validation)

	data_set_type = 'train'
	path_data_set = os.path.join(claim_database.path_raw_data, data_set_type + '.jsonl')
	claim_dict_list = load_jsonl(path_data_set)


	dir_database = 'tmp'
	mkdir_if_not_exist(dir_database)
	path_stage_2_correct_db = os.path.join(dir_database, 'stage_2_correct.sqlite')
	path_stage_2_refuted_db = os.path.join(dir_database, 'stage_2_refuted.sqlite')
	path_stage_2_nei_db = os.path.join(dir_database, 'stage_2_nei.sqlite')

	nr_claims = 100

	with SqliteDict(path_stage_2_correct_db) as stage_2_correct_db:
	    with SqliteDict(path_stage_2_refuted_db) as stage_2_refuted_db:
	        with SqliteDict(path_stage_2_nei_db) as stage_2_nei_db:
	            for claim_nr in tqdm(range(0, nr_claims)):
	                claim_dict = claim_dict_list[claim_nr]
	                correct_evidence_list, incorrect_evidence_list = get_claim_dict_stage_2(
	                    claim_dict, wiki_database, claim_nr)
	                
	                label = claim_dict['label']
	                
	                if label == 'SUPPORTS':
	                    stage_2_correct_db[claim_nr] = correct_evidence_list
	                elif label  == 'REFUTES':
	                    stage_2_refuted_db[claim_nr] = correct_evidence_list
	                if len(incorrect_evidence_list) > 0:
	                    stage_2_nei_db[claim_nr] = incorrect_evidence_list

	                    
	            stage_2_nei_db.commit()
	        stage_2_refuted_db.commit()
	    stage_2_correct_db.commit()
	    