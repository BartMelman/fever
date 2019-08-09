import os
import spacy
import random
from random import shuffle
from sqlitedict import SqliteDict
import sqlite3   
from tqdm import tqdm
import pickle 

from utils_db import dict_load_json, dict_save_json, mkdir_if_not_exist, load_jsonl
from utils_doc_results_db import get_tag_2_id_dict_unigrams

class ClaimDatabaseStage_2_3:
    def __init__(self, path_database_dir, path_raw_data, fraction_validation, method_combination, wiki_database):
        # description: create a database in which the 
        
        # folder layout:n
        # base_folder_name
        # - settings.json
        # - databases
            # - train
                # - 
            # - validation
                # - 
            # - test
                # - 
        
        self.path_database_dir = path_database_dir
        self.path_raw_data = path_raw_data
        self.fraction_validation = fraction_validation
        self.method_combination = method_combination
        
        base_folder_name = 'claim_database'
        self.path_base_folder_dir = os.path.join(self.path_database_dir, base_folder_name)
        mkdir_if_not_exist(self.path_base_folder_dir)
        self.path_settings = os.path.join(self.path_base_folder_dir, 'settings.json')
        self.path_databases_dir = os.path.join(self.path_base_folder_dir, 'databases')
        self.pickle_files_dir = os.path.join(self.path_base_folder_dir, 'pickle_files')
        mkdir_if_not_exist(self.pickle_files_dir)
        mkdir_if_not_exist(self.path_databases_dir)

        print('ClaimDatabaseStage2')
        if os.path.isfile(self.path_settings):
            self.settings = dict_load_json(self.path_settings)
        else:
            self.settings = {}
        
        # --- set the paths for the databases --- #
        stage_list = ['stage_2'] # ['stage_2', 'stage_3']
        data_set_type_list = ['train', 'validation', 'dev']
        label_list = ['correct', 'refuted', 'nei']
        self.path_db = {}
        for stage in stage_list:
            if stage not in self.path_db:
                self.path_db[stage] = {}
            for label in label_list:
                if label not in self.path_db[stage]:
                    self.path_db[stage][label] = {}
                for data_set_type in data_set_type_list:
                    self.path_db[stage][label][data_set_type] = os.path.join(
                        self.path_databases_dir, 
                        stage + '_' + data_set_type + '_' + label + '.sqlite')

        # --- set paths for pickle file --- #
        method_combination_list = ['max_correct', 'one_from_every_claim', 'equal']
        
        self.path_pickle = {}
        for stage in stage_list:
            if stage not in self.path_pickle:
                self.path_pickle[stage] = {}
            for data_set_type in data_set_type_list:
                if data_set_type not in self.path_pickle[stage]:
                    self.path_pickle[stage][data_set_type] = {}
                for method_combination in method_combination_list:
                    self.path_pickle[stage][data_set_type][method_combination] = os.path.join(
                        self.pickle_files_dir, 
                        stage + '_' + data_set_type + '_' + method_combination + "_data.pkl")

        # --- get total count in data sets --- #
        if 'total_nr_claims_train_set' in self.settings:
            self.total_nr_claims_train_set = self.settings['total_nr_claims_train_set']
        else:
            print('- total_nr_claims_train_set')
            self.total_nr_claims_train_set = self.get_nr_claims(data_set_type = 'train')
            self.settings['total_nr_claims_train_set'] = self.total_nr_claims_train_set
            self.save_settings()
            
        if 'total_nr_claims_dev_set' in self.settings:
            self.total_nr_claims_dev_set = self.settings['total_nr_claims_dev_set']
        else:
            print('- total_nr_claims_dev_set')
            self.total_nr_claims_dev_set = self.get_nr_claims(data_set_type = 'dev')
            self.settings['total_nr_claims_dev_set'] = self.total_nr_claims_train_set
            self.save_settings()

        self.random_seed_nr = 1       
        self.nlp = spacy.load('en', disable=["parser", "ner"])
        
        if 'partition' in self.settings:
            self.partition = self.settings['partition']
        else:
            print('- get partition')
            self.partition = self.get_partition_list(fraction_validation = fraction_validation)
            self.settings['partition'] = self.partition
            self.save_settings()
        
        self.nr_claims_train = self.settings['nr_claims_train']
        self.nr_claims_validation = self.settings['nr_claims_validation']
        self.nr_claims_dev = self.settings['nr_claims_dev']
        
        
        # --- create database train and validation set --- #
        claim_dict_list = self.get_raw_data(data_set_type = 'train')
        
        for stage in stage_list:
            for dataset_type in ['train', 'validation']:
                self.create_database(wiki_database = wiki_database,
                                     dataset_type = dataset_type, 
                                     claim_dict_list = claim_dict_list, 
                                     partition = self.settings['partition'][dataset_type], 
                                     stage = stage)
        
        # --- create database dev set --- #
        claim_dict_list = self.get_raw_data(data_set_type = 'dev')
        
        for stage in stage_list:
            self.create_database(wiki_database = wiki_database,
                                 dataset_type = 'dev', 
                                 claim_dict_list = claim_dict_list, 
                                 partition = self.settings['partition']['dev'], 
                                 stage = stage)
        
        # --- save in pickle format for stage 2 and 3 and all datasets --- #
        for stage in stage_list:
            for dataset_type in data_set_type_list:
                self.save_2_pickle(dataset_type = dataset_type, 
                                   method_combination = self.method_combination, 
                                   stage = stage) 
 
    def get_raw_data(self, data_set_type):
        path_data_set = os.path.join(self.path_raw_data, data_set_type + '.jsonl')
        return load_jsonl(path_data_set)
    
    def save_2_pickle(self, dataset_type, method_combination, stage):
        path_save = self.path_pickle[stage][dataset_type][method_combination]
#         dict_out['premises'] = word_list_claim
#         dict_out['ids'] = str(claim_dict['id']) + '_' + str(claim_nr) + '_' + str(interpreters_nr) + '_correct'
#         dict_out['labels'] = claim_dict['label']
#         dict_out['primises_tags'] = tag_list_claim
#         text_hypothesis = wiki_database.get_line_from_title(title, line_nr)
#         tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis, nlp = claim_database.nlp, method_tokenization_str = method_tokenization)
#         dict_out['hypotheses'] = word_list_hypotheses
#         dict_out['hypotheses_tags'] = tag_list_hypotheses
        if os.path.isfile(path_save):
            print('pickle file already exists: ' + dataset_type + '_' + method_combination)
        else:
            if method_combination == 'equal':
                min_nr_observations = min(self.settings[stage][dataset_type]['nr_supports'], 
                                          self.settings[stage][dataset_type]['nr_refuted'],
                                          self.settings[stage][dataset_type]['nr_nei'])
                path_stage_2_correct_db = self.path_db[stage]['correct'][dataset_type]
                path_stage_2_refuted_db = self.path_db[stage]['refuted'][dataset_type]
                path_stage_2_nei_db = self.path_db[stage]['nei'][dataset_type]

                stage_2_correct_db = SqliteDict(path_stage_2_correct_db)
                stage_2_refuted_db = SqliteDict(path_stage_2_refuted_db)
                stage_2_nei_db = SqliteDict(path_stage_2_nei_db)

                correct_keys_list = list(stage_2_correct_db.keys())
                refuted_keys_list = list(stage_2_refuted_db.keys())
                nei_keys_list = list(stage_2_nei_db.keys())

                ids = []
                premises = []
                hypotheses = []
                premises_part_of_speech = []
                premises_out_of_vocabulary = []
                hypotheses_part_of_speech = []
                hypotheses_out_of_vocabulary = []
                labels = []
                
                for idx in tqdm(range(min_nr_observations), desc = 'save pickle'):
                    # update correct
                    key_correct = correct_keys_list[idx]
                    key_refuted = refuted_keys_list[idx]
                    key_nei = nei_keys_list[idx]
                    
                    for method in ['correct', 'refuted', 'nei']:
                        if method == 'correct':
                            claim_dict = stage_2_correct_db[key_correct][0]
                        elif method == 'refuted':
                            claim_dict = stage_2_refuted_db[key_refuted][0]
                        elif method == 'nei':
                            claim_dict = stage_2_nei_db[key_nei][0]
                        else:
                            raise ValueError('method not in list', method)
                            
                        ids.append(claim_dict['ids'])
                        premises.append(claim_dict['premises'])
                        premises_part_of_speech.append(claim_dict["premises_tags"])
                        ids_hypothesis, ids_evidence = hypothesis_evidence_2_index(hypothesis = claim_dict['hypotheses'], 
                                                                                 premise = claim_dict['premises'], 
                                                                                 max_length = 75, 
                                                                                 randomise_flag = True)
                        premises_out_of_vocabulary.append(ids_evidence)
                        hypotheses.append(claim_dict['hypotheses'])
                        hypotheses_part_of_speech.append(claim_dict["hypotheses_tags"])
                        hypotheses_out_of_vocabulary.append(ids_hypothesis)
                        if method == 'nei':
                            labels.append('NOT ENOUGH INFO')
                        else:
                            labels.append(claim_dict['labels'])
            # ==== #
            
            elif method_combination == 'max_correct':
                path_stage_2_correct_db = self.path_db[stage]['correct'][dataset_type]
                path_stage_2_refuted_db = self.path_db[stage]['refuted'][dataset_type]
                path_stage_2_nei_db = self.path_db[stage]['nei'][dataset_type]

                stage_2_correct_db = SqliteDict(path_stage_2_correct_db)
                stage_2_refuted_db = SqliteDict(path_stage_2_refuted_db)
                stage_2_nei_db = SqliteDict(path_stage_2_nei_db)

                correct_keys_list = list(stage_2_correct_db.keys())
                refuted_keys_list = list(stage_2_refuted_db.keys())
                nei_keys_list = list(stage_2_nei_db.keys())

                ids = []
                premises = []
                hypotheses = []
                premises_part_of_speech = []
                premises_out_of_vocabulary = []
                hypotheses_part_of_speech = []
                hypotheses_out_of_vocabulary = []
                labels = []
                
                for idx in tqdm(range(self.settings[stage][dataset_type]['nr_supports']), 
                                desc='save pickle_correct_' + dataset_type):
                    # update correct
                    key_correct = correct_keys_list[idx]
                    claim_dict = stage_2_correct_db[key_correct][0]
                    ids.append(claim_dict['ids'])
                    premises.append(claim_dict['premises'])
                    premises_part_of_speech.append(claim_dict["premises_tags"])
                    ids_hypothesis, ids_evidence = hypothesis_evidence_2_index(hypothesis = claim_dict['hypotheses'], 
                                                                             premise = claim_dict['premises'], 
                                                                             max_length = 75, 
                                                                             randomise_flag = True)
                    premises_out_of_vocabulary.append(ids_evidence)
                    hypotheses.append(claim_dict['hypotheses'])
                    hypotheses_part_of_speech.append(claim_dict["hypotheses_tags"])
                    hypotheses_out_of_vocabulary.append(ids_hypothesis)
                    labels.append(claim_dict['labels'])
                idx_refuted = 0
                for idx_refuted in tqdm(range(self.settings[stage][dataset_type]['nr_supports']), 
                                desc='save pickle_refuted_' + dataset_type):
#                     for idx in tqdm(range(self.settings[stage][dataset_type]['nr_refuted']), 
#                                     desc='save pickle_refuted_' + dataset_type):
                        # update refuted
                        idx_tmp = idx_refuted % self.settings[stage][dataset_type]['nr_refuted']
                        key_refuted = refuted_keys_list[idx_tmp]
                        claim_dict = stage_2_refuted_db[key_refuted][0]
                        ids.append(claim_dict['ids'])
                        premises.append(claim_dict['premises'])
                        premises_part_of_speech.append(claim_dict["premises_tags"])
                        ids_hypothesis, ids_evidence = hypothesis_evidence_2_index(hypothesis = claim_dict['hypotheses'], 
                                                                                 premise = claim_dict['premises'], 
                                                                                 max_length = 75, 
                                                                                 randomise_flag = True)
                        premises_out_of_vocabulary.append(ids_evidence)
                        hypotheses.append(claim_dict['hypotheses'])
                        hypotheses_part_of_speech.append(claim_dict["hypotheses_tags"])
                        hypotheses_out_of_vocabulary.append(ids_hypothesis)
                        labels.append(claim_dict['labels'])
                    
                for idx in tqdm(range(self.settings[stage][dataset_type]['nr_supports']), desc='save pickle_nei_'+dataset_type):
                    # update not enough info
                    key_nei = nei_keys_list[idx]
                    claim_dict = stage_2_nei_db[key_nei][0]
                    ids.append(claim_dict['ids'])
                    premises.append(claim_dict['premises'])
                    premises_part_of_speech.append(claim_dict["premises_tags"])
                    ids_hypothesis, ids_evidence = hypothesis_evidence_2_index(hypothesis = claim_dict['hypotheses'], 
                                                                             premise = claim_dict['premises'], 
                                                                             max_length = 75, 
                                                                             randomise_flag = True)
                    premises_out_of_vocabulary.append(ids_evidence)
                    hypotheses.append(claim_dict['hypotheses'])
                    hypotheses_part_of_speech.append(claim_dict["hypotheses_tags"])
                    hypotheses_out_of_vocabulary.append(ids_hypothesis)
                    labels.append(claim_dict['labels'])
                    
            # ==== #
            elif method_combination == 'one_from_every_claim':
                path_stage_2_correct_db = self.path_db[stage]['correct'][dataset_type]
                path_stage_2_refuted_db = self.path_db[stage]['refuted'][dataset_type]
                path_stage_2_nei_db = self.path_db[stage]['nei'][dataset_type]

                stage_2_correct_db = SqliteDict(path_stage_2_correct_db)
                stage_2_refuted_db = SqliteDict(path_stage_2_refuted_db)
                stage_2_nei_db = SqliteDict(path_stage_2_nei_db)

                correct_keys_list = list(stage_2_correct_db.keys())
                refuted_keys_list = list(stage_2_refuted_db.keys())
                nei_keys_list = list(stage_2_nei_db.keys())

                ids = []
                premises = []
                hypotheses = []
                premises_part_of_speech = []
                premises_out_of_vocabulary = []
                hypotheses_part_of_speech = []
                hypotheses_out_of_vocabulary = []
                labels = []
                
                for idx in tqdm(range(self.settings[stage][dataset_type]['nr_refuted']), desc='save pickle_refuted_'+dataset_type):
                    # update refuted
                    key_refuted = refuted_keys_list[idx]
                    claim_dict = stage_2_refuted_db[key_refuted][0]
                    ids.append(claim_dict['ids'])
                    premises.append(claim_dict['premises'])
                    premises_part_of_speech.append(claim_dict["premises_tags"])
                    ids_hypothesis, ids_evidence = hypothesis_evidence_2_index(hypothesis = claim_dict['hypotheses'], 
                                                                             premise = claim_dict['premises'], 
                                                                             max_length = 75, 
                                                                             randomise_flag = True)
                    premises_out_of_vocabulary.append(ids_evidence)
                    hypotheses.append(claim_dict['hypotheses'])
                    hypotheses_part_of_speech.append(claim_dict["hypotheses_tags"])
                    hypotheses_out_of_vocabulary.append(ids_hypothesis)
                    labels.append(claim_dict['labels'])
                for idx in tqdm(range(self.settings[stage][dataset_type]['nr_nei']), desc='save pickle_nei_'+dataset_type):
                    # update not enough info
                    key_nei = nei_keys_list[idx]
                    claim_dict = stage_2_nei_db[key_nei][0]
                    ids.append(claim_dict['ids'])
                    premises.append(claim_dict['premises'])
                    premises_part_of_speech.append(claim_dict["premises_tags"])
                    ids_hypothesis, ids_evidence = hypothesis_evidence_2_index(hypothesis = claim_dict['hypotheses'], 
                                                                             premise = claim_dict['premises'], 
                                                                             max_length = 75, 
                                                                             randomise_flag = True)
                    premises_out_of_vocabulary.append(ids_evidence)
                    hypotheses.append(claim_dict['hypotheses'])
                    hypotheses_part_of_speech.append(claim_dict["hypotheses_tags"])
                    hypotheses_out_of_vocabulary.append(ids_hypothesis)
                    labels.append(claim_dict['labels'])
                for idx in tqdm(range(self.settings[stage][dataset_type]['nr_nei'] - self.settings[stage][dataset_type]['nr_refuted']), desc='save pickle_correct_'+dataset_type):
                    # update correct
                    key_correct = correct_keys_list[idx]
                    claim_dict = stage_2_correct_db[key_correct][0]
                    ids.append(claim_dict['ids'])
                    premises.append(claim_dict['premises'])
                    premises_part_of_speech.append(claim_dict["premises_tags"])
                    ids_hypothesis, ids_evidence = hypothesis_evidence_2_index(hypothesis = claim_dict['hypotheses'], 
                                                                             premise = claim_dict['premises'], 
                                                                             max_length = 75, 
                                                                             randomise_flag = True)
                    premises_out_of_vocabulary.append(ids_evidence)
                    hypotheses.append(claim_dict['hypotheses'])
                    hypotheses_part_of_speech.append(claim_dict["hypotheses_tags"])
                    hypotheses_out_of_vocabulary.append(ids_hypothesis)
                    labels.append(claim_dict['labels'])

#             c = list(zip(ids, premises, hypotheses, premises_part_of_speech, premises_out_of_vocabulary, hypotheses_part_of_speech, hypotheses_out_of_vocabulary, labels))
#             random.shuffle(c)

#             ids, premises, hypotheses, premises_part_of_speech, premises_out_of_vocabulary, hypotheses_part_of_speech, hypotheses_out_of_vocabulary, labels = zip(*c)

#             target_dict = {"ids": ids,
#                 "premises": premises,
#                 "hypotheses": hypotheses,
#                 "premises_part_of_speech": premises_part_of_speech,
#                 "premises_out_of_vocabulary": premises_out_of_vocabulary,
#                 "hypotheses_part_of_speech": hypotheses_part_of_speech,
#                 "hypotheses_out_of_vocabulary": hypotheses_out_of_vocabulary,           
#                 "labels": labels}
            
            c = list(zip(ids, premises, hypotheses, premises_part_of_speech, premises_out_of_vocabulary, hypotheses_part_of_speech, hypotheses_out_of_vocabulary, labels))
            random.shuffle(c)

            ids, premises, hypotheses, premises_part_of_speech, premises_out_of_vocabulary, hypotheses_part_of_speech, hypotheses_out_of_vocabulary, labels = zip(*c)

            target_dict = {"ids": ids,
                "premises": premises,
                "hypotheses": hypotheses,
                "premises_part_of_speech": premises_part_of_speech,
                "premises_out_of_vocabulary": premises_out_of_vocabulary,
                "hypotheses_part_of_speech": hypotheses_part_of_speech,
                "hypotheses_out_of_vocabulary": hypotheses_out_of_vocabulary,           
                "labels": labels}           

            with open(os.path.join(self.pickle_files_dir, path_save), "wb") as pkl_file:
                pickle.dump(target_dict, pkl_file)
            
    def save_settings(self):
        dict_save_json(self.settings, self.path_settings)
        
    def create_database(self, wiki_database, dataset_type, claim_dict_list, partition, stage):
        print(stage, dataset_type)
        path_stage_2_correct_db = self.path_db[stage]['correct'][dataset_type]
        path_stage_2_refuted_db = self.path_db[stage]['refuted'][dataset_type]
        path_stage_2_nei_db = self.path_db[stage]['nei'][dataset_type]
        
        mkdir_if_not_exist(os.path.dirname(path_stage_2_correct_db))
        
        if os.path.isfile(path_stage_2_correct_db) and os.path.isfile(path_stage_2_refuted_db) and os.path.isfile(path_stage_2_nei_db):
            print('sqlite database already created', dataset_type)
        else:
            if dataset_type not in self.settings:
                self.settings[dataset_type] = {}
            
            if stage not in self.settings:
                self.settings[stage] = {}
            if dataset_type not in self.settings[stage]:
                self.settings[stage][dataset_type] = {}
                
            self.settings[stage][dataset_type]['nr_supports'] = 0
            self.settings[stage][dataset_type]['nr_refuted'] = 0
            self.settings[stage][dataset_type]['nr_nei'] = 0

            with SqliteDict(path_stage_2_correct_db) as stage_2_correct_db:
                with SqliteDict(path_stage_2_refuted_db) as stage_2_refuted_db:
                    with SqliteDict(path_stage_2_nei_db) as stage_2_nei_db:
                        for claim_nr in tqdm(partition, total = len(partition), desc='create database '+dataset_type):
                                claim_dict = claim_dict_list[claim_nr]
                                if stage == 'stage_2':
                                    correct_evidence_list, incorrect_evidence_list = get_claim_dict_stage_2(
                                        claim_dict, wiki_database, claim_nr, self.nlp)
                                elif stage == 'stage_3':
                                    correct_evidence_list, incorrect_evidence_list = get_claim_dict_stage_3(
                                        claim_dict, wiki_database, claim_nr, self.nlp)
                                else:
                                    raise ValueError('correct evidence not in list')

                                label = claim_dict['label']

                                if label == 'SUPPORTS':
                                    if len(correct_evidence_list) > 0:
                                        stage_2_correct_db[claim_nr] = correct_evidence_list
                                        self.settings[stage][dataset_type]['nr_supports'] += 1
                                elif label  == 'REFUTES':
                                    if len(correct_evidence_list) > 0:
                                        stage_2_refuted_db[claim_nr] = correct_evidence_list
                                        self.settings[stage][dataset_type]['nr_refuted'] += 1
                                if len(incorrect_evidence_list) > 0:
                                    stage_2_nei_db[claim_nr] = incorrect_evidence_list
                                    self.settings[stage][dataset_type]['nr_nei'] += 1

                        stage_2_nei_db.commit()
                    stage_2_refuted_db.commit()
                stage_2_correct_db.commit()
            self.save_settings()

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
        self.settings['nr_claims_train'] = len(partition['train'])
        self.settings['nr_claims_validation'] = len(partition['validation'])
        self.settings['nr_claims_dev'] = len(partition['dev'])

        return partition

def get_claim_dict_stage_3(claim_dict, wiki_database, claim_nr, nlp):
    method_tokenization = 'tokenize_text_pos'
    
    tag_2_id_dict = get_tag_2_id_dict_unigrams()
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
                    if line_nr not in sentence_dict[normalised_title]:
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
    tag_list_claim, word_list_claim = get_word_tag_list_from_text(text_str = text_claim, nlp = nlp, method_tokenization_str = method_tokenization)
    tag_list_claim = [tag_2_id_dict[pos] for pos in tag_list_claim]

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
                tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis, nlp = nlp, method_tokenization_str = method_tokenization)
                tag_list_hypotheses = [tag_2_id_dict[pos] for pos in tag_list_hypotheses]
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
                tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis_incorrect, nlp = nlp, method_tokenization_str = method_tokenization)
                tag_list_hypotheses = [tag_2_id_dict[pos] for pos in tag_list_hypotheses]
                dict_tmp['hypotheses'] = word_list_hypotheses
                dict_tmp['hypotheses_tags'] = tag_list_hypotheses
                potential_dict_list.append(dict_tmp)
        
        nr_correct_sentences = len(correct_dict_list)
        nr_random_sentences = len(potential_dict_list)
        # --- only add correct claim if enough random sentences --- #
        if nr_correct_sentences + nr_random_sentences >= 5:
            correct_generated_observation_flag = True
            indices_selected_list = random.sample(range(nr_random_sentences), max(0, min(nr_random_sentences, 5-nr_correct_sentences)))
            
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
            dict_out['premises_tags'] = tag_list_claim
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
            dict_out['labels'] = 'NOT ENOUGH INFO'#claim_dict['label']
            dict_out['premises_tags'] = tag_list_claim
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

def get_claim_dict_stage_2(claim_dict, wiki_database, claim_nr, nlp):
    method_tokenization = 'tokenize_text_pos'
    
    tag_2_id_dict = get_tag_2_id_dict_unigrams()

    sentence_dict = {}
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
            tag_list_claim, word_list_claim = get_word_tag_list_from_text(text_str = text_claim, nlp = nlp, method_tokenization_str = method_tokenization)
            tag_list_claim = [tag_2_id_dict[pos] for pos in tag_list_claim]
            dict_out['premises'] = word_list_claim
            dict_out['ids'] = str(claim_dict['id']) + '_' + str(claim_nr) + '_' + str(interpreters_nr) + '_correct'
            dict_out['labels'] = claim_dict['label']
            dict_out['premises_tags'] = tag_list_claim


            text_hypothesis = wiki_database.get_line_from_title(title, line_nr)
            tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis, nlp = nlp, method_tokenization_str = method_tokenization)
            dict_out['hypotheses'] = word_list_hypotheses
            tag_list_hypotheses = [tag_2_id_dict[pos] for pos in tag_list_hypotheses]
            dict_out['hypotheses_tags'] = tag_list_hypotheses
            
            correct_evidence_list.append(dict_out)

            # incorrect
            dict_out = {}
            lines_file = wiki_database.get_lines_list_from_title(title)
            cosine_distance_list = []
            for i in range(len(lines_file)):
                text_line = lines_file[i]
                if len(text_line)>4 and (i not in sentences_correct_list):
                    cosine_distance = get_cosine(text_claim, text_line)
                else:
                    cosine_distance = 0 
                cosine_distance_list.append(cosine_distance)
            index_largest_cosine_distance = cosine_distance_list.index(max(cosine_distance_list))
            text_hypothesis_incorrect = lines_file[index_largest_cosine_distance]
            
            dict_out['premises'] = word_list_claim
            dict_out['ids'] = str(claim_dict['id']) + '_' + str(claim_nr) + '_' + str(interpreters_nr) + '_random'
            dict_out['labels'] = 'NOT ENOUGH INFO'
            dict_out['premises_tags'] = tag_list_claim
            tag_list_hypotheses, word_list_hypotheses = get_word_tag_list_from_text(text_str = text_hypothesis_incorrect, nlp = nlp, method_tokenization_str = method_tokenization)
            dict_out['hypotheses'] = word_list_hypotheses
            tag_list_hypotheses = [tag_2_id_dict[pos] for pos in tag_list_hypotheses]
            dict_out['hypotheses_tags'] = tag_list_hypotheses
            incorrect_evidence_list.append(dict_out)
        interpreters_nr += 1
    return correct_evidence_list, incorrect_evidence_list

import random
from random import shuffle
import spacy

from utils_db import mkdir_if_not_exist, dict_load_json, dict_save_json


        
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

from random import shuffle

def hypothesis_evidence_2_index(hypothesis, premise, max_length = 75, randomise_flag = False):
    # input
    # - hypothesis: list of words
    # - evidence : list of words
    # first two indices are not used (for beginning of sentence and end of sentence tags)
    # last element is always the same (shuffled or not), same for first two

    vocab = list(set(premise))
    nr_empty_start = 2
    loc_hypothesis = []
    loc_premise = []
    
    for word in hypothesis:
        if word in vocab:
            loc_hypothesis.append(vocab.index(word))
        else: 
            loc_hypothesis.append(max_length-1)
            
    loc_premise = [vocab.index(word) for word in premise]

    shuffle_list = list(range(2, max_length-1))
    
    if randomise_flag:
        shuffle(shuffle_list)

    # shuffle_list.append(max_length-1)
    
    # print(max(loc_hopothesis), len(shuffle_list))
    loc_hypothesis_new = []
    for loc in loc_hypothesis:
        if loc == max_length-1:
            loc_hypothesis_new.append(loc)
        else:
            loc_hypothesis_new.append(shuffle_list[loc])

    # loc_premise_new = []
    # for loc in loc_premise:
    #     if loc == max_length-1:
    #         loc_premise_new.append(loc)
    #     else:
    #         loc_premise_new.append(shuffle_list[loc])

    # print(loc_hypothesis, len(shuffle_list))
    # loc_hypothesis = [shuffle_list[loc] for loc in loc_hypothesis]
    loc_premise_new = [shuffle_list[loc] for loc in loc_premise]
    
    return loc_hypothesis_new, loc_premise_new

import os
from wiki_database import WikiDatabaseSqlite
from utils_db import mkdir_if_not_exist
import config
import os
from utils_db import load_jsonl
from claim_database_stage_2_3 import ClaimDatabaseStage_2_3

import config
if __name__ == '__main__':
    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
    # mkdir_if_not_exist(path_wiki_database_dir)
    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)


    path_dir_database = os.path.join(config.ROOT,'claim_db')
    path_raw_data = os.path.join(config.ROOT, config.DATA_DIR, config.RAW_DATA_DIR)
    fraction_validation = 0.1
    method_combination = 'max_correct' # 'equal', one_from_every_claim
    claim_database = ClaimDatabaseStage_2_3(path_database_dir = path_dir_database, 
                                         path_raw_data = path_raw_data, 
                                         fraction_validation = fraction_validation,
                                         method_combination = method_combination,
                                         wiki_database = wiki_database)
