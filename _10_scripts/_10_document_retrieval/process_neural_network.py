import os 
from wiki_database import WikiDatabaseSqlite
import torch

from utils_db import dict_load_json, dict_save_json, mkdir_if_not_exist
from utils_doc_results import Claim, ClaimDatabase
from tqdm import tqdm 
from doc_results_db import ClaimFile, ClaimTensorDatabase
from neural_network import NeuralNetwork

import config

class PredictLabels():
    def __init__(self, K, threshold, method, claim_tensor_db, wiki_database, neural_network, claim_database):
        # --- process input --- #
        self.K = K
        self.threshold = threshold
        self.method = method
        self.claim_tensor_db = claim_tensor_db
        self.model_nn = neural_network.model
        # --- variables --- #
        self.nr_claims = self.claim_tensor_db.settings['nr_total']
        self.path_predict_label_dir = os.path.join(claim_tensor_db.path_setup_dir, 'Predictions_' + str(self.K) + '_' + str(self.threshold) + '_' + method + '_' + neural_network.file_name)
        mkdir_if_not_exist(self.path_predict_label_dir)
        self.path_settings = os.path.join(self.path_predict_label_dir, 'settings.json')
        
        print('PredictLabels')
        if os.path.isfile(self.path_settings):
            print('- experiment alread performed')
            self.settings = dict_load_json(self.path_settings)
        else:
            self.settings = {}
            self.get_accuracy_save_results(wiki_database, claim_database)
            dict_save_json(self.settings, self.path_settings)
        
    def get_accuracy_save_results(self, wiki_database, claim_database):
        nr_correct = 0
        nr_documents_selected = 0
        
        for id_nr in tqdm(range(self.nr_claims)):
            path_file = os.path.join(self.claim_tensor_db.path_dict_variable_list_dir, str(id_nr) + '.json')
            dict_variables = dict_load_json(path_file)
            id = dict_variables['id']
            selected_documents_list = list(dict_variables['selected_documents'].keys())
            pred_value_list = []
            for selected_document_str in selected_documents_list:
                flag_process = 0
                if method == 'generated':
                    if int(selected_document_str) in dict_variables['ids_generated']:
                        flag_process = 1
                elif method == 'correct':
                    if int(selected_document_str) in dict_variables['ids_correct_docs']:
                        flag_process = 1
                elif method == 'selected':
                    flag_process = 1
                else:
                    raise ValueError('method not in method_list', method)
                    
                if flag_process == 1:
                    variable_list = dict_variables['selected_documents'][selected_document_str]['list_variables']
                    variable_tensor = torch.FloatTensor(variable_list)
                    pred_value_list += [self.model_nn(variable_tensor.unsqueeze(0)).item()]
                    if 'predicted_true' in dict_variables:
                        dict_variables['predicted_true'].append(selected_document_str) 
                    else:
                        dict_variables['predicted_true'] = [selected_document_str]
                        
            pred_value_list_sorted = [x for x,_ in sorted(zip(pred_value_list, dict_variables['predicted_true']))]
            pred_id_correct_list = [x for _,x in sorted(zip(pred_value_list, dict_variables['predicted_true']))]
            id_correct_list = []
            length_list = len(pred_id_correct_list)
            for i in range(length_list):
                if i<self.K:
                    id_correct_list.append(pred_id_correct_list[length_list-1-i])
                elif pred_value_list_sorted[length_list-1-i] > self.threshold:
                    id_correct_list.append(pred_id_correct_list[length_list-1-i])
                else:
                    break
                nr_documents_selected += 1

            file = ClaimFile(id = id, path_dir_files = self.claim_tensor_db.path_claims_dir)
            claim_dict = claim_database.get_claim_from_id(id)
            claim = Claim(claim_dict)
            for interpreter in claim.evidence:
                flag_correctly_predicted = True
                for proof in interpreter:
                    title_proof = proof[2]
                    if title_proof == None:
                        raise ValueError('should contain proof')
                    else:
                        id_proof = wiki_database.get_id_from_title(title_proof)
                        if str(id_proof) not in id_correct_list:
                            flag_correctly_predicted = False
                            break
                if flag_correctly_predicted == True:
                    nr_correct += 1
                    break
        
        print('accuracy', nr_correct / float(self.nr_claims), 'documents per claim', nr_documents_selected / float(self.nr_claims))
        self.settings['nr_correct'] = nr_correct
        self.settings['accuracy'] = nr_correct / float(self.nr_claims)
        self.settings['nr_documents_selected'] = nr_documents_selected
        self.settings['nr_claims'] = self.nr_claims
        self.settings['documents_per_claim'] = nr_documents_selected / float(self.nr_claims)


if __name__ == '__main__':
    # === variables === #
    claim_data_set_nn = 'train_adj'
    claim_data_set_results = 'dev'
    method_database = 'equal_class' # include_all, equal_class
    setup = 3
    nn_model_name = 'NetHighWayConnections' # 'NetHighWayConnections', 'LogisticRegression', 'NetFullyConnectedAutomated'   
    settings_model = {}
    settings_model['fraction_training'] = 0.9
    settings_model['use_cuda'] = False
    settings_model['seed'] = 1
    settings_model['lr'] = 0.001
    settings_model['momentum'] = 0.9 # 0.5
    settings_model['optimizer'] = 'ADAM'

    if method_database == 'include_all':
        settings_model['batch_size'] = 256
        settings_model['params'] = {'batch_size': 256, 'shuffle': True}
    elif method_database == 'equal_class':
        settings_model['batch_size'] = 128
        settings_model['params'] = {'batch_size': 128, 'shuffle': True}
    else:
        raise ValueError('method_database not in list', method_database)

    settings_model['nr_epochs'] = 5
    settings_model['log_interval'] = 10
    settings_model['width'] = 500
    settings_model['depth'] = 2
    settings_model['flag_weighted_criterion'] = False

    selection_experiment_dict = {}
    selection_experiment_dict['experiment_nr'] = 31
    selection_experiment_dict['K'] = 100
    selection_experiment_dict['title_tf_idf_normalise_flag'] = True
    selection_experiment_dict['selection_generation_or_selected'] = 'ids_generated' # 

    method_list = ['generated', 'correct', 'selected']
    method = 'generated'
    # selection_generation_or_selected = 'ids_generated' # 'ids_generated'
    K = 5
    # threshold = 0.1
    threshold_list = [0.99] # [0.01, 0.1, 0.25, 0.5, 0.75] # [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
    path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
    wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)
    print('step 1')
    claim_tensor_db = ClaimTensorDatabase(setup = setup, 
        wiki_database = wiki_database, 
        claim_data_set = claim_data_set_results, 
        selection_experiment_dict = selection_experiment_dict)
    print('step 2')
    claim_database = ClaimDatabase(path_dir_database = claim_tensor_db.path_dir_claim_database, 
                                   path_raw_data = claim_tensor_db.path_raw_claim_data, 
                                   claim_data_set = claim_tensor_db.claim_data_set)
    print('step 3')
    neural_network = NeuralNetwork(claim_data_set = claim_data_set_nn,
        method_database = method_database, 
        setup = setup, 
        settings_model = settings_model, 
        nn_model_name = nn_model_name,
        wiki_database = wiki_database, 
        selection_experiment_dict  = selection_experiment_dict)
        
    model_nn = neural_network.model

    for threshold in threshold_list:
        predict_labels_db = PredictLabels(K, threshold, method, claim_tensor_db, 
            wiki_database, neural_network, claim_database)
        # predict_labels_db = PredictLabels(K, threshold, method, claim_tensor_db, 
        #     wiki_database, neural_network, claim_database)


    

