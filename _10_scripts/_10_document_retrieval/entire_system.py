from utils_db import dict_load_json, dict_save_json
from utils_db import get_file_name_from_variable_list
from tqdm import tqdm

class EntireSystem():
    
    def __init__(self, wiki_database, nlp, path_stage_2_model, path_stage_3_model,
                 path_dir_doc_selected, method_tokenization, path_base_dir,
                 path_word_dict_stage_2, path_word_dict_stage_3,
                 embeddings_settings_sentence_retrieval_list = [],
                 embeddings_settings_label_prediction_list = []):
        # === process inputs === #
        self.path_stage_2_model = path_stage_2_model
        self.path_stage_3_model = path_stage_3_model
        self.nlp = nlp
        self.path_dir_doc_selected = path_dir_doc_selected
        self.method_tokenization = method_tokenization
        self.path_base_dir = path_base_dir
        self.path_word_dict_stage_2 = path_word_dict_stage_2
        self.path_word_dict_stage_3 = path_word_dict_stage_3
        self.embeddings_settings_sentence_retrieval_list = embeddings_settings_sentence_retrieval_list
        self.embeddings_settings_label_prediction_list = embeddings_settings_label_prediction_list
        
        # === paths === #
        self.path_document_retrieval_dir = os.path.join(path_base_dir, get_file_name_from_variable_list(['document_retrieval']))
        self.path_sentence_retrieval_dir = os.path.join(path_base_dir, 'sentence_retrieval')
        self.path_label_prediction_dir = os.path.join(path_base_dir, 'label_prediction')
        
        for embeddings_setting in embeddings_settings_sentence_retrieval_list:
            self.path_sentence_retrieval_dir = get_file_name_from_variable_list([self.path_sentence_retrieval_dir, embeddings_setting])

        for embeddings_setting in embeddings_settings_label_prediction_list:
            self.path_label_prediction_dir = get_file_name_from_variable_list([self.path_label_prediction_dir, embeddings_setting])
        
        if not os.path.isdir(self.path_base_dir):
            os.makedirs(self.path_base_dir)
                       
        self.path_settings = os.path.join(self.path_base_dir, 'settings.json')
        
        if os.path.isfile(self.path_settings):
            self.settings = dict_load_json(self.path_settings)
        else:
            self.settings = {}
        
        if 'nr_claims' not in self.settings:
            self.settings['nr_claims'] = self.nr_files_in_dir(self.path_dir_doc_selected)
            self.save_settings()
            
        self.nr_claims = self.settings['nr_claims']
        self.nr_claims = 100
        print('nr claims:', self.nr_claims)
        
        # === process === #
        self.tag_2_id_dict = get_tag_2_id_dict_unigrams()
    
        if not os.path.isdir(self.path_document_retrieval_dir):
            os.makedirs(self.path_document_retrieval_dir)
            self.document_retrieval()
        
        if not os.path.isdir(self.path_sentence_retrieval_dir):
            os.makedirs(self.path_sentence_retrieval_dir)
            self.sentence_retrieval()
            
        if not os.path.isdir(self.path_label_prediction_dir):
            os.makedirs(self.path_label_prediction_dir)
            self.label_prediction()
            
        self.compute_score()
        
    def compute_score(self):
        # STAGE 2
        # F1
        # PRECISION
        # RECALL
        
        # STAGE 3
        # FEVER
        
        list_claims = []
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_sentence_retrieval_dir, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            list_claims.append(claim_dict)
        
        strict_score, acc_score, pr, rec, f1 = fever_score(predictions=list_claims, actual=None, max_evidence=5)
        
        print(strict_score, acc_score, pr, rec, f1)
        
    
    def label_prediction(self):
        print('- label prediction: initialise')
        word_dict = pickle.load( open( self.path_word_dict_stage_3, "rb" ) )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.path_stage_3_model)

        vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
        embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
        hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
        num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
        
        use_oov_flag=0
        if 'oov' in self.embeddings_settings_label_prediction_list:
            use_oov_flag=1
            
        use_pos_tag_flag=0
        if 'pos' in self.embeddings_settings_label_prediction_list:
            use_pos_tag_flag=1
            
        model = ESIM(vocab_size,
                     embedding_dim,
                     hidden_size,
                     num_classes=num_classes,
                     use_pos_tag_flag=use_pos_tag_flag,
                     use_oov_flag=use_oov_flag,
                     device=device).to(device)

        model.load_state_dict(checkpoint["model"])

        model.eval()
        
        print('- label prediction: iterate through claims')
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_sentence_retrieval_dir, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            
            prob_list  = []
            prob_list_supported = []
            prob_list_refuted = []
            for i in range(len(claim_dict['sentence_retrieval']['doc_nr_list'])):
                doc_nr = claim_dict['sentence_retrieval']['doc_nr_list'][i]
                line_nr = claim_dict['sentence_retrieval']['line_nr_list'][i]
                prob = compute_prob_stage_3(model, claim_dict, doc_nr, line_nr)
                prob_list.append(prob)
                prob_list_supported.append(prob[2])
                prob_list_refuted.append(prob[1])
                
            if max(prob_list_supported) > 0.5:
                claim_dict['predicted_label'] = 'SUPPORTS'
            elif max(prob_list_refuted) > 0.5:
                claim_dict['predicted_label'] = 'REFUTES'
            else:
                claim_dict['predicted_label'] = 'NOT ENOUGH INFO'
            
            path_save = os.path.join(self.path_label_prediction_dir, str(claim_nr) + '.json')
            self.save_dict(claim_dict, path_save)
                
    def sentence_retrieval(self):
        print('- sentence retrieval: initialise')
        word_dict = pickle.load( open( self.path_word_dict_stage_2, "rb" ) )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.path_stage_2_model)

        vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
        embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
        hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
        num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
        
        use_oov_flag=0
        if 'oov' in self.embeddings_settings_sentence_retrieval_list:
            use_oov_flag=1
            
        use_pos_tag_flag=0
        if 'pos' in self.embeddings_settings_sentence_retrieval_list:
            use_pos_tag_flag=1
            
        model = ESIM(vocab_size,
                     embedding_dim,
                     hidden_size,
                     num_classes=num_classes,
                     use_pos_tag_flag=use_pos_tag_flag,
                     use_oov_flag=use_oov_flag,
                     device=device).to(device)

        model.load_state_dict(checkpoint["model"])

        model.eval()
        
        print('- sentence retrieval: iterate through claims')
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_document_retrieval_dir, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            
            list_prob = []
            list_doc_nr = []
            list_line_nr = []
        
            for doc_nr in claim_dict['document_retrieval']:
                for line_nr in claim_dict['document_retrieval'][doc_nr]:
                    if 'sentence_retrieval' not in claim_dict:
                        claim_dict['sentence_retrieval'] = {}
                    if doc_nr not in claim_dict['sentence_retrieval']:
                        claim_dict['sentence_retrieval'][doc_nr] = {}
                    if line_nr not in claim_dict['sentence_retrieval'][doc_nr]:
                        claim_dict['sentence_retrieval'][doc_nr][line_nr] = {}
                    
                    prob = compute_prob_stage_2(model, claim_dict, doc_nr, line_nr)
                    claim_dict['sentence_retrieval'][doc_nr][line_nr]['prob'] = prob
                    
                    list_doc_nr.append(doc_nr)
                    list_line_nr.append(line_nr)
                    list_prob.append(prob)
                    
            sorted_list_doc_nr = sort_list(list_doc_nr, list_prob)[-5:]
            sorted_list_line_nr = sort_list(list_line_nr, list_prob)[-5:]
            sorted_list_prob = sort_list(list_prob, list_prob)[-5:]
            claim_dict['sentence_retrieval']['doc_nr_list'] = sorted_list_doc_nr  
            claim_dict['sentence_retrieval']['line_nr_list'] = sorted_list_line_nr  
            claim_dict['sentence_retrieval']['prob_list'] = sorted_list_prob 
            
            claim_dict['predicted_evidence'] = []
            for i in range(len(sorted_list_doc_nr)):
                doc_nr = sorted_list_doc_nr[i]
                title = wiki_database.get_title_from_id(int(doc_nr)
                line_nr = int(sorted_list_line_nr[i])
                claim_dict['predicted_evidence'].append([title, line_nr])                                   
            
            path_save = os.path.join(self.path_sentence_retrieval_dir, str(claim_nr) + '.json')
            self.save_dict(claim_dict, path_save)
                    
        
    def document_retrieval(self):
#         claim_nr = 12
#         line_nr = 0
#         nr_in_doc_selected_list = 0
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_dir_doc_selected, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            claim = Claim(claim_dict)
            claim_text = claim.claim
            # === process word tags and word list === #
            tag_list_claim, word_list_claim = get_word_tag_list_from_text(text_str = claim_text, 
                                                                          nlp = nlp, 
                                                                          method_tokenization_str = method_tokenization)

            for doc_nr in claim_dict['docs_selected']:
                line_list = wiki_database.get_lines_from_id(doc_nr)
                nr_lines = len(line_list)
                for line_nr in range(nr_lines):
                    line_text = line_list[line_nr]
                    
                    # === process word tags and word list === #
                    tag_list_line, word_list_line = get_word_tag_list_from_text(text_str = line_text, 
                                                                                nlp = nlp, 
                                                                                method_tokenization_str = method_tokenization)

                    claim_dict['document_retrieval'] = {}
                    if str(doc_nr) not in claim_dict['document_retrieval']:
                        claim_dict['document_retrieval'][str(doc_nr)] = {}

                    if str(line_nr) not in claim_dict['document_retrieval'][str(doc_nr)]:
                        claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)] = {}

                    if 'claim' not in claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]:
                        claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim'] = {}

                    if 'document' not in claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]:
                        claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document'] = {}

                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['tag_list'] = [17] + tag_str_2_id_list(
                        tag_list_claim, tag_2_id_dict) + [17]
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['word_list'] = word_list_2_id_list(
                        ["_BOS_"] + word_list_claim + ["_EOS_"], word_dict)
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['tag_list'] = [17] + tag_str_2_id_list(
                        tag_list_line, tag_2_id_dict) + [17]
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['word_list'] = word_list_2_id_list(
                        ["_BOS_"] + word_list_line + ["_EOS_"], word_dict)

                    ids_document, ids_claim = hypothesis_evidence_2_index(hypothesis = word_list_line,
                                               premise = word_list_claim,
                                               randomise_flag = False)

                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['exact_match_list'] = [0] + ids_claim + [1]
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['exact_match_list'] = [0] + ids_document + [1]
            
            path_save = os.path.join(self.path_document_retrieval_dir, str(claim_nr) + '.json')
            self.save_dict(claim_dict, path_save)
            
    def save_settings(self):
        dict_save_json(self.settings, self.path_settings)
    
    def save_dict(self, input_dict, path):
        dict_save_json(input_dict, path)
        
    def load_dict(self, path):
        return dict_load_json(path)
    
    def nr_files_in_dir(self, path_dir):
        list_files = os.listdir(path_dir) # dir is your directory path
        number_files = len(list_files)
        return number_files
