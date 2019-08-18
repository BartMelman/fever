from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms

import os
from random import shuffle
from tqdm import tqdm

from doc_results_db import ClaimTensorDatabase
from utils_db import mkdir_if_not_exist, dict_save_json, dict_load_json, get_file_name_from_variable_list

import config

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class DataNeuralNetwork():
    def __init__(self, method_database, setup, claim_data_set, wiki_database, selection_experiment_dict):
        self.setup = setup
        self.method_database = method_database
        self.claim_data_set = claim_data_set
        self.selection_experiment_dict = selection_experiment_dict
        self.tensor_db = ClaimTensorDatabase(setup = self.setup, 
            wiki_database = wiki_database, 
            claim_data_set = self.claim_data_set, 
            selection_experiment_dict = self.selection_experiment_dict)

        self.path_combined_data_dir = os.path.join(self.tensor_db.path_results_dir, 'neural_network', 'data_setup_' + str(claim_data_set) + '_' + str(self.setup) + '_' + self.method_database)
        self.path_settings = os.path.join(self.path_combined_data_dir, 'settings.json')
        
        print('DataNeuralNetwork')
        if os.path.isfile(self.path_settings):
        	print('- data already created')
        	self.settings = dict_load_json(self.path_settings)
        else:
            self.settings = {}
            self.create_folder_combined_files()
            
    def create_folder_combined_files(self):
        mkdir_if_not_exist(self.path_combined_data_dir)

        dir_list = [self.tensor_db.path_label_correct_evidence_true_dir, 
                   self.tensor_db.path_label_correct_evidence_false_dir,
                   self.tensor_db.path_label_refuted_evidence_true_dir,
                   self.tensor_db.path_label_refuted_evidence_false_dir]
        
        nr_observations_list = [self.tensor_db.settings['nr_correct_true'], 
                                self.tensor_db.settings['nr_correct_false'], 
                                self.tensor_db.settings['nr_refuted_true'], 
                                self.tensor_db.settings['nr_refuted_false']]

        if self.method_database == 'include_all':
            total_nr_observations = sum(nr_observations_list)
            random_id_list = list(range(total_nr_observations))
            shuffle(random_id_list)

            for i in tqdm(range(len(dir_list))):
                dir = dir_list[i]
                nr_obervations = nr_observations_list[i]
                for idx in range(nr_obervations):
                    transformed_ids = random_id_list[idx + sum(nr_observations_list[0:i])]
                    
                    file_name_variables_load = os.path.join(dir, 'variable_' + str(idx) + '.pt')
                    file_name_label_load = os.path.join(dir, 'label_' + str(idx) + '.pt')
                    
                    file_name_variables_write = os.path.join(self.path_combined_data_dir, 'variable_' + str(transformed_ids) + '.pt')
                    file_name_label_write = os.path.join(self.path_combined_data_dir, 'label_' + str(transformed_ids) + '.pt')

                    X = torch.load(file_name_variables_load)
                    Y = torch.load(file_name_label_load)

                    torch.save(X, file_name_variables_write)
                    torch.save(Y, file_name_label_write)

            self.settings['nr_observations'] = len(random_id_list)
            self.settings['nr_correct_true'] = self.tensor_db.settings['nr_correct_true']
            self.settings['nr_correct_false'] = self.tensor_db.settings['nr_correct_false']
            self.settings['nr_refuted_true'] = self.tensor_db.settings['nr_refuted_true']
            self.settings['nr_refuted_false'] = self.tensor_db.settings['nr_refuted_false']

        elif self.method_database == 'equal_class':
            min_nr_observations = min(nr_observations_list)
            random_id_list = list(range(min_nr_observations*4))
            shuffle(random_id_list)

            for i in tqdm(range(len(dir_list))):
                dir = dir_list[i]
                nr_obervations = nr_observations_list[i]
                random_id_list_setting = list(range(nr_obervations))
                shuffle(random_id_list_setting)
                j=0
                for idx in random_id_list_setting[0:min_nr_observations]:
                    transformed_ids = random_id_list[j + min_nr_observations*i]
                    
                    file_name_variables_load = os.path.join(dir, 'variable_' + str(idx) + '.pt')
                    file_name_label_load = os.path.join(dir, 'label_' + str(idx) + '.pt')                    
                    file_name_variables_write = os.path.join(self.path_combined_data_dir, 'variable_' + str(transformed_ids) + '.pt')
                    file_name_label_write = os.path.join(self.path_combined_data_dir, 'label_' + str(transformed_ids) + '.pt')

                    X = torch.load(file_name_variables_load)
                    Y = torch.load(file_name_label_load)

                    torch.save(X, file_name_variables_write)
                    torch.save(Y, file_name_label_write)
                    j += 1

            self.settings['nr_observations'] = len(random_id_list)
            self.settings['nr_correct_true'] = min_nr_observations
            self.settings['nr_correct_false'] = min_nr_observations
            self.settings['nr_refuted_true'] = min_nr_observations
            self.settings['nr_refuted_false'] = min_nr_observations

        else:
            raise ValueError('method not in method options', method_database)

        dict_save_json(self.settings, self.path_settings)

# https://github.com/pytorch/examples/blob/master/mnist/main.py
class NetHighWayConnections(nn.Module):
    def __init__(self, nr_input_variables, width, depth, output_dim):
        super(NetHighWayConnections, self).__init__()
        # input layer
        print(nr_input_variables, width, depth, output_dim)
        proposal_layers = [nn.Linear(nr_input_variables, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)]
        # body
        for i in range(depth-1):
            proposal_layers.append(
                SkipConnection( nn.Linear(width, width), nn.ReLU(), nn.BatchNorm1d(num_features=width), ))

        # output layer
        proposal_layers.append(
            nn.Linear(width, output_dim)
        )
        proposal_layers.append(
            nn.Sigmoid()
        )

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)

class SkipConnection(nn.Module):
    """
    Skip-connection over the sequence of layers in the constructor.
    The module passes input data sequentially through these layers
    and then adds original data to the result.
    """
    def __init__(self, *args):
        super(SkipConnection, self).__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)

class NetFullyConnectedAutomated(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(NetFullyConnectedAutomated, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)]
        # body
        for i in range(depth-1):
            # proposal_layers.append(
            #         nn.Linear(width, width),
            #         nn.ReLU(),
            #         nn.BatchNorm1d(num_features=width)
            #         )
            proposal_layers.append(nn.Linear(width, width))
            proposal_layers.append(nn.ReLU())
            proposal_layers.append(nn.BatchNorm1d(num_features=width))

        # output layer
        # proposal_layers.append(
        #     nn.Linear(width, output_dim),
        #     nn.Sigmoid()
        # )
        proposal_layers.append(
            nn.Linear(width, output_dim)
        )
        proposal_layers.append(
            nn.Sigmoid()
        )


        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)

class NetFullyConnected(nn.Module):
    def __init__(self, nr_input_variables, nr_hidden_neurons, nr_output_variables):
        super(NetFullyConnected, self).x
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc_input = nn.Linear(nr_input_variables, nr_hidden_neurons)
        self.fc_hidden = nn.Linear(nr_hidden_neurons, nr_hidden_neurons)
        self.fc_output = nn.Linear(nr_hidden_neurons, nr_output_variables)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(nr_hidden_neurons)
        self.bn2 = nn.BatchNorm1d(nr_hidden_neurons)
        
    def forward(self, x):
        x = F.relu(self.fc_input(x))
        x = self.bn1(x)
        x = F.relu(self.fc_hidden(x))
        x = self.bn2(x)
        x = F.relu(self.fc_hidden(x))
        # x = self.bn2(x)

        return self.sigmoid(self.fc_output(x))

class LogisticRegression(nn.Module):
    def __init__(self, nr_input_variables, nr_output_variables):
        super(LogisticRegression, self).__init__()
        self.fc_input = nn.Linear(nr_input_variables, nr_output_variables)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc_input(x))

def train(log_interval, model, device, train_loader, optimizer, epoch, criterion, flag_weighted_criterion = False, criterion_dict = None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.view(-1, 1)
        optimizer.zero_grad()
        output = model(data)
                                
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_performance(model, device, test_loader, criterion, threshold, flag_weighted_criterion = False, criterion_dict = None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1, 1)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss

            threshold = 0.5
            target_unit8 = target>0.5
            targ = torch.FloatTensor([1, 2]).view(-1,1)>0.5
            pred = output>threshold
            correct += target_unit8.eq(pred).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTraining set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)
    
def test_performance(model, device, test_loader, criterion, threshold, flag_weighted_criterion = False, criterion_dict = None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1, 1)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            threshold = 0.5
            target_unit8 = target>0.5
            targ = torch.FloatTensor([1, 2]).view(-1,1)>0.5
            pred = output>threshold
            correct += target_unit8.eq(pred).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)



class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path_data_set, list_ids):
        'Initialization'
        self.path_data_set = path_data_set
        self.list_ids = list_ids
        self.nr_observations = len(list_ids)
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.nr_observations

    def __getitem__(self, index):
        'Generates one sample of data'
        
        file_name_variables = os.path.join(self.path_data_set, 'variable_' + str(self.list_ids[index])  + '.pt')
        file_name_label = os.path.join(self.path_data_set, 'label_' + str(self.list_ids[index])  + '.pt')
        
        X = torch.load(file_name_variables).float()
        y = torch.load(file_name_label).float().view(-1, 1)
        
        return X, y

class NeuralNetwork():
    def __init__(self, claim_data_set, method_database, setup, settings_model, wiki_database, nn_model_name, selection_experiment_dict):
        self.claim_data_set = claim_data_set
        self.selection_experiment_dict = selection_experiment_dict
        self.method_database = method_database
        self.setup = setup
        self.nn_model_name = nn_model_name
        self.flag_weighted_criterion = settings_model['flag_weighted_criterion']
        self.width = settings_model['width']
        self.depth = settings_model['depth']
        self.fraction_training = settings_model['fraction_training']
        self.use_cuda = settings_model['use_cuda']
        self.seed = settings_model['seed']
        self.lr = settings_model['lr']
        self.momentum = settings_model['momentum']
        self.params = settings_model['params']
        self.nr_epochs = settings_model['nr_epochs']
        self.log_interval = settings_model['log_interval']
        self.batch_size = settings_model['batch_size']
        self.optimizer = settings_model['optimizer']
        self.data_nn = self.get_data(wiki_database)
        self.settings_data = self.data_nn.settings
        self.nr_observations = self.data_nn.settings['nr_observations']
        print('nr observations', self.nr_observations)
        self.nr_variables = self.data_nn.tensor_db.settings['nr_variables']
        self.file_name = 'model_' + self.method_database + '_' + get_file_name_from_variable_list([self.setup,
                self.claim_data_set, self.nn_model_name, self.width, self.depth, self.nr_epochs,
                self.fraction_training, self.batch_size, self.optimizer, self.flag_weighted_criterion, self.lr])

        self.path_model_dir = os.path.join(self.data_nn.tensor_db.path_results_dir, 
        	'neural_network', self.file_name)
        self.path_settings = os.path.join(self.path_model_dir, 'settings.json')
        self.path_plots_dir = os.path.join(self.path_model_dir, 'plots')
        mkdir_if_not_exist(self.path_plots_dir)
        self.criterion = nn.BCEWithLogitsLoss() # F.nll_loss(
        self.threshold = 0.5

        mkdir_if_not_exist(self.path_model_dir)

        print('NeuralNetwork')
        if os.path.isfile(self.path_settings):
            print('- load model')
            self.settings = dict_load_json(self.path_settings)
            self.model = torch.load(self.settings['path_final_model'])
            self.model.eval()
            
        else:
            print('- train model')
            self.settings = {}
            self.partition = self.get_partition()
            self.training_data_loader, self.validation_data_loader = self.get_data_generators()
            self.initialise_model()
            self.train_model()
            self.plot_and_save_results()
            self.model = torch.load(self.settings['path_final_model'])
            self.model.eval()
            dict_save_json(self.settings, self.path_settings)
    
    def plot_and_save_results(self):
    	# description: Create two plots with the loss and accuracy for all epochs and save them
        idx = 1
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        epoch_list = []

        for key in self.settings['training']:
            epoch_list.append(idx)
            
            train_loss_list.append(self.settings['training'][key]['loss'])
            train_acc_list.append(self.settings['training'][key]['acc'])
            test_loss_list.append(self.settings['test'][key]['loss'])
            test_acc_list.append(self.settings['test'][key]['acc'])
            
            idx+=1

		# --- figure with training and test loss --- #
        x = [epoch_list, epoch_list]
        y = [train_loss_list, test_loss_list]
        label_list = ['training loss', 'test loss']
        x_min = 0
        x_max = max(max(experiment) for experiment in x)+1
        y_min = min(min(experiment) for experiment in y)*0.9
        y_max = max(max(experiment) for experiment in y)*1.1
        fontsize = 15

        path_save = os.path.join(self.path_plots_dir, 'loss.png')

        save_plot(x_list = x, y_list = y, x_label_str = 'epoch', y_label_str ='loss', label_list = label_list,
		          x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, fontsize = fontsize, path_save = path_save)

		# --- figure with training and test accuracy --- #
        x = [epoch_list, epoch_list]
        y = [train_acc_list, test_acc_list]
        label_list = ['training acc', 'test acc']
        x_min = 0
        x_max = max(max(experiment) for experiment in x)+1
        y_min = min(min(experiment) for experiment in y)*0.9
        y_max = max(max(experiment) for experiment in y)*1.1

        fontsize = 15

        path_save = os.path.join(self.path_plots_dir, 'acc.png')

        save_plot(x_list = x, y_list = y, x_label_str = 'epoch', y_label_str ='acc', label_list = label_list,
		          x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, fontsize = fontsize, path_save = path_save)

    def get_data(self, wiki_database):
        return DataNeuralNetwork(self.method_database, 
            self.setup, 
            self.claim_data_set, 
            wiki_database, 
            self.selection_experiment_dict)
    
    def get_partition(self):
        partition = {}
        partition['train'] = list(range(0, int(self.nr_observations*self.fraction_training)))
        partition['validation'] = list(range(int(self.nr_observations*self.fraction_training), self.nr_observations))
        return partition
    
    def get_data_generators(self):
        training_set = Dataset(self.data_nn.path_combined_data_dir, self.partition['train'])
        training_data_loader = data.DataLoader(training_set, **self.params)

        validation_set = Dataset(self.data_nn.path_combined_data_dir, self.partition['validation'])
        validation_data_loader = data.DataLoader(validation_set, **self.params)
        return training_data_loader, validation_data_loader
    
    def initialise_model(self):
        torch.manual_seed(self.seed)

        kwargs = {'num_workers': 6, 'pin_memory': True} if self.use_cuda else {}

        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.nn_model_name == 'NetFullyConnected':
            self.model = NetFullyConnected(nr_input_variables = self.nr_variables, 
            	nr_hidden_neurons = self.width, nr_output_variables = 1).to(self.device)
        elif self.nn_model_name == 'NetHighWayConnections': 
        	self.model = NetHighWayConnections(nr_input_variables = self.nr_variables, 
        		width = self.width, depth = self.depth, output_dim = 1).to(self.device)
       	elif self.nn_model_name == 'LogisticRegression':
       		self.model = LogisticRegression(nr_input_variables = self.nr_variables, nr_output_variables = 1).to(self.device)
       	elif self.nn_model_name == 'NetFullyConnectedAutomated':
       		self.model = NetFullyConnectedAutomated(input_dim = self.nr_variables, 
       			width = self.width, depth = self.depth, output_dim = 1).to(self.device)
        else:
        	raise ValueError('model_nr not in selection', self.model_nr)

    def train_model(self):
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
        	raise ValueError('optimizer not in list', self.optimizer)
        self.settings['training'] = {}
        self.settings['test'] = {}
        epoch_lowest_loss = 0	
        lowest_test_loss = None
        for epoch in range(1, self.nr_epochs + 1):
            train(self.log_interval, self.model, self.device, self.training_data_loader, 
                optimizer, epoch, self.criterion,
                flag_weighted_criterion = self.flag_weighted_criterion, 
                criterion_dict = self.settings_data)
            loss_train, acc_train = train_performance(self.model, self.device, 
                self.training_data_loader, self.criterion, self.threshold,
                flag_weighted_criterion = self.flag_weighted_criterion, 
                criterion_dict = self.settings_data)
            loss_test, acc_test = test_performance(self.model, self.device, 
                self.validation_data_loader, self.criterion, self.threshold,
                flag_weighted_criterion = self.flag_weighted_criterion, 
                criterion_dict = self.settings_data)
            path_model_epoch = os.path.join(self.path_model_dir, 'model_epoch_' + str(epoch) + '.pt')
            torch.save(self.model, path_model_epoch)
            self.settings['training']['epoch_' + str(epoch)] = {}
            self.settings['training']['epoch_' + str(epoch)]['loss'] = loss_train
            self.settings['training']['epoch_' + str(epoch)]['acc'] = acc_train
            self.settings['test']['epoch_' + str(epoch)] = {}
            self.settings['test']['epoch_' + str(epoch)]['loss'] = loss_test
            self.settings['test']['epoch_' + str(epoch)]['acc'] = acc_test
            if lowest_test_loss == None or loss_test < lowest_test_loss:
            	lowest_test_loss = loss_test
            	epoch_lowest_loss = epoch

        self.settings['path_final_model'] = os.path.join(self.path_model_dir, 'model_epoch_' + str(epoch_lowest_loss) + '.pt')
        self.settings['epoch_final_model'] = epoch_lowest_loss

def save_plot(x_list, y_list, x_label_str, y_label_str, label_list, x_min, x_max, y_min, y_max, fontsize, path_save):
    plt.figure()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        label = label_list[i]
        plt.plot(x, y, '-*', linewidth=2, markersize=10, label=label)
        
    plt.grid(color='k', linestyle='--', linewidth=0.5)

    plt.xlabel(x_label_str, fontsize=fontsize)
    plt.ylabel(y_label_str, fontsize=fontsize)
    
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    
    plt.legend(loc=(1.02, 0.33), fontsize=fontsize-3)
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(path_save)

    
