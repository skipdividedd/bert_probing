import json
import re
import torch
import numpy as np
from copy import deepcopy
from val import evaluate_probe
from NeuroX.neurox.data import loader as data_loader
from NeuroX.neurox.interpretation import utils
from NeuroX.neurox.interpretation import ablation
from NeuroX.neurox.interpretation import linear_probe

def tok2idx(tokens): # from NeuroX https://github.com/fdalvi/NeuroX/blob/master/neurox/interpretation/utils.py
    uniq_tokens = set().union(*tokens)
    return {p: idx for idx, p in enumerate(uniq_tokens)}

def idx2tok(srcidx):  # from NeuroX https://github.com/fdalvi/NeuroX/blob/master/neurox/interpretation/utils.py
    return {v: k for k, v in srcidx.items()}

class Experiment:
    
    def __init__(self, path_trdata, path_trlabel, path_tedata, path_telabel, path_work, probing_type='cls'):

        
        self.path_trdata = path_trdata
        self.path_trlabel = path_trlabel
        self.path_tedata = path_tedata
        self.path_telabel = path_telabel
        self.path_work = path_work
        self.category = re.search(r'[a-zA-Z]+_[a-zA-Z]+(?=.txt)', path_trdata)[0]
        self.dataset = re.search(r'(?<=_)[a-zA-Z]+_[a-zA-Z]+(?=\/)', path_trdata)[0]
        self.path = self.path_work+f'large_data_{self.dataset}/data_{self.category}'

        if probing_type == 'token':
            self.activations_tr, self.num_layers = data_loader.load_activations(self.path+'/activations_train.json', 768)
            self.activations_te, self.num_layers = data_loader.load_activations(self.path+'/activations_te.json', 768)
            self.tokens_tr = data_loader.load_data(self.path_trdata, self.path_trlabel, self.activations_tr, max_sent_l=500, sentence_classification=False)
            self.tokens_te = data_loader.load_data(self.path_tedata, self.path_telabel, self.activations_te, max_sent_l=500, sentence_classification=False)
            self.X_tr_, self.y_tr_, self.mapping = utils.create_tensors(self.tokens_tr, self.activations_tr, 'None')
            self.label2idx_, self.idx2label_, self.src2idx, self.idx2src = self.mapping
            self.X_te_, self.y_te_, mapping = utils.create_tensors(self.tokens_te, self.activations_te, 'None', mappings = self.mapping)
            mask = self.y_tr_ != self.label2idx_['None']
            self.X_tr = self.X_tr_[mask]
            self.y_tr_2 = self.y_tr_[mask]
            mask = self.y_te_ != self.label2idx_['None']
            self.X_te = self.X_te_[mask]
            self.y_te_2 = self.y_te_[mask]
            self.label2idx = {}
            i = 0
            for x, y in self.label2idx_.items():
                if x != 'None':
                    self.label2idx[x] = i
                    i += 1
            self.y_tr = [self.label2idx[self.idx2label_[y]] for y in self.y_tr_2]
            self.y_tr = np.array(self.y_tr, dtype=np.int)
            self.y_te = [self.label2idx[self.idx2label_[y]] for y in self.y_te_2]
            self.y_te = np.array(self.y_te, dtype=np.int)
            self.idx2label = {v:k for k, v in self.label2idx.items()}


        elif probing_type == 'cls':
            with open(f"{self.path}/activations_train.json", "r") as content:
                  self.X_train = json.load(content)
            self.X_tr = []
            for idx, act in self.X_train.items():
                self.X_tr.append(np.asarray(act))
            self.X_tr = np.asarray(self.X_tr)    
            
            with open(f"{self.path}/activations_te.json", "r") as content:
                  self.X_test = json.load(content)
            self.X_te = []
            for idx, act in self.X_test.items():
                self.X_te.append(np.asarray(act))
            self.X_te = np.asarray(self.X_te)

            with open(path_trlabel, encoding="utf-8") as f:
                target_tokens = [[line.split('\t')[0].replace('\n', '')] for line in f]
    
            label2idx = tok2idx(target_tokens)
            idx2label = idx2tok(label2idx)
            
            labels = sorted(label2idx.keys())
            indeces = [int(i) for i in range(len(labels))]
            
            self.label2idx = dict(zip(labels, indeces))
            self.idx2label = {v: k for k, v in self.label2idx.items()}
            
            self.y_tr = [self.label2idx[y[0]] for y in target_tokens]
            self.y_tr = np.array(self.y_tr, dtype=np.int)
            
            with open(path_telabel, encoding="utf-8") as f:
                target_tokens_test = [[line.split('\t')[0].replace('\n', '')] for line in f]
                
            self.y_te = [self.label2idx[y[0]] for y in  target_tokens_test]
            self.y_te = np.array(self.y_te, dtype=np.int)

        elif probing_type == 'avg':
            self.activations_tr, self.num_layers = data_loader.load_activations(self.path+'/activations_train.json', 768)
            self.activations_te, self.num_layers = data_loader.load_activations(self.path+'/activations_te.json', 768)
            self.X_tr = [np.asarray(torch.tensor(act).mean(dim=0)) for act in self.activations_tr]
            self.X_tr = np.asarray(self.X_tr)  
            self.X_te = [np.asarray(torch.tensor(act).mean(dim=0)) for act in self.activations_te]
            self.X_te = np.asarray(self.X_te)
            with open(path_trlabel, encoding="utf-8") as f:
                target_tokens = [[line.split('\t')[0].replace('\n', '')] for line in f]   
            label2idx = tok2idx(target_tokens)
            idx2label = idx2tok(label2idx)
            labels = sorted(label2idx.keys())
            indeces = [int(i) for i in range(len(labels))]
            self.label2idx = dict(zip(labels, indeces))
            self.idx2label = {v: k for k, v in self.label2idx.items()}
            self.y_tr = [self.label2idx[y[0]] for y in target_tokens]
            self.y_tr = np.array(self.y_tr, dtype=np.int)
            with open(path_telabel, encoding="utf-8") as f:
                target_tokens =[[line.split('\t')[0].replace('\n', '')] for line in f]
            self.y_te = [self.label2idx[y[0]] for y in target_tokens]
            self.y_te = np.array(self.y_te, dtype=np.int)

        
    def run_classification(self, return_predictions=False):# just probing
        probe = linear_probe.train_logistic_regression_probe(self.X_tr, self.y_tr, lambda_l1=0.001, lambda_l2=0.01, batch_size=128)
        scores_tr = evaluate_probe(probe, self.X_tr, self.y_tr, idx_to_class=self.idx2label, batch_size=128, metric='f1')
        if return_predictions:
            scores_te, predictions = evaluate_probe(probe, self.X_te, self.y_te, idx_to_class=self.idx2label, batch_size=128, metric='f1', return_predictions=True)
            return probe, scores_tr, scores_te, predictions
        else:
            scores_te = evaluate_probe(probe, self.X_te, self.y_te, idx_to_class=self.idx2label, batch_size=128, metric='f1', return_predictions=False)
            return probe, scores_tr, scores_te
            
    def nranking(self, probe): # ranking of neurons https://github.com/fdalvi/NeuroX/blob/master/neurox/interpretation/linear_probe.py
        ordering, cutoffs = linear_probe.get_neuron_ordering(probe, self.label2idx, search_stride=100)
        return ordering, cutoffs
    
    def top_n(self, probe, percentage=0.2): # only top-percentage neurons
        return linear_probe.get_top_neurons(probe, percentage, self.label2idx) 
    
    def threshold_n(self, probe, fraction=2):
        return linear_probe.get_top_neurons_hard_threshold(probe, fraction, self.label2idx) 
    
    def keep_bottom(self, neurons):
        X_tr_b = deepcopy(self.X_tr)
        X_te_b = deepcopy(self.X_te)
        X_tr_selected = ablation.filter_activations_remove_neurons(X_tr_b, neurons)
        probe_selected = linear_probe.train_logistic_regression_probe(X_tr_selected, self.y_tr, lambda_l1=0.001, lambda_l2=0.01, batch_size=128)
        scores_tr = evaluate_probe(probe_selected, X_tr_selected, self.y_tr, idx_to_class=self.idx2label, batch_size=128, metric='f1')
        
        X_te_selected = ablation.filter_activations_remove_neurons(X_te_b, neurons)
        scores_te = evaluate_probe(probe_selected, X_te_selected, self.y_te, idx_to_class=self.idx2label, batch_size=128, metric='f1')
        return scores_tr, scores_te
    
    def keep_util(self, neurons, X_tr, X_te):
        X_tr_selected = ablation.filter_activations_keep_neurons(X_tr, neurons)
        probe_selected = linear_probe.train_logistic_regression_probe(X_tr_selected, self.y_tr, lambda_l1=0.001, lambda_l2=0.01, batch_size=128)
        scores_tr = evaluate_probe(probe_selected, X_tr_selected, self.y_tr, idx_to_class=self.idx2label, metric='f1')
        X_te_selected = ablation.filter_activations_keep_neurons(X_te, neurons)
        scores_te = evaluate_probe(probe_selected, X_te_selected, self.y_te, idx_to_class=self.idx2label, 
                                                batch_size=128, metric='f1')
        return scores_tr, scores_te
    
    def return_weights(self, probe):
        weights1 = list(probe.parameters())[0].data.cpu()
        weights2 = np.abs(weights1.numpy())
        return weights1, weights2
    
    def layer_wise(self, n: int, metric: str ="f1"): 
        # for probing specific layers
        layer_n_X_tr = ablation.filter_activations_by_layers(self.X_tr, [n], 13)
        probe_layer_n = linear_probe.train_logistic_regression_probe(layer_n_X_tr, self.y_tr, 
                                    lambda_l1=0.001, lambda_l2=0.01, batch_size=128)
        scores_tr = evaluate_probe(probe_layer_n, layer_n_X_tr, self.y_tr, 
                            idx_to_class=self.idx2label, metric=metric, batch_size=128)
        
        layer_n_X_te = ablation.filter_activations_by_layers(self.X_te, [n], 13)
        scores_te = evaluate_probe(probe_layer_n, layer_n_X_te, self.y_te, 
                                idx_to_class=self.idx2label, metric=metric, batch_size=128)
        return probe_layer_n, scores_tr, scores_te
    
    def train_layers(self,metric='f1'):
        scores = {}
        for n in range(13):
            probe_layer_n, scores_tr, scores_te = self.layer_wise(n, metric=metric)
            scores[n] = [scores_tr, scores_te]
        return scores
        
    def keep_only(self, neurons, goal='top'):
        if goal == 'top':
            X_tr_top = deepcopy(self.X_tr)
            X_te_top = deepcopy(self.X_te)
            return self.keep_util(neurons, X_tr_top, X_te_top)
            
        elif goal == 'threshold':
            X_tr_t = deepcopy(self.X_tr)
            X_te_t = deepcopy(self.X_te)
            return self.keep_util(neurons, X_tr_t, X_te_t)  
        