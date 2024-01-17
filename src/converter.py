import re
import os
import torch
import random
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict


class ConvertSample:
    """"
    Gets .csv files, makes train & test split in .txt format.
    """
    
    def __init__(self, path, path_work, train_size=2000, test_size=1000, shuffle: bool = False): 

        self.shuffle = shuffle
        self.path = path
        self.path_work = path_work
        self.project_path = str(Path(os.getcwd()).parents[0])
        self.category = re.search(r'[a-zA-Z]+_[a-zA-Z]+(?=.txt)', path)[0]
        self.train_size = train_size
        self.test_size = test_size
        

    def read(self) -> list: 
        with open(self.path, encoding="utf-8") as f:
            lines = [line.split('\t') for line in f]
            return lines
    
    def sampler(self) -> dict: # data sampling
        
        sents = self.read()
        all_values = []
        all_sents = []
        
        for idx, line in enumerate(sents):
            if idx % 2 == 0:
                line = re.sub('  ', ' ', line[0])
                all_sents.append(line)
            else:
                line = line[0].replace("\'", "").replace(",", "")
                all_values.append(line)
        
        sents_train = all_sents[:self.train_size]
        values_train = all_values[:self.train_size]
        sents_test = all_sents[-self.test_size:]
        values_test = all_values[-self.test_size:]
        
        train_dict = OrderedDict(zip(sents_train, values_train))
        test_dict = OrderedDict(zip(sents_test, values_test))
        return train_dict, test_dict
        
    def using_shuffle(self, a):
        keys = list(a.keys())
        values = list(a.values())
        random.seed(12345)
        new_values = []
        for i in values:
            i = i.split()
            random.shuffle(i)
            new_values.append(i)
        new_values2 = []
        for j in new_values:
            new_values2.append(" ".join(str(x) for x in j)+'\n')
        d = OrderedDict(zip(keys, new_values2))
        return d

    def create_dicts(self):
        dict_filter_train, dict_filter_test = self.sampler()
        dict_control_task = dict_filter_train.copy()
        dict_control_task = self.using_shuffle(dict_control_task)
        return dict_filter_train, dict_filter_test, dict_control_task


    def create_paths(self) -> str:
        
        if re.search(r'(?<=\/)[a-zA-Z][a-zA-Z]_[a-zA-Z]+(?=_)', self.path)[0]:
            dataset = re.search(r'(?<=\/)[a-zA-Z][a-zA-Z]_[a-zA-Z]+(?=_)', self.path)[0]
            path = self.path_work+f'/large_data_{dataset}'
        else:
            path = self.path_work+'/large_data'
            
        if not os.path.isdir(path):
            os.mkdir(path)
            
        if not os.path.isdir(path+f'/data_{self.category}'):
            os.mkdir(path+f'/data_{self.category}')
        
        result_path_datatrain = path+f"/data_{self.category}/datatrain_{self.category}.txt"
        result_path_labeltrain = path+f"/data_{self.category}/labeltrain_{self.category}.txt"
        
        result_path_cdatatrain = path+f"/data_{self.category}/cdatatrain_{self.category}.txt"
        result_path_clabeltrain = path+f"/data_{self.category}/clabeltrain_{self.category}.txt"
        
        result_path_datatest = path+f"/data_{self.category}/datatest_{self.category}.txt"
        result_path_labeltest = path+f"/data_{self.category}/labeltest_{self.category}.txt"

        return result_path_datatrain, result_path_labeltrain, result_path_cdatatrain, result_path_clabeltrain, \
               result_path_datatest, result_path_labeltest

    def writer(self) -> str: 
        """
        Writes to a file
        """
        result_datatrain, result_labeltrain, result_cdatatrain, result_clabeltrain, result_datatest, result_labeltest = self.create_paths()
       
        
        dict_filter_train, dict_filter_test, dict_control_task = self.create_dicts()

        with open(result_datatrain, "w", encoding="utf-8") as traindata, \
             open(result_labeltrain, "w", encoding="utf-8") as trainlabel, \
             open(result_cdatatrain, "w", encoding="utf-8") as ctraindata, \
             open(result_clabeltrain, "w", encoding="utf-8") as ctrainlabel, \
             open(result_datatest, "w", encoding="utf-8") as testdata, \
             open(result_labeltest, "w", encoding="utf-8") as testlabel:
            
            for sentence, value in dict_filter_train.items():
                traindata.writelines(sentence)
                trainlabel.writelines(value)

            for sentence, value in dict_control_task.items():
                ctraindata.writelines(sentence)
                ctrainlabel.writelines(value)

            for sentence, value in dict_filter_test.items():
                testdata.writelines(sentence)
                testlabel.writelines(value)
                                                                    
        return result_datatrain, result_labeltrain, result_cdatatrain, result_clabeltrain, result_datatest, result_labeltest