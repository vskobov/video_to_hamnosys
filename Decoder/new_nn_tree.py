import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import pathlib
import argparse
import numpy as np
import vector2sigml.v2s
from ete3 import Tree
import os
import glob
import joblib

BATCH_SIZE=1
EPOCHS=2000000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DEVICE = torch.device("cpu")
LR = 0.00005
MAX_LEVEL = 3
TEST_SIZE = 1100
parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')

args = parser.parse_args()

path = args.output_dest[0]


class SignDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, mod_name, add_name):
        'Initialization'
        self.mod_name = mod_name
        self.add_name = add_name
        self.files = glob.glob(path+'input_X_1d_np_'+self.mod_name+self.add_name+'*')
        self.num_packages = len(self.files)
        self.indexes = self.get_indexes()
        self.x_vector_len, self.y_vector_len, self.input_len = self.get_vec_len()
        
    
    def get_vec_len(self):
        x_vec_len = np.load(path+'input_X_1d_np_'+self.mod_name+self.add_name+str(1)+'.npy').shape[1]
        y_vec_len = np.load(path+'input_Y_1d_np_'+self.mod_name+self.add_name+str(1)+'.npy').shape[1]
        if x_vec_len >= y_vec_len:
        	input_len = x_vec_len
        else:
        	input_len = y_vec_len
        return x_vec_len, y_vec_len ,input_len
        
    def get_indexes(self):
        indexes = []
        idx = 0
        for package in range(0,self.num_packages):
        	pac_len = np.load(path+'input_X_1d_np_'+self.mod_name+self.add_name+str(package+1)+'.npy').shape[0]
        	for i in range(0, pac_len):
        		indexes.append((package, i))
        return indexes
        

    def __len__(self):
        'Denotes the total number of samples'
        total_len = 0
        for package in range(0,self.num_packages):
        	total_len += np.load(path+'input_X_1d_np_'+self.mod_name+self.add_name+str(package+1)+'.npy').shape[0]
        return total_len
    
    def __getitem__(self, index):

        package, x = self.indexes[index]
        
        if self.x_vector_len >= self.y_vector_len:
            X_sample = torch.tensor(np.load(path+'input_X_1d_np_'+self.mod_name+self.add_name+str(package+1)+'.npy')[x],dtype=torch.float)
            Y_sample = torch.tensor(np.hstack((np.load(path+'input_Y_1d_np_'+self.mod_name+self.add_name+str(package+1)+'.npy')[x], np.zeros(self.x_vector_len-self.y_vector_len, ))),dtype=torch.float)
        else:
            X_sample = torch.tensor(np.hstack((np.load(path+'input_X_1d_np_'+self.mod_name+self.add_name+str(package+1)+'.npy')[x],np.zeros(self.y_vector_len-self.x_vector_len,))),dtype=torch.float)
            Y_sample = torch.tensor(np.load(path+'input_Y_1d_np_'+self.mod_name+self.add_name+str(package+1)+'.npy')[x],dtype=torch.float)

        return X_sample, Y_sample


def load_data(mod_name):
    d_train = SignDataset(mod_name, '_noized_')
    train_loader = torch.utils.data.DataLoader(d_train,batch_size=BATCH_SIZE, shuffle=True, num_workers = 6)
    print('Training SET Size loaded: '+ str(d_train.__len__()))
   
    d_valid = SignDataset(mod_name, '_valid_')
    test_loader = torch.utils.data.DataLoader(d_valid,batch_size=BATCH_SIZE, shuffle=True, num_workers = 6)
    print('Valiadation SET Size loaded: '+ str(d_valid.__len__()))
    
    return (train_loader,test_loader,d_train.input_len)
    