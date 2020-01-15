import datetime
from time import sleep
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
import h5py
import random, time
import multiprocessing
from tabulate import tabulate 

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class Sequential(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.drop = nn.Dropout(0.1)
        #self.fc = nn.Linear(self.input_size,self.output_size)
        self.fc = nn.Linear(self.input_size,self.output_size)
        #self.fc1 = nn.Linear(2*self.input_size,2*self.input_size)
        #self.fc2 = nn.Linear(2*self.input_size,self.output_size)
        
    def forward(self,x):
        out = self.drop(x)
        #out = self.fc(x)
        #out = nn.LeakyReLU(0.01)(out)
        #out = nn.Softmax(dim=1)(out)
        
        out = self.fc(x)
        out = nn.LeakyReLU(0.01)(out)
        out = nn.Sigmoid()(out)
        #out = nn.ReLU()(out)
        #out = nn.Softmax(dim=1)(out)
    
        #out = self.fc1(out)
        #out = nn.LeakyReLU(0.01)(out)
        #out = nn.ReLU()(out)
        #out = nn.Sigmoid()(out)

        #out= self.fc2(out)
        #out = nn.ReLU()(out)
        #out = nn.LeakyReLU(0.01)(out)
        #out = nn.Softmax(dim=1)(out)
        #out = nn.LeakyReLU(0.01)(out)

        return out

class H5Dataset(data.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        #print(file_path)
        h5_file = h5py.File(file_path, 'r')
        self.data = h5_file['X_data']
        self.target = h5_file['Y_data']
        self.input_len = self.data[0].shape[0]
        self.tree_len = self.target[0].shape[0]

    def __getitem__(self, index): 
        #x_noized = data_noizer_waky(self.data[index])
        #return (torch.from_numpy(x_noized).float(),torch.from_numpy(self.target[index]).float())
        return (torch.from_numpy(self.data[index]).float(),torch.from_numpy(self.target[index]).float())

    def __len__(self):
        return self.data.shape[0]

class H5Dataset_level(data.Dataset):

    def __init__(self, file_path, opt_t):
        super(H5Dataset_level, self).__init__()
        #print(file_path)
        
        h5_file = h5py.File(file_path, 'r', libver='latest')
        self.data = h5_file['X_data']
        self.target = h5_file['Y_data']
        self.input_len = self.data[0].shape[0]
        self.tree_len = self.target[0].shape[0]
        self.opt_data, self.opt_target, self.pool = self.find_node_data(opt_t)
        


    def find_node_data(self, opt_t):

        def missing_rate(self,inp):
            all_v = len(inp)
            missing_v = 0
            for i in range(0, all_v):
                if inp[i] == 0:
                    missing_v += 1
            rate = missing_v/all_v
            return rate

        #print('Finding target ')
        o_d = []
        o_t = []
        #all_rates = 0
        #classes = len(opt_t)
        pool = [[] for c in range(0,len(opt_t))]
        for i in range(0, self.data.shape[0]):
            mod_target = []
            append = False
            bucket = None
            for ind in range(0,len(opt_t)):
                #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                temp_ind = self.target[i][opt_t[ind][0]:opt_t[ind][1]].tolist()
                if any(x == 1.0 for x in temp_ind):
                    mod_target.append(1.0)
                    #print(pool)
                    #if min(pool) == pool[ind]:
                    #pool[ind] += 1
                    #if append==True:
                    #    print('dip')
                    append = True
                    bucket = ind
                else:
                    mod_target.append(0.0)
                del(temp_ind)
            if append:
                #o_d.append(self.data[i])
                #o_t.append(np.array(mod_target))
                pool[bucket].append((self.data[i],np.array(mod_target)))
            #print('Loading training data:',(int((i/self.data.shape[0])*100)),'%',end='\r')
                #all_rates += missing_rate(self,self.data[i])
        min_b_list = min(list([len(x) for x in pool]))
        #print(list([len(x) for x in pool]))
        for c in pool:
            random.shuffle(c)

        for b in range(len(pool)):
            for i in range(min_b_list):
                for n in range(4):
                    o_d.append( data_noizer_waky( pool[b][i][0] ) )
                    o_t.append(pool[b][i][1])
                o_d.append(pool[b][i][0])
                o_t.append(pool[b][i][1])
        pool = min_b_list
        #print('Av. Missing rate : '+str(all_rates/len(o_d)))
        #print(pool)
        #print('Found  '+str(len(o_d))+' samples')
        #o_d, mask = mask_samples(torch.tensor(o_d))
        #print(mask.numpy())
        #print(str(int(sum(mask.numpy())/len(mask.numpy()))))
        #o_d = np.stack(o_d, axis=0)
        #o_t = np.stack(o_t, axis=0)
        return o_d, o_t, pool

    def __getitem__(self, index): 
        #x_noized = data_noizer_waky(self.opt_data[index])
        #return (torch.from_numpy(x_noized).float(),torch.from_numpy(self.opt_target[index]).float())
        return (torch.from_numpy(self.opt_data[index]).float(),torch.from_numpy(self.opt_target[index]).float())

    def __len__(self):
        return len(self.opt_data)

class H5Dataset_level_validation(data.Dataset):

    def __init__(self, file_path,file_path_2, opt_t):
        super(H5Dataset_level_validation, self).__init__()
        #print('Valid'+file_path)
        h5_file = h5py.File(file_path, 'r', libver='latest')
        h5_file_2 = h5py.File(file_path_2, 'r', libver='latest')
        self.data = h5_file['X_data'] 
        self.data_2 = h5_file_2['X_data']
        self.target = h5_file['Y_data']
        self.target_2 = h5_file_2['Y_data']
        self.input_len = self.data[0].shape[0]
        self.tree_len = self.target[0].shape[0]
        self.opt_data, self.opt_target, self.pool = self.find_node_data(opt_t)

    def find_node_data(self, opt_t):

        def missing_rate(self,inp):
            all_v = len(inp)
            missing_v = 0
            for i in range(0, all_v):
                if inp[i] == 0:
                    missing_v += 1
            rate = missing_v/all_v
            return rate

        #print('Finding target ')
        o_d = []
        o_t = []
        #all_rates = 0
        count = 0
        base = self.data.shape[0] + self.data_2.shape[0]
        pool = [[] for c in range(0,len(opt_t))]

        for i in range(0, self.data.shape[0]):
            mod_target = []
            append = False
            bucket = None
            for ind in range(0,len(opt_t)):
                #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                temp_ind = self.target[i][opt_t[ind][0]:opt_t[ind][1]].tolist()
                if any(x == 1.0 for x in temp_ind):
                    mod_target.append(1.0)
                    #if min(pool) == pool[ind]:
                    #pool[ind] += 1
                    #if append==True:
                    #    print('double double')
                    append = True
                    bucket = ind
                else:
                    mod_target.append(0.0)
                del(temp_ind)
            if append:
                #o_d.append(self.data[i])
                #o_t.append(np.array(mod_target))
                pool[bucket].append((self.data[i],np.array(mod_target)))
            count+=1

            #print('Loading validation data:',(int((count/base)*100)),'%',end='\r')

        for i in range(0, self.data_2.shape[0]):
            mod_target = []
            append = False
            bucket = None
            for ind in range(0,len(opt_t)):
                #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                temp_ind = self.target_2[i][opt_t[ind][0]:opt_t[ind][1]].tolist()
                if any(x == 1.0 for x in temp_ind):
                    mod_target.append(1.0)
                    #if min(pool) == pool[ind]:
                    #pool[ind] += 1
                    #if append==True:
                    #    print('triple double')
                    append = True
                    bucket = ind
                else:
                    mod_target.append(0.0)
                del(temp_ind)
            if append:
                #o_d.append(self.data[i])
                #o_t.append(np.array(mod_target))
                pool[bucket].append((self.data_2[i],np.array(mod_target)))
            count+=1
            #print('Loading validation data:',(int((count/base)*100)),'%',end='\r')

        min_b_list = min(list([len(x) for x in pool]))
        #print(list([len(x) for x in pool]))
        
        for i in range(min_b_list):
            for b in range(len(pool)):
                o_d.append(pool[b][i][0])
                o_t.append(pool[b][i][1])

        #print(list([len(x) for x in pool]), 'min', min_b_list,'data',len(o_d))
        pool = min_b_list
            
        #for i in range(0, self.data_2.shape[0]):
        #    mod_target = []
        #    append = False
        #
        #    for ind in range(0,len(opt_t)):
        #        #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
        #        temp_ind = self.target_2[i][opt_t[ind][0]:opt_t[ind][1]].tolist()
        #        if any(x == 1.0 for x in temp_ind):
        #            mod_target.append(1.0)
        #            append = True
        #        else:
        #            mod_target.append(0.0)
        #        del(temp_ind)
        #    if append:
        #        o_d.append(self.data_2[i])
        #        o_t.append(np.array(mod_target))

                #all_rates += missing_rate(self,self.data[i])

            #print(str(int(i/self.data.shape[0]*100))+' %', end='\r')
        #print('Av. Missing rate : '+str(all_rates/len(o_d)))
        #print(pool)
        #print('Found  '+str(len(o_d))+' samples')
        #o_d = np.stack(o_d, axis=0)
        #o_t = np.stack(o_t, axis=0)
        return o_d, o_t, pool

    def __getitem__(self, index): 
        #x_noized = data_noizer_waky(self.opt_data[index])
        #return (torch.from_numpy(x_noized).float(),torch.from_numpy(self.opt_target[index]).float())
        return (torch.from_numpy(self.opt_data[index]).float(),torch.from_numpy(self.opt_target[index]).float())

    def __len__(self):
        return len(self.opt_data)

def l_counter(leaf):
    #global tree_leaves
    #global leaves_counter
    leaves_counter[tree_leaves.index(leaf)] += 1
    return 

def init_weights(m):
    if type(m) == nn.Linear:
        #torch.nn.init.constant_(m.weight, 0.0)
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0.0)

def cosine_similarity(source_word_vector,target_word_vector):
    """
    >>> cosine_similarity([1,0],[1,0])
    1.0
    >>> cosine_similarity([0,1],[1,0])
    0.0
    >>> cosine_similarity([-1,0],[1,0])
    -1.0
    """

    up = []
    down_source = []
    down_target = []

    for c in range(0,len(source_word_vector)): 
        up.append(source_word_vector[c]*target_word_vector[c]) #upper part of the formula
        down_source.append(source_word_vector[c]*source_word_vector[c]) # lower part of the formula
        down_target.append(target_word_vector[c]*target_word_vector[c])

    dot = sum(up)
    norms = ((sum(down_source))**0.5) * ((sum(down_target))**0.5)

    similarity = dot / norms

    return torch.tensor(similarity,requires_grad=False).type(torch.float) #remember to output 3 digits "%.3f" %  

def validation_loss(model,device, dataloader):
    model.to(device)
    val_loss = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(dataloader):
        d, tr = data.to(device,non_blocking=True),target.to(device,non_blocking=True)
        #try:
        #d, tr = data.to(device,non_blocking=True), torch.tensor([(tar==1.0).nonzero() for tar in target],dtype=torch.long).to(device,non_blocking=True)
        #except:
        #    print(target)
        output = model(d)
        #loss = nn.CrossEntropyLoss(reduction='mean')
        #loss = nn.BCELoss(reduction='mean')
        loss = nn.MSELoss(reduction='mean')
        #loss = nn.MSELoss(reduction='mean')
        #val_loss += loss(output, tr.type(torch.long)).float().detach()
        val_loss += loss(output, tr).float().detach()
    return val_loss

def train_one_node_full(model, device, dataloader , optimizer, scheduler, testdataloader): 
    #print('Start node training .. ')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_base = 0
        net = torch.nn.DataParallel(model) #for multi GPUs
        for batch_idx, (data, target) in enumerate(dataloader):
            base = 0
            #d, tr = data.to(device,non_blocking=True), torch.tensor([(tar==1.0).nonzero() for tar in target],dtype=torch.long).to(device,non_blocking=True)
            d, tr = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
            output = net(d)
            #nll_t = torch.tensor([(tar==1.0).nonzero() for tar in tr],dtype=torch.long).to(device,non_blocking=True)
            #loss = nn.NLLLoss(reduction='sum')
            loss = nn.MSELoss(reduction='mean')
            #loss = nn.CrossEntropyLoss(reduction='mean')
            #loss = nn.BCELoss(reduction='mean')
            loss_a = loss(output, tr)
            loss_a.backward()
            optimizer.step()
            optimizer.zero_grad()
            pred = output

            if len(pred.shape)>1:
                max_p = max(pred[0])
                for p in range(0,len(pred[0])):
                    if pred[0][p] == max_p:
                        pred[0][p] = 1.0
                    else:
                        pred[0][p] = 0.0
            else:
                max_p = max(pred)
                for p in range(0,len(pred)):
                    if pred[p] == max_p:
                        pred[p] = 1.0
                    else:
                        pred[p] = 0.0
            base  += len(target[0])
            #print_str = "Training epoch "+str(epoch) + ' ' + '%0.2f' % (100. * batch_idx/len(dataloader))+ '%'  
            #print(print_str, end='\r')
            #del(print_str)
            epoch_loss += float(loss_a.detach())
            epoch_base += base
            del(output,pred,loss_a,d)
        total_loss = 100. * epoch_loss/epoch_base
        if testdataloader != None:
            val_loss = validation_loss(model,device,testdataloader)
            scheduler.step(val_loss)
        else:
            scheduler.step(epoch_loss)
        #total_enc_loss = 100. * epoch_enc_loss/epoch_enc_base
        if epoch == EPOCHS:
        #if epoch%1==0:
            if testdataloader != None:
                #print("Total epoch loss: "+str(total_loss))
                #print("Total epoch encoder loss: "+str(total_enc_loss))
                #if epoch%1==0:
                res = level_node_test(testdataloader,model)
                out_len = res[0]
                mean_accuracy = res[1]
                mean_recall = res[2]
                mean_precision = res[3]
                mean_f1_score = res[4]
                mean_fully = res[5]
                print('TEST Result: Classes: '+str(out_len)+' Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully))
                return res
            else:
                return (None,None,None,None,None,None)

def quick_test(tree, data):
    print('Target :'+str(data))
    test_vec  = get_test_np_vector(tree,data)
    print('Output :'+str(test_vec))
    return 

def get_test_np_vector(tt, data):
    tree = tt
    tree_leaves = list([ln for ln in tree.iter_leaves()])
    #print(len(tree_leaves))
    test_walker(tree, data)
    r_vector =[0. for i in range(0,len(tree_leaves))]
    i = 0
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_vector[i] = 1.
        i = i + 1
    return r_vector

def prepare_for_multitrain(name_tree, tree, start_ind, input_len, level):
    #if level == MAX_LEVEL:
    #    return
    if tree.is_leaf():
        #l_counter(tree)
        #tree.name = tree.name +'*'
        return
    else:
        #next_step = []
        choicer = False

        indicies = [start_ind]

        for i in range(1, (len(tree.children)+1)): #Cumulative indicies 
            tmp_leaves = list(tree.children[i-1].iter_leaves())
            indicies.append(indicies[i-1]+len(tmp_leaves))
            del(tmp_leaves)


        for i in range(0, (len(tree.children))):
            if tree.children[i].name.find('OPT') != -1:
                choicer = True

        if choicer:
            options = []
            train_choice = Tree()
            
            for i in range(0, (len(tree.children))):
                if tree.children[i].name.find('OPT') != -1 and tree.children[i].name != "NON_OPT":
                        options.append(tree.children[i])

            if len(options)>1:
                mod_target = []
                
                for ind in range(0,len(indicies)-1):
                    #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                    mod_target.append((indicies[ind], indicies[ind+1]))
                    #del(temp_ind)
                #print(mod_target)

                assert len(mod_target) == len(tree.children), 'TRAIN: Target lenght is not equal to children lenght' 

                opt_target = []
                for o_i in range(0,len(tree.children)):
                    if tree.children[o_i].name.find('OPT') != -1 and tree.children[o_i].name != "NON_OPT":
                        opt_target.append(mod_target[o_i])

                assert len(opt_target) == len(options), 'TRAIN: Options Target lenght is not equal to options lenght'

                tree.add_feature('level',level)
                tree.add_feature('opt_target',opt_target)
                tree.add_feature('classes',len(opt_target))
                tree.add_feature('done_epochs',0)
                #joblib.dump(tree.get_tree_root(), path+"../models/nn_tree_"+name_tree+".joblib")
                global submodels_done
                global submodels_total
                submodels_done += 1
                print('Done :'+str(submodels_done),end='\r')

                level = level + 1
                del(opt_target,mod_target)

            del(options)
            for o_i in range(0,len(tree.children)):
                    prepare_for_multitrain(name_tree,tree.children[o_i],indicies[o_i],input_len,(level))
            del(indicies)
            
            
        else:

            for o_i in range(0,len(tree.children)):
                prepare_for_multitrain(name_tree,tree.children[o_i],indicies[o_i],input_len,level)
            del(indicies)

def test_walker(tree, data):

    if tree.is_leaf():
        #l_counter(tree)
        tree.name = tree.name +'*'
        return
    else:

        choicer = False

        for o in tree.children:
            if o.name.find('OPT') != -1:
                choicer = True

        if choicer:
            options = []
            weights = []
            trained_choice = Tree()
            for o in tree.children:
                if o.name.find('OPT') != -1 and o.name != "NON_OPT":
                        options.append(o)
            if len(options)>1:
                t_model = tree.model.to(DEVICE)
                t_model.eval()
                output = t_model(data.to(DEVICE))
                output = output[0].tolist()
                #print(output)
                assert len(options) == len(output), 'TEST: Options length '+str((len(options)))+' is not equal to the model output lenght '+str((len(output)))
                
                for o_i in range(0,len(options)):
                    if output[o_i] == max(output):
                        trained_choice = options[o_i]
                if tree.level == 0 or tree.level == 1:
                    #print('!')
                    trained_choice = options[0]
                        #print(str(output)+" "+str(output[o_i]))
                
                del(t_model)
            if len(options)==1:
                trained_choice = options[0]
            del(options)
            del(weights)
            for child in tree.children:
                if child == trained_choice or child.name == "NON_OPT":
                    test_walker(child,data)
        else:
            for child in tree.children:
                
                test_walker(child, data)

def test(name_addition, test_loader):
    print('Testing .. ',end='\r')
    global subclasses_total
    tree = joblib.load(path+"../models/3_nn_tree_"+name_addition+".joblib")
    scores = []
    for batch_idx, (test_data, target) in enumerate(test_loader):
        test_vec  = np.array(get_test_np_vector(tree,test_data))
        if len(test_vec) < len(target[0]):
            target = target[0].cpu().numpy()[:len(test_vec)]
        else:
            target = target[0].cpu().numpy()
        scores.append(calculate_scores(test_vec,target))
        #print(test_vec)
        #print(target)
        #test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        #test_v.save_sigml('./output_rs/pred_saved_all_nn_tree_'+str(batch_idx)+'.txt','predicted')

        #test_vec_tar = vector2sigml.v2s.Vec2sigml(target)
        #test_vec_tar.save_sigml('./output_rs/target_saved_all_nn_tree_'+str(batch_idx)+'.txt','target')
        #print("Saved "+str(batch_idx), end='\r')
    mean_accuracy = 100.* sum(s[0] for s in scores)/len(scores)
    mean_recall = sum(s[1] for s in scores)/len(scores)
    mean_precision = sum(s[2] for s in scores)/len(scores)
    mean_f1_score = sum(s[3] for s in scores)/len(scores)
    mean_fully = 100.* sum(s[4] for s in scores)/len(scores)
    print('Accuracy across all',subclasses_total,'Subclasses:', round(mean_accuracy), '%')
    del(tree,scores)
    return

def annotate(tree_path, data_path):
    print('Making annotations..')
    tree = joblib.load(path+tree_path)
    data_file = h5py.File(path+data_path, 'r')
    for vec in range(len(data_file['X_data'])):
        test_vec  = np.array(get_test_np_vector(tree,torch.from_numpy(data_file['X_data'][vec]).float(),False))
        test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        test_v.save_sigml('./output_greek/pred_saved_all_nn_tree_'+str(vec)+'.txt',str(vec))
    return

def level_node_test(test_loader, model):
    #print('Testing .. ',end='\r')
    scores = []
    out_len = 0
    model.eval()
    pool = []
    #hidden = model.test_initHidden()
    for batch_idx, (test_data, target) in enumerate(test_loader):
        d = test_data.to(DEVICE,non_blocking=True)

        test_vec  = model(d)
        #test_vec = torch.cat([t for t in test_vec], dim=1)
        test_vec = test_vec[0].detach().cpu().numpy()
        tar_vec = target[0].cpu().numpy()
        
        assert len(test_vec) == len(tar_vec), 'Something wrong '+ str(len(test_vec))+' '+str(len(tar_vec))
        #print(len(tar_vec))
        pred = test_vec
        for p in range(0,len(pred)):
                if pred[p] == max(test_vec):
                    pred[p] = 1.0
                else:
                    pred[p] = 0.0
        test_vec = pred 
        del(pred)

        if batch_idx == 0:
            out_len = len(tar_vec)
            #print(tar_vec)
            #print(test_vec)
        #if len(test_vec) < len(tar_vec):
        #    target = tar_vec[:len(test_vec)]
        #else:
        #    target = tar_vec
        scores.append(calculate_scores(test_vec,tar_vec))
        #print(test_vec)
        #print(target)
        #test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        #test_v.save_sigml(path+'../pred_saved_all_nn_tree_'+str(batch_idx)+'.txt','predicted')

        #test_vec_tar = vector2sigml.v2s.Vec2sigml(target[0].cpu().numpy())
        #test_vec_tar.save_sigml(path+'../target_saved_all_nn_tree_'+str(batch_idx)+'.txt','target')
        #print("Saved "+str(batch_idx), end='\r')
    mean_accuracy = int(100.* sum(s[0] for s in scores)/len(scores))
    mean_recall = int(100.*sum(s[1] for s in scores)/len(scores))
    mean_precision = int(100.*sum(s[2] for s in scores)/len(scores))
    mean_f1_score = int(100.*sum(s[3] for s in scores)/len(scores))
    mean_fully = int(100.* sum(s[4] for s in scores)/len(scores))
    #print('TEST: Classes: '+str(out_len)+' Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully) + ' ('+str(sum(s[4] for s in scores))+')'+' across '+str(len(scores))+ ' samples' )
    del(scores)
    return (out_len,mean_accuracy,mean_recall,mean_precision,mean_f1_score,mean_fully)

def calculate_scores(prediction, target):
    if (prediction == target).all():
        full = 1
    else:
        full = 0
    tp,tn,fn,fp = [],[],[],[]
    for t in range(0, len(target)):
        if prediction[t]==target[t] and target[t] == 1:
            tp.append(1)
        if prediction[t]==target[t] and target[t] == 0:
            tn.append(1)
        if prediction[t]!=target[t] and target[t] == 1:
            fn.append(1)
        if prediction[t]!=target[t] and target[t] == 0:
            fp.append(1)
    if sum(tp)>0:
        accuracy = (sum(tp)+sum(tn))/(sum(tp)+sum(tn)+sum(fp)+sum(fn))
        recall = sum(tp)/(sum(tp)+sum(fn))
        precision = sum(tp)/(sum(tp)+sum(fp))
        f1_score = 2*(recall * precision)/(recall + precision)
        return (accuracy, recall, precision, f1_score, full)
    else:
        return (0,0,0,0,full)

def data_noizer_waky(X_data):
    noise = np.random.normal(1, 0.03, len(X_data))
    #rand = random.uniform(0.95,1.)
    #noise  = np.empty(len(X_data))
    #noise.fill(rand)
    for x in range(0,len(X_data)):
        if X_data[x] == 1:
            noise[x] = 1
    X_noized_data = X_data * noise
    return X_noized_data

def data_noizer(X_data):
    #noise = np.random.normal(1, 0.1, len(X_data))
    rand = random.uniform(0.9,1.1)
    noise  = np.empty(len(X_data))
    noise.fill(rand)
    for x in range(0,len(X_data)):
        if X_data[x] == 1:
            noise[x] = 1
    X_noized_data = X_data * noise
    return X_noized_data

def load_h5_data(mod_name):
    d_train = H5Dataset(path+'train_input_keys_1d_np_'+mod_name+"_h5.hdf5")
    #train_loader = torch.utils.data.DataLoader(d_train,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
    train_loader = []
    print('Training SET Size loaded: '+ str(d_train.__len__()))
   
    d_valid = H5Dataset(path+'test_input_keys_1d_np_'+mod_name+"_h5.hdf5")
    test_loader = torch.utils.data.DataLoader(d_valid,batch_size=1, shuffle=True, num_workers = 1)
    #test_loader = []
    print('Valiadation SET Size loaded: '+ str(d_valid.__len__()))
    
    return (train_loader, test_loader, d_train.input_len, d_train.tree_len)

def count_classes(tree):
    if tree.is_leaf():
        return
    else:
        choicer = False

        for i in range(0, (len(tree.children))):
            if tree.children[i].name.find('OPT') != -1:
                choicer = True

        if choicer:
            options = []
            
            for i in range(0, (len(tree.children))):
                if tree.children[i].name.find('OPT') != -1 and tree.children[i].name != "NON_OPT":
                        options.append(tree.children[i])

            if len(options)>1:
                global submodels_total
                global subclasses_total
                subclasses_total+=len(options)
                submodels_total+=1
            del(options)
            for o_i in range(0,len(tree.children)):
                count_classes(tree.children[o_i])
        else:
            for o_i in range(0,len(tree.children)):
                count_classes(tree.children[o_i])

def print_table_results(tree):
    def Average(lst): 
        average = sum(lst) / len(lst) 
        return int(average)
    levels = []
    res = []
    n_t = 0
    for node in tree.traverse("levelorder"):
        if hasattr(node, 'train_result'):
            if node.train_result != None:
                levels.append(node.level)
                res.append( ( node.level, node.samples_per_class, node.classes, node.train_result[1] ) )
                #print('Level : '+str(node.level)+' Samp/Class: '+str(node.samples_per_class)+' Classes: '+str(node.classes)+' Result '+str(node.train_result))
            else:
                n_t +=1
                res.append( ( node.level, node.samples_per_class, node.classes, None ) )
    print('Results found',len(res), 'Nodes without any training:', n_t)
    levels_set  = sorted(set(levels))
    del(levels)
    tab= []
    for level in levels_set:
        level = [level+1, len(list([s for s in res if s[0]==level])), Average(list([s[1] for s in res if s[0]==level])), Average(list([s[2] for s in res if s[0]==level])),round(np.average(list([s[3] for s in res if s[0]==level and s[3] != None])))]
        tab.append(level)
    tab.append(['All Levels', len(list(res)), round(np.average(list([s[1] for s in res]))), round(np.average(list([s[2] for s in res]))),round(np.average(list([s[3] for s in res if s[3] != None])))])
    print(tabulate(tab,tablefmt='latex',headers=['Tree level','SM','Avg. N/SC','Avg. SC','Avg. Valid Accuracy']))

def multi_node_train(tree):
    #print('Train node: '+str(tree.number))
    start_time = datetime.datetime.now()
    valid_dataset_ready = False
    node_dataset_ready = False
    global lock_train_data, lock_val_data

    while node_dataset_ready == False:
        if lock_train_data.value == 0:
            try:
                lock_train_data.value = 1
                node_dataset = H5Dataset_level(path+'train_input_keys_1d_np_'+mod_name+"_h5.hdf5",tree.opt_target)
                if node_dataset.pool >0:
                    train_dataloader = torch.utils.data.DataLoader(node_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
                else: 
                    train_dataloader = None
                node_dataset_ready = True
                lock_train_data.value = 0
            except: pass
        #else:
         #   sleep(random.uniform(0.05,0.03))

    while valid_dataset_ready == False:
        if lock_val_data.value == 0:
            try: 
                lock_val_data.value = 1
                valid_dataset = H5Dataset_level_validation(path+'test_input_keys_1d_np_'+mod_name+"_h5.hdf5", path+'rs_test_input_keys_1d_np_'+mod_name+"_h5.hdf5",tree.opt_target)
                if valid_dataset.pool > 0:
                    test_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers = 1)
                else:
                    test_dataloader = None
                valid_dataset_ready = True
                lock_val_data.value = 0
            except: pass
        #else:
         #   sleep(random.uniform(0.05,0.03))

    if train_dataloader != None:
        print('Node number',tree.number,'Train samples per class:',node_dataset.pool,' Validation samples per class:',valid_dataset.pool)
        model_t = Sequential(loader[2],len(tree.opt_target)).to(DEVICE,non_blocking=True)
        model_t.apply(init_weights)
        optimizer_t = torch.optim.Adam(model_t.parameters(),LR)
        scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'min')
        train_result = train_one_node_full(model_t, DEVICE, train_dataloader,optimizer_t, scheduler_t, test_dataloader)
    else:
        print('Node',tree.number,'NO DATA')
        model_t = Sequential(loader[2],len(tree.opt_target)).to(DEVICE,non_blocking=True)
        model_t.apply(init_weights)
        optimizer_t = torch.optim.Adam(model_t.parameters(),LR)
        scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'min')
        train_result = None

    tree.add_feature('model',model_t)
    tree.add_feature('scheduler',scheduler_t)
    tree.add_feature('train_result',train_result)
    tree.add_feature('samples_per_class',int(node_dataset.pool))
    tree.add_feature('optimizer',optimizer_t)
    del(model_t,node_dataset,train_dataloader,optimizer_t)

    tree.done_epochs += EPOCHS
    tree.add_feature('done_training',True)
    joblib.dump(tree, path+"../models/nodes/"+str(tree.number)+".joblib")
    global submodels_done
    global submodels_total
    submodels_done.value += 1
    time_ = datetime.datetime.now() - start_time
    print(str(int(submodels_done.value/submodels_total*100))+'% trained '+str(submodels_done.value)+'/'+str(submodels_total)+' Aprox. Time left: '+str((time_*(submodels_total-submodels_done.value)/(multiprocessing.cpu_count()-1))))
    return


#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda")#use that for multiprocessing

LR = 0.00321
BATCH_SIZE=32
EPOCHS=800

parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')
args = parser.parse_args()
path = args.output_dest[0]

print("NEURAL NETWORK TREE")
print("EPOCHS: "+str(EPOCHS)+" DEVICE: "+str(DEVICE)+" LR: "+str(LR))

print("\n")
print('HAND CONFIGURATION:')


submodels_total = 0
subclasses_total = 0
submodels_done = 0

loader = load_h5_data('lr_hand_conf') #'all_h_conf'
vec = vector2sigml.v2s.Vec2sigml(np.ones_like(loader[3]))
count_classes(Tree(vec.h_conf_tree_path,format=1))

t = Tree(vec.h_conf_tree_path,format=1)
prepare_for_multitrain('multi_train',t,0,loader[2],0)

joblib.dump(t, path+"../models/1_nn_tree_"+'multi_train'+".joblib")
print('Prepared')
mod_name = 'lr_hand_conf'


multi_nodes = list([node for node in t.traverse("levelorder") if hasattr(node, 'opt_target')])

for node in t.traverse("levelorder"):
    if hasattr(node, 'opt_target'):
        for i in range(0,len(multi_nodes)):
            if multi_nodes[i]==node:
                node.add_feature('number',i)

joblib.dump(t, path+"../models/2_nn_tree_"+'multi_train'+".joblib")

multi_nodes = list([node for node in t.traverse("levelorder") if hasattr(node, 'opt_target')])

print('Collected nodes :'+str(len(multi_nodes)))

submodels_done = multiprocessing.Value('i', 0)
lock_train_data = multiprocessing.Value('i',0)
lock_val_data = multiprocessing.Value('i',0)
use one less process to be a little more stable
p = MyPool(processes = multiprocessing.cpu_count()-1)
#p = MyPool(processes = 9)

#timing it...
start = time.time()
print('Started Training with',multiprocessing.cpu_count()-1,'threads')
p.map(multi_node_train, multi_nodes)

#multi_node_train(multi_nodes[3])
p.close()
p.join()
print("Nodes training Complete")
end = time.time()
print('total time (s)= ' + str(end-start))
joblib.dump(t, path+"../models/2_nn_tree_"+'multi_train'+".joblib")

t = joblib.load(path+"../models/2_nn_tree_"+'multi_train'+".joblib",)

for node in t.traverse("levelorder"):
    if hasattr(node, 'opt_target'):
        node_2 = joblib.load(path+"../models/nodes/"+str(node.number)+".joblib",)
        node.add_feature('model',node_2.model)
        node.add_feature('level',node_2.level)
        node.add_feature('train_result',node_2.train_result)
        node.add_feature('samples_per_class',node_2.samples_per_class)
        node.add_feature('done_epochs',node_2.done_epochs)
        print(str(node.number),end='\r')

joblib.dump(t, path+"../models/3_nn_tree_"+'multi_train'+".joblib")
print('Done training HAND CONFIGURATION tree')

test('multi_train', loader[1])
t = joblib.load(path+"../models/3_nn_tree_"+'multi_train'+".joblib",)
#annotate("../models/3_nn_tree_"+'multi_train'+".joblib","greek_lf_test_input_keys_1d_np_lr_hand_conf_h5.hdf5",)
print_table_results(t)

del(loader,vec)


