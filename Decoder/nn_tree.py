import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
#from torchvision import datasets, transforms
import pathlib
import argparse
import numpy as np
import vector2sigml.v2s
from ete3 import Tree
import os
import glob
#from sklearn.utils import shuffle
#from sklearn.svm import SVC
import joblib
import h5py
import random, time
import multiprocessing
#import torch.multiprocessing as multiprocessing
#from torch.multiprocessing import Pool, set_start_method, freeze_support
def l_counter(leaf):
    #global tree_leaves
    #global leaves_counter
    leaves_counter[tree_leaves.index(leaf)] += 1
    return 

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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = self.initHidden()
        self.drop = nn.Dropout(0.2)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i1o = nn.Linear(input_size + hidden_size, input_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        
        output = self.drop(input)
        #combined = torch.cat((output, hidden), 1)
        #hidden = self.i2h(combined)
        #output = self.i1o(combined)
        #output = torch.sigmoid(output)
        #combined = torch.cat((output, hidden), 1)
        #hidden = self.i2h(combined)
        #output = self.i1o(combined)
        #output = torch.sigmoid(output)
        combined = torch.cat((output, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = torch.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(BATCH_SIZE, self.hidden_size)

    def test_initHidden(self):
        return torch.zeros(1, self.hidden_size)

def init_weights(m):
    if type(m) == nn.Linear:
        #torch.nn.init.constant_(m.weight, 0.0)
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0.0)

def train_one_node(model, device, data, mod_target, optimizer):
    model.train()
    global big_loss
    global base
    tr_c = torch.tensor([mod_target],dtype=torch.float)
    d, tr = data.to(device,non_blocking=True), tr_c.to(device,non_blocking=True)
    
    output = model(d)

    loss = nn.BCELoss(reduction='mean')
    loss_a = loss(output, tr)

    loss_a.backward()
    
    optimizer.step()
    optimizer.zero_grad()

    pred = output
    
    for p in range(0,len(pred[0])):
        if pred[0][p] == max(output[0]):
            pred[0][p] = 1.0
        else:
            pred[0][p] = 0.0
    #print(pred)
    big_loss += float(loss_a) #if not using float it will memory leak like craaaaazyyy 
    #big_loss +=len(tr[0])- pred[0].eq(tr[0].view_as(pred[0])).sum().item()
    base  += len(tr[0])
    del(output,pred,loss_a,tr_c,d)
    #scheduler.step(loss_a)

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
        #epoch_enc_loss = 0
        #epoch_enc_base = 0
        net = torch.nn.DataParallel(model)
        for batch_idx, (data, target) in enumerate(dataloader):
            #big_loss = 0
            base = 0
            #encoder_loss = 0
            #encoder_base = 0
            #d, tr = data.to(device,non_blocking=True), torch.tensor([(tar==1.0).nonzero() for tar in target],dtype=torch.long).to(device,non_blocking=True)
            #d, mask = mask_samples(d)
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

            #print(pred)
            #big_loss += float(loss_a.detach()) #if not using float it will memory leak like craaaaazyyy
            #encoder_loss += float(loss_en.detach())
            #big_loss +=len(tr[0])- pred[0].eq(tr[0].view_as(pred[0])).sum().item()
            base  += len(target[0])
            #encoder_base += len(d[0])
            
            #scheduler.step(loss_a)
            print_str = "Training epoch "+str(epoch) + ' ' + '%0.2f' % (100. * batch_idx/len(dataloader))+ '%'  
            #print(print_str, end='\r')
            
            del(print_str)
            epoch_loss += float(loss_a.detach())
            epoch_base += base
            del(output,pred,loss_a,d)
        total_loss = 100. * epoch_loss/epoch_base
        val_loss = validation_loss(model,device,testdataloader)
        scheduler.step(val_loss)


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
                #print(' **** '+ 'Train : ' +'%0.2f' % (100*big_loss/len(dataloader)) +' **** '+'Validation: ' + '%0.2f' % (100*val_loss/len(testdataloader))+' **** ')
                #print(' **** '+ 'Train : ' +'%0.2f' % (100*epoch_loss/epoch_base) +' **** '+'Validation: ' + '%0.2f' % (100*val_loss/len(testdataloader))+' **** '+'\n'+'TEST: Classes: '+str(out_len)+' Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully),end='\r')
                #if res[1] > 95: #res[1] == 100 or 
                if epoch == EPOCHS:
                #print(' **** '+ 'Train : ' +'%0.2f' % (100*epoch_loss/epoch_base) +' **** '+'Validation: ' + '%0.2f' % (100*val_loss/len(testdataloader))+' **** '+'\n'+'TEST: Classes: '+str(out_len)+' Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully))

                #print("Total epoch loss: "+str(total_loss))
                #print("Total epoch encoder loss: "+str(total_enc_loss))
                    print('TEST: Classes: '+str(out_len)+' Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully))
                #print("Done node training with: "+ str(epoch)+ " epochs")
                    return res
            else:
                #if epoch == EPOCHS:
                    #print("Total epoch loss: "+str(total_loss))
                    #print("Total epoch encoder loss: "+str(total_enc_loss))
                    #print('TEST: Classes: '+str(out_len)+' Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully))
                    #print("Done node training with: "+ str(epoch)+ " epochs")
                return (out_len,None,None,None,None,None)

def quick_test(tree, data):
    print('Target :'+str(data))
    test_vec  = get_test_np_vector(tree,data)
    print('Output :'+str(test_vec))
    return 

def get_test_np_vector(tt, data, transfer=False):
    tree = tt
    tree_leaves = list([ln for ln in tree.iter_leaves()])
    #print(len(tree_leaves))
    if transfer:
        test_walker_transfer(tree, data)
    else:
        test_walker(tree, data)
    r_vector =[0. for i in range(0,len(tree_leaves))]
    i = 0
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_vector[i] = 1.
        i = i + 1
    return r_vector

def backed_up_get_test_np_vector(tt, data):
    tree = tt
    tree_leaves = list([ln for ln in tree.iter_leaves()])
    backed_up_test_walker(tree, data, np.array([]))
    r_vector =[0 for i in range(0,len(tree_leaves))]
    i = 0
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_vector[i] = 1
        i = i + 1
    return r_vector

def get_test_voting_np_vector(tt, data):
    tree = tt
    s_tree = tt
    
    test_voting_walker_one(tree, data,0,0)
    test_voting_walker_two(tree,data)
    tree_leaves = list([ln for ln in tree.iter_leaves()])
    #for l in range(0,len(tree_leaves)):
    #    print("Sc: "+("%.4f" % tree_leaves[l].c_score)+" Mc: "+str(tree_leaves[l].m_count))
    r_vector =[0 for i in range(0,len(tree_leaves))]
    i = 0
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_vector[i] = 1
        i = i + 1
    #r_vector =[(tree_leaves[i].c_score/tree_leaves[i].m_count) for i in range(0,len(tree_leaves))]

    test_walker(s_tree, data)
    tree_leaves = list([ln for ln in s_tree.iter_leaves()])
    s_r_vector =[0 for i in range(0,len(tree_leaves))]
    i = 0
    for leaf in s_tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            s_r_vector[i] = 1
        i = i + 1

    return (r_vector,s_r_vector)

def train_walker(tree,data,target, start_ind,input_len):
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
                    #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                    temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                    if any(x == 1.0 for x in temp_ind):
                        mod_target.append(1.0)
                        #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                    else:
                        mod_target.append(0.0)
                    del(temp_ind)
                #print(mod_target)

                assert len(mod_target) == len(tree.children), 'TRAIN: Target lenght is not equal to children lenght' 

                opt_target = []
                for o_i in range(0,len(tree.children)):
                    if tree.children[o_i].name.find('OPT') != -1 and tree.children[o_i].name != "NON_OPT":
                        opt_target.append(mod_target[o_i])
                        if mod_target[o_i] == 1:
                            train_choice = tree.children[o_i]

                assert len(opt_target) == len(options), 'TRAIN: Options Target lenght is not equal to options lenght'

                if hasattr(tree, 'model'):
                    model_t = tree.model.to(DEVICE)
                    optimizer_t = tree.optimizer
                    #for state in optimizer_t.state.values():
                    #    for k, v in state.items():
                    #        if isinstance(v, torch.Tensor):
                    #            state[k] = v.to(DEVICE)
                else: 
                    #print('Input :'+str(input_len)+" opt Target: "+str(len(opt_target)))
                    model_t = Sequential(input_len,len(opt_target)).to(DEVICE)
                    model_t.apply(init_weights)
                    tree.add_feature('model',model_t)
                    optimizer_t = optim.Adam(model_t.parameters(),lr= LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
                    tree.add_feature('optimizer',optimizer_t)

                train_one_node(model_t, DEVICE, data, opt_target, optimizer_t)
                tree.model = model_t
                tree.optimizer = optimizer_t 
                del(model_t)
                del(optimizer_t)
                del(opt_target,mod_target)
            if len(options)==1:
                train_choice = options[0]

            del(options)
            for o_i in range(0,len(tree.children)):
                if tree.children[o_i] == train_choice or tree.children[o_i].name == "NON_OPT":
                    #next_step.append((tree.children[o_i], indicies[o_i]))
                    train_walker(tree.children[o_i],data,target,indicies[o_i],input_len)
            del(indicies)
            
        else:

            for o_i in range(0,len(tree.children)):
                #next_step.append((tree.children[o_i], indicies[o_i]))
                train_walker(tree.children[o_i],data,target,indicies[o_i],input_len)
            del(indicies)
        #for i in range(0, len(next_step)):
        #    train_walker(next_step[i][0],data,target,next_step[i][1],input_len)

def level_train_walker(name_tree,mod_name, tree, start_ind, input_len, level):
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
        global submodels_done
        global submodels_total
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
                #if level==MAX_LEVEL-1:
                if hasattr(tree, 'model')==True:
                    if MAX_LEVEL <= tree.level:
                        model_t = tree.model.to(DEVICE)
                        optimizer_t = tree.optimizer
                        #optimizer_t = optim.SGD(model_t.parameters(),lr= LR, momentum= 0.9, nesterov=True)
                        try: 
                            valid_dataset = H5Dataset_level_validation(path+'test_input_keys_1d_np_'+mod_name+"_h5.hdf5", path+'rs_test_input_keys_1d_np_'+mod_name+"_h5.hdf5",opt_target)
                            test_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=1, shuffle=True, num_workers = 1)
                        except:
                            test_dataloader = None
                            print('No Test Samples found')
                        try:
                            node_dataset = H5Dataset_level(path+'train_input_keys_1d_np_'+mod_name+"_h5.hdf5",opt_target)
                            train_dataloader = torch.utils.data.DataLoader(node_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
                        except:
                            train_dataloader = None
                            print('No Train Samples found')

                        #tree.add_feature('data_set',node_dataset)
                        if train_dataloader != None:
                            time_start = datetime.datetime.now()
                            train_result = train_one_node_full(model_t, DEVICE, train_dataloader, optimizer_t, test_dataloader)
                            time_ = datetime.datetime.now() - time_start
                            tree.model= model_t
                            tree.train_result= train_result
                            tree.level=level
                            tree.classes=len(opt_target)
                            tree.samples_per_class=int(node_dataset.__len__()/len(opt_target))
                            tree.optimizer=optimizer_t
                            del(model_t,node_dataset,train_dataloader,optimizer_t)
                            tree.done_training=True
                            joblib.dump(tree.get_tree_root(), path+"../models/nn_tree_"+name_tree+".joblib")

                            submodels_done += 1
                            print(str(int(submodels_done/submodels_total*100))+'% trained '+str(submodels_done)+'/'+str(submodels_total)+' Hours left: '+str((time_*(submodels_total-submodels_done))))
                            del(time_,time_start)
                else:
                    if hasattr(tree, 'done_training')==False:
                        #n_hidden = 520
                        #model_t = RNN(input_len,n_hidden,len(opt_target)).to(DEVICE)
                        #print('Level : '+str(level))
                        model_t = Sequential(input_len,len(opt_target)).to(DEVICE)
                        model_t.apply(init_weights)
                        optimizer_t = optim.Adam(model_t.parameters(),lr= LR)
                        #optimizer_t = optim.SGD(model_t.parameters(),lr= LR, momentum= 0.9, nesterov=True)
                        try: 
                            valid_dataset = H5Dataset_level_validation(path+'test_input_keys_1d_np_'+mod_name+"_h5.hdf5", path+'rs_test_input_keys_1d_np_'+mod_name+"_h5.hdf5",opt_target)
                            test_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=1, shuffle=True, num_workers = 1)
                        except:
                            test_dataloader = None
                            print('No Test Samples found')
                        try:
                            node_dataset = H5Dataset_level(path+'train_input_keys_1d_np_'+mod_name+"_h5.hdf5",opt_target)
                            train_dataloader = torch.utils.data.DataLoader(node_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
                        except:
                            train_dataloader = None
                            #print('No Train Samples found')

                        #tree.add_feature('data_set',node_dataset)
                        if train_dataloader != None:
                            time_start = datetime.datetime.now()
                            train_result = train_one_node_full(model_t, DEVICE, train_dataloader, optimizer_t, test_dataloader)
                            time_ = datetime.datetime.now() - time_start
                            tree.add_feature('model',model_t)
                            tree.add_feature('train_result',train_result)
                            tree.add_feature('level',level)
                            tree.add_feature('classes',len(opt_target))
                            tree.add_feature('samples_per_class',int(node_dataset.__len__()/len(opt_target)))
                            tree.add_feature('optimizer',optimizer_t)
                            del(model_t,node_dataset,train_dataloader,optimizer_t)
                            tree.add_feature('done_training',True)
                            joblib.dump(tree.get_tree_root(), path+"../models/nn_tree_"+mod_name+".joblib")

                            submodels_done += 1
                            print(str(int(submodels_done/submodels_total*100))+'% trained '+str(submodels_done)+'/'+str(submodels_total)+' Hours left: '+str((time_*(submodels_total-submodels_done))))
                            del(time_,time_start)
                    else:
                        if tree.done_training == True:

                            submodels_done += 1
                    level = level + 1
                    del(opt_target,mod_target)

            del(options)
            for o_i in range(0,len(tree.children)):
                    level_train_walker(name_tree,mod_name,tree.children[o_i],indicies[o_i],input_len,(level))
            del(indicies)
            
            
        else:

            for o_i in range(0,len(tree.children)):
                level_train_walker(name_tree,mod_name,tree.children[o_i],indicies[o_i],input_len,level)
            del(indicies)

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

def level_target_return_walker(tree, target, start_ind,level):
    if level == MAX_LEVEL:
        return []
    if tree.is_leaf():
        #l_counter(tree)
        #tree.name = tree.name +'*'
        return []
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
            opt_target = []
            for i in range(0, (len(tree.children))):
                if tree.children[i].name.find('OPT') != -1 and tree.children[i].name != "NON_OPT":
                        options.append(tree.children[i])

            if len(options)>0:
                mod_target = []

                for ind in range(0,len(indicies)-1):
                    #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                    temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                    if any(x == 1.0 for x in temp_ind):
                        mod_target.append(1.0)
                        #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                    else:
                        mod_target.append(0.0)
                    del(temp_ind)
                #print(mod_target)

                assert len(mod_target) == len(tree.children), 'TRAIN: Target lenght is not equal to children lenght' 

                
                for o_i in range(0,len(tree.children)):
                    if tree.children[o_i].name.find('OPT') != -1 and tree.children[o_i].name != "NON_OPT":
                        if level == MAX_LEVEL-1:
                            opt_target.append(mod_target[o_i])
                        if mod_target[o_i] == 1:
                            train_choice = tree.children[o_i]
                if level == MAX_LEVEL-1:
                    assert len(opt_target) == len(options), 'TRAIN: Options Target lenght is not equal to options lenght'

                del(mod_target)
                level = level + 1
            
            ret_list = []

            for o_i in range(0,len(tree.children)):
                if tree.children[o_i] == train_choice or tree.children[o_i].name == "NON_OPT":
                    #next_step.append((tree.children[o_i], indicies[o_i]))
                    for c in level_target_return_walker(tree.children[o_i],target,indicies[o_i],(level)):
                        ret_list.append(c)

            if len(opt_target)>0:
                ret_list = opt_target

            
            del(indicies)
            del(options)
            return ret_list
        else:
            ret_list = []
            for o_i in range(0,len(tree.children)):
                #next_step.append((tree.children[o_i], indicies[o_i]))
                for c in level_target_return_walker(tree.children[o_i],target,indicies[o_i],level):
                    ret_list.append(c)
            return ret_list
            del(indicies)
        #for i in range(0, len(next_step)):
        #    train_walker(next_step[i][0],data,target,next_step[i][1],input_len)

def backed_up_train_walker(tree,data,target, start_ind,input_len, extra_input):
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
            opt_target = []
            for i in range(0, (len(tree.children))):
                if tree.children[i].name.find('OPT') != -1 and tree.children[i].name != "NON_OPT":
                        options.append(tree.children[i])

            if len(options)>0:
                mod_target = []

                for ind in range(0,len(indicies)-1):
                    #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                    temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                    if any(x == 1.0 for x in temp_ind):
                        mod_target.append(1.0)
                        #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                    else:
                        mod_target.append(0.0)
                    del(temp_ind)
                #print(mod_target)

                assert len(mod_target) == len(tree.children), 'TRAIN: Target lenght is not equal to children lenght' 

                
                for o_i in range(0,len(tree.children)):
                    if tree.children[o_i].name.find('OPT') != -1 and tree.children[o_i].name != "NON_OPT":
                        opt_target.append(mod_target[o_i])
                        if mod_target[o_i] == 1:
                            train_choice = tree.children[o_i]

                assert len(opt_target) == len(options), 'TRAIN: Options Target lenght is not equal to options lenght'

                if hasattr(tree, 'model'):
                    model_t = tree.model.to(DEVICE)
                    optimizer_t = tree.optimizer
                else: 
                    model_t = Sequential((input_len+len(extra_input)),len(opt_target)).to(DEVICE)
                    model_t.apply(init_weights)
                    tree.add_feature('model',model_t)
                    optimizer_t = optim.Adam(model_t.parameters(),lr= LR)
                    tree.add_feature('optimizer',optimizer_t)

                train_one_node(model_t, DEVICE, torch.cat((data,torch.tensor([extra_input],dtype=torch.float)),1), opt_target, optimizer_t)
                tree.model = model_t
                tree.optimizer = optimizer_t 
                del(model_t)
                del(optimizer_t)
                del(mod_target)

            del(options)
            for o_i in range(0,len(tree.children)):
                if tree.children[o_i] == train_choice or tree.children[o_i].name == "NON_OPT":
                    #next_step.append((tree.children[o_i], indicies[o_i]))
                    backed_up_train_walker(tree.children[o_i],data,target,indicies[o_i],input_len, np.concatenate((extra_input,opt_target),axis=0))
            del(indicies)
            del(opt_target)
        else:

            for o_i in range(0,len(tree.children)):
                #next_step.append((tree.children[o_i], indicies[o_i]))
                backed_up_train_walker(tree.children[o_i],data,target,indicies[o_i],input_len, extra_input)
            del(indicies)
        #for i in range(0, len(next_step)):
        #    train_walker(next_step[i][0],data,target,next_step[i][1],input_len)

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
                #output = output[0].tolist()
                #print(output)
                assert len(options) == len(output), 'TEST: Options length'+str((len(options)))+' is not equal to the model output lenght '+str((len(output)))
                
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

def test_walker_transfer(tree, data):

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
                #t_model = tree.model.to(DEVICE)
                t_model = joblib.load(path+"../models/transfer_models/"+tree.name+".joblib")
                t_model.eval()
                output = t_model(data.to(DEVICE))
                output = output[0].tolist()
                #print(output)
                assert len(options) == len(output), 'TEST: Options length is not equal to the model output lenght'
                
                for o_i in range(0,len(options)):
                    if output[o_i] == max(output):
                        trained_choice = options[o_i]
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

def level_test_walker(tree, data, target, start_ind, level):
    if level == MAX_LEVEL:
        return []
    if tree.is_leaf():
        #l_counter(tree)
        #tree.name = tree.name +'*'
        return []
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
            output = []
            train_choice = Tree()
            for i in range(0, (len(tree.children))):
                if tree.children[i].name.find('OPT') != -1 and tree.children[i].name != "NON_OPT":
                        options.append(tree.children[i])

            if len(options)>0:
                if level==MAX_LEVEL-1:
                    t_model = tree.model.to(DEVICE)
                    t_model.eval()
                    output = t_model(data.to(DEVICE))
                    output = output[0].tolist()
                    #print(output)
                    assert len(options) == len(output), 'TEST: Options length is not equal to the model output lenght'
                    
                    #for o_i in range(0,len(options)):
                    #    if output[o_i] == max(output):
                    #        trained_choice = options[o_i]
                            #print(str(output)+" "+str(output[o_i]))
                    
                    del(t_model)
                else:
                    mod_target = []

                    for ind in range(0,len(indicies)-1):
                        #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                        temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                        if any(x == 1.0 for x in temp_ind):
                            mod_target.append(1.0)
                            #print(target[0][indicies[ind]:indicies[ind+1]].tolist())
                        else:
                            mod_target.append(0.0)
                        del(temp_ind)
                    #print(mod_target)

                    assert len(mod_target) == len(tree.children), 'TRAIN: Target lenght is not equal to children lenght' 

                    opt_target = []
                    for o_i in range(0,len(tree.children)):
                        if tree.children[o_i].name.find('OPT') != -1 and tree.children[o_i].name != "NON_OPT":
                            opt_target.append(mod_target[o_i])
                            if mod_target[o_i] == 1:
                                train_choice = tree.children[o_i]

                    assert len(opt_target) == len(options), 'TRAIN: Options Target lenght is not equal to options lenght'

                    del(mod_target)
                level = level + 1
            
            ret_list = []

            for o_i in range(0,len(tree.children)):
                if tree.children[o_i] == train_choice or tree.children[o_i].name == "NON_OPT":
                    #next_step.append((tree.children[o_i], indicies[o_i]))
                    for c in level_test_walker(tree.children[o_i],data,target,indicies[o_i],level):
                        ret_list.append(c)

            if len(output)>0:
                for o in  output:
                    if o == max(output):
                        ret_list.append(1.0)
                    else:
                        ret_list.append(0.0)
                        #ret_list = output
                        
                    #else:
                    #    ret_list = opt_target

            
            del(indicies)
            del(options)
            return ret_list
        else:
            ret_list = []
            for o_i in range(0,len(tree.children)):
                #next_step.append((tree.children[o_i], indicies[o_i]))
                for c in level_test_walker(tree.children[o_i],data, target,indicies[o_i],level):
                    ret_list.append(c)
            return ret_list
            del(indicies)
        #for i in range(0, len(next_step)):
        #    train_walker(next_step[i][0],data,target,next_step[i][1],input_len)

def backed_up_test_walker(tree, data, extra_input):
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
            output_res = []
            trained_choice = Tree()
            for o in tree.children:
                if o.name.find('OPT') != -1 and o.name != "NON_OPT":
                        options.append(o)
            if len(options)>0:
                t_model = tree.model.to(DEVICE)
                t_model.eval()
                #print(str(next(t_model.parameters()).size())+" "+str(torch.cat((data,torch.tensor([extra_input],dtype=torch.float)),1).size()))
                output = t_model(torch.cat((data,torch.tensor([extra_input],dtype=torch.float)),1)).to(DEVICE)
                output = output[0].tolist()
                #print(output)
                assert len(options) == len(output), 'TEST: Options length is not equal to the model output lenght'
                
                for o_i in range(0,len(options)):
                    output_res.append(output[o_i])
                    if output[o_i] == max(output):
                        trained_choice = options[o_i]
                        #print(str(output)+" "+str(output[o_i]))
                
                del(t_model)
                del(output)
            del(options)
            del(weights)
            for child in tree.children:
                if child == trained_choice or child.name == "NON_OPT":
                    backed_up_test_walker(child,data, np.concatenate((extra_input,output_res),axis=0))
        else:
            for child in tree.children:
                
                backed_up_test_walker(child, data, extra_input)

def test_voting_walker_one(tree, data, c_score, m_count):
    if tree.is_leaf():
        #l_counter(tree)
        #tree.name = tree.name +'*'
        tree.add_feature('c_score',c_score)
        tree.add_feature('m_count',m_count)
        return
    else:

        choicer = False

        for o in tree.children:
            if o.name.find('OPT') != -1:
                choicer = True

        if choicer:
            options = []
            trained_choice = Tree()
            for o in tree.children:
                if o.name.find('OPT') != -1 and o.name != "NON_OPT":
                        options.append(o)
            if len(options)>0:
                t_model = tree.model.to(DEVICE)
                t_model.eval()
                output = t_model(data.to(DEVICE))
                
                output = output[0].tolist()
                #print(output)
                assert len(options) == len(output), 'TEST: Options length is not equal to the model output lenght'

                del(t_model)
            #del(options)
            #s = 0
            for child in tree.children:
                #if child == trained_choice or child.name == "NON_OPT":
                #if o.name.find('OPT') != -1 and o.name != "NON_OPT":

                if o.name == "NON_OPT":
                    test_voting_walker_one(child, data, c_score, (m_count+1))
                else:
                    for opt in range(0,len(options)):
                        test_voting_walker_one(options[opt], data, output[opt], (m_count+1))
                    #s += 1
        else:
            for child in tree.children:
                test_voting_walker_one(child, data, c_score, m_count)

def test_voting_walker_two(tree, data):

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
            trained_choice = Tree()
            for o in tree.children:
                if o.name.find('OPT') != -1 and o.name != "NON_OPT":
                        max_val = max([(l.c_score/l.m_count) for l in o.iter_leaves()])
                        options.append((o,max_val))
                        del(max_val)
            if len(options)>0:
                #t_model = tree.model.to(DEVICE)
                #t_model.eval()
                #output = t_model(data.to(DEVICE))
                #output = output[0].tolist()
                #print(output)
                #assert len(options) == len(output), 'TEST: Options length is not equal to the model output lenght'
                vals = list([v[1] for v in options])
                for o_i in range(0,len(options)):
                    if options[o_i][1] == max(vals):
                        trained_choice = options[o_i][0]
                        #print(str(output)+" "+str(output[o_i]))
                
                #del(t_model)
            del(options)
            for child in tree.children:
                if child == trained_choice or child.name == "NON_OPT":
                    test_voting_walker_two(child,data)
        else:
            for child in tree.children:
                
                test_voting_walker_two(child, data)

def train(tree_path, name_addition, train_loader, test_loader, input_len):
    global big_loss
    global base

    if os.path.exists(path+"../models/nn_tree_"+name_addition+".joblib"): 
        print('Training tree at: '+ path + "../models/nn_tree_"+name_addition+".joblib")
    else:
        print('Training new tree')
    print('Start training .. ')
    time_start = datetime.datetime.now()
    total_loss = 100
    losses_to_save = []
    for epoch in range(1, EPOCHS + 1):
        if os.path.exists(path+"../models/nn_tree_"+name_addition+".joblib"): 
            tree = joblib.load(path+"../models/nn_tree_"+name_addition+".joblib")
        else:
            tree = Tree(tree_path,format=1)
        epoch_loss = 0
        epoch_base = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            #if(batch_idx)==0: 
            #    print("Data vector size:"+str(len(target[0])))
            #    print("Target vector size:"+str(len(data[0])))
            big_loss = 0
            base = 0
            train_walker(tree,data,target,0,input_len)
            print_str = "Training epoch "+str(epoch) + ' ' + '%0.2f' % (100. * batch_idx/len(train_loader))+ '%'  
            print(print_str, end='\r')
            del(print_str)
            epoch_loss += big_loss
            epoch_base += base

            #if (100. * batch_idx/len(train_loader)) % 25 == 0:
            #    h=hpy()
            #    print(h.heap(), end='\r')

        #if batch_idx == len(train_loader)-1:
        #     quick_test(tree,data)
        
        #print('Time epoch cost',time_end - time_start)
        total_loss = 100. * epoch_loss/epoch_base
        losses_to_save.append(total_loss)
        print("TRAIN Epoch loss: "+ '%0.2f' % total_loss+ '%' +'                        ')
        joblib.dump(tree, path+"../models/nn_tree_"+name_addition+".joblib")
        joblib.dump(losses_to_save, path+"../models/nn_tree_"+name_addition+"_LOSSES.joblib")
        del(tree)
        test(name_addition, test_loader)
        if total_loss < 1.5:
            break

    time_end = datetime.datetime.now()
    print('Time cost',time_end - time_start)
    return

def level_train(tree_path, name_tree, mod_name, input_len, out_len):
    global big_loss
    global base

    if os.path.exists(path+"../models/nn_tree_"+name_tree+".joblib"): 
        tree = joblib.load(path+"../models/nn_tree_"+name_tree+".joblib")
        print('Training tree at: '+ path + "../models/nn_tree_"+name_tree+".joblib")
    else:
        tree = Tree(tree_path,format=1)
        print('Training new tree')

    time_start = datetime.datetime.now()
    print(time_start)
    total_loss = 100
    losses_to_save = []

    level_train_walker(name_tree,mod_name,tree,0,input_len,0)

    del(tree)
    time_end = datetime.datetime.now()
    print(time_end)
    print('Time cost',time_end - time_start)
    return

def backed_up_train(tree_path,name_addition, train_loader, test_loader, input_len):
    global big_loss
    global base
    #global tree_leaves
    #global leaves_counter
    if os.path.exists(path+"../models/nn_tree_"+name_addition+".joblib"): 
        tree = joblib.load(path+"../models/nn_tree_"+name_addition+".joblib")
        print('Training tree at: '+ path + "../models/nn_tree_"+name_addition+".joblib")
    else:
        tree = Tree(tree_path,format=1)
        print('Training new tree')
    print('Start training .. ')
    time_start = datetime.datetime.now()
    total_loss = 100
    losses_to_save = []
    for epoch in range(1, EPOCHS + 1):
        #if os.path.exists(path+"../models/nn_tree_"+name_addition+".joblib"): 
        #    tree = joblib.load(path+"../models/nn_tree_"+name_addition+".joblib")
        #else:
        #    tree = Tree(tree_path,format=1)
        #tree_leaves = list([l for l in tree.iter_leaves()])
        #leaves_counter = list([0 for l in range(0,len(tree_leaves))])
        epoch_loss = 0
        epoch_base = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            #if(batch_idx)==1: 
            #    print(target[0])
            big_loss = 0
            base = 0
            backed_up_train_walker(tree,data,target,0,input_len, np.array([]),0)
            print_str = "Training epoch "+str(epoch) + ' ' + '%0.2f' % (100. * batch_idx/len(train_loader))+ '%'  
            print(print_str, end='\r')
            del(print_str)
            epoch_loss += big_loss
            epoch_base += base

            #if (100. * batch_idx/len(train_loader)) % 25 == 0:
            #    h=hpy()
            #    print(h.heap(), end='\r')

        #if batch_idx == len(train_loader)-1:
        #     quick_test(tree,data)
        
        #print('Time epoch cost',time_end - time_start)
        total_loss = 100. * epoch_loss/epoch_base
        losses_to_save.append(total_loss)
        print("TRAIN Epoch loss: "+ '%0.2f' % total_loss+ '%' +'                        ')

        #
        if epoch%25==0:
            joblib.dump(tree, path+"../models/nn_tree_"+name_addition+".joblib")
            joblib.dump(losses_to_save, path+"../models/nn_tree_"+name_addition+"_LOSSES.joblib")
            backed_up_test(name_addition, test_loader,0)
        if total_loss < 1.5:
            break
    del(tree)
    time_end = datetime.datetime.now()
    print('Time cost',time_end - time_start)
    return

def test(name_addition, test_loader, transfer=False):
    print('Testing .. ',end='\r')
    if transfer:
        tree = Tree(name_addition,format=1)
    else:
        #tree = joblib.load(path+"../models/nn_tree_"+name_addition+".joblib")
        tree = joblib.load(path+"../models/3_nn_tree_"+name_addition+".joblib")
    scores = []
    for batch_idx, (test_data, target) in enumerate(test_loader):
        test_vec  = np.array(get_test_np_vector(tree,test_data,transfer))
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
    print('TEST: Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully) )
    del(tree,scores)
    return

def annotate(tree_path, data_path):
    print('Making annotations..')
    tree = joblib.load(path+tree_path)
    data_file = h5py.File(path+data_path, 'r')
    for vec in range(len(data_file['X_data'])):
        #print(torch.from_numpy(data_file['X_data'][vec]).float())
        test_vec  = np.array(get_test_np_vector(tree,torch.from_numpy(data_file['X_data'][vec]).float(),False))
        #print(test_vec)
        test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        test_v.save_sigml('./output_greek/pred_saved_all_nn_tree_'+str(vec)+'.txt',str(vec))
    return

def level_test(name_addition, test_loader):
    print('Testing .. ',end='\r')
    tree = joblib.load(path+"../models/nn_tree_"+name_addition+".joblib")
    scores = []
    out_len = 0
    for batch_idx, (test_data, target) in enumerate(test_loader):
        test_vec  = np.array(level_test_walker(tree,test_data,target,0,0))
        tar_vec = np.array(level_target_return_walker(tree,target,0,0))
        
        assert len(test_vec) == len(tar_vec), 'Something wrong '+ str(len(test_vec))+' '+str(len(tar_vec))
        #print(len(tar_vec))
        if batch_idx == 0:
            out_len = len(tar_vec)
            #print(len(tar_vec))
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
    mean_accuracy = 100.* sum(s[0] for s in scores)/len(scores)
    mean_recall = sum(s[1] for s in scores)/len(scores)
    mean_precision = sum(s[2] for s in scores)/len(scores)
    mean_f1_score = sum(s[3] for s in scores)/len(scores)
    mean_fully = 100.* sum(s[4] for s in scores)/len(scores)
    print('TEST: Length: '+str(out_len)+' Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully) )
    del(tree,scores)
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

def backed_up_test(name_addition, test_loader):
    print('Testing .. ',end='\r')
    tree = joblib.load(path+"../models/nn_tree_"+name_addition+".joblib")
    scores = []
    for batch_idx, (test_data, target) in enumerate(test_loader):
        test_vec  = np.array(backed_up_get_test_np_vector(tree,test_data))
        
        if len(test_vec) < len(target[0]):
            target = target[0].cpu().numpy()[:len(test_vec)]
        else:
            target = target[0].cpu().numpy()
        scores.append(calculate_scores(test_vec,target))
        #print(test_vec)
        #print(target)
        #test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        #test_v.save_sigml(path+'../pred_saved_all_nn_tree_'+str(batch_idx)+'.txt','predicted')

        #test_vec_tar = vector2sigml.v2s.Vec2sigml(target[0].cpu().numpy())
        #test_vec_tar.save_sigml(path+'../target_saved_all_nn_tree_'+str(batch_idx)+'.txt','target')
        #print("Saved "+str(batch_idx), end='\r')
    mean_accuracy = 100.* sum(s[0] for s in scores)/len(scores)
    mean_recall = sum(s[1] for s in scores)/len(scores)
    mean_precision = sum(s[2] for s in scores)/len(scores)
    mean_f1_score = sum(s[3] for s in scores)/len(scores)
    mean_fully = 100.* sum(s[4] for s in scores)/len(scores)
    print('TEST: Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully) )
    del(tree,scores)
    return

def test_voting(name_addition, test_loader):
    print('Testing .. ',end='\r')
    tree = joblib.load(path+"../models/nn_tree_"+name_addition+".joblib")
    scores = []
    for batch_idx, (test_data, target) in enumerate(test_loader):
        if batch_idx <= 9:
            vote_test, hier_test = get_test_voting_np_vector(tree,test_data)
            test_vec  = np.array(vote_test)
            test_h_vec = np.array(hier_test)
            if len(test_vec) < len(target[0]):
                target = target[0].cpu().numpy()[:len(test_vec)]
            else:
                target = target[0].cpu().numpy()
            
            
            #np.transpose(np_ar)
            print(np.concatenate((np.array([test_h_vec]),np.array([test_vec]),np.array([target])),axis=0).T)
            scores.append(calculate_scores(test_vec,target))
        
        
        #print(test_vec)
        #print(target)
        #test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        #test_v.save_sigml(path+'../pred_saved_all_nn_tree_'+str(batch_idx)+'.txt','predicted')

        #test_vec_tar = vector2sigml.v2s.Vec2sigml(target[0].cpu().numpy())
        #test_vec_tar.save_sigml(path+'../target_saved_all_nn_tree_'+str(batch_idx)+'.txt','target')
        #print("Saved "+str(batch_idx), end='\r')
    mean_accuracy = 100.* sum(s[0] for s in scores)/len(scores)
    mean_recall = sum(s[1] for s in scores)/len(scores)
    mean_precision = sum(s[2] for s in scores)/len(scores)
    mean_f1_score = sum(s[3] for s in scores)/len(scores)
    mean_fully = 100.* sum(s[4] for s in scores)/len(scores)
    print('TEST: Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully) )
    del(tree,scores)
    return

def test_all(name_addition_conf,name_addition_or,name_addition_loc, test_loader):
    print('Testing .. ',end='\r')
    tree_conf = joblib.load(path+"../models/nn_tree_"+name_addition_conf+".joblib")
    tree_or = joblib.load(path+"../models/nn_tree_"+name_addition_or+".joblib")
    tree_loc = joblib.load(path+"../models/nn_tree_"+name_addition_loc+".joblib")
    scores = []

    for batch_idx, (test_data, target) in enumerate(test_loader):
        test_vec_conf  = np.array(get_test_np_vector(tree_conf,test_data))
        test_vec_or  = np.array(get_test_np_vector(tree_or,test_data))
        test_vec_loc  = np.array(get_test_np_vector(tree_loc,test_data))
        test_vec = np.concatenate((test_vec_conf,test_vec_or,test_vec_loc),axis=0)
        del(test_vec_conf,test_vec_or,test_vec_loc)

        if len(test_vec) < len(target[0]):
            target = target[0].cpu().numpy()[:len(test_vec)]
        else:
            target = target[0].cpu().numpy()
        scores.append(calculate_scores(test_vec,target))
        #print(test_vec)
        #print(target)
        #test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        #test_v.save_sigml(path+'../pred_saved_all_nn_tree_'+str(batch_idx)+'.txt','predicted')

        #test_vec_tar = vector2sigml.v2s.Vec2sigml(target[0].cpu().numpy())
        #test_vec_tar.save_sigml(path+'../target_saved_all_nn_tree_'+str(batch_idx)+'.txt','target')
        #print("Saved "+str(batch_idx), end='\r')
    mean_accuracy = 100.* sum(s[0] for s in scores)/len(scores)
    mean_recall = sum(s[1] for s in scores)/len(scores)
    mean_precision = sum(s[2] for s in scores)/len(scores)
    mean_f1_score = sum(s[3] for s in scores)/len(scores)
    mean_fully = 100.* sum(s[4] for s in scores)/len(scores)
    print('TEST: Av. Accuracy: '+str('%0.2f' % mean_accuracy)+'% Av. Recall: '+str('%0.2f' % mean_recall)+' Av. Precicion: '+str('%0.2f' % mean_precision)+' Av. F1_Score: '+str('%0.2f' % mean_f1_score)+' Fully_Recognized: '+str('%0.2f' % mean_fully) )
    del(tree_conf,tree_or,tree_loc,scores)
    return

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

class SignDataset_depr(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X_data, Y_data):
        'Initialization'
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X_data)
    
    def __getitem__(self, index):

        # Load data and get label
        X = self.X_data[index]
        y = self.Y_data[index]

        return X, y

    def add_data(self,X_data_unit, Y_data_unit):
        self.X_data.append(X_data_unit)
        self.Y_data.append(Y_data_unit)
        return
        
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
        return o_d, o_t, pool

    def __getitem__(self, index): 
        #x_noized = data_noizer_waky(self.opt_data[index])
        #return (torch.from_numpy(x_noized).float(),torch.from_numpy(self.opt_target[index]).float())
        return (torch.from_numpy(self.opt_data[index]).float(),torch.from_numpy(self.opt_target[index]).float())

    def __len__(self):
        return len(self.opt_data)

class H5Dataset_level_transfer(data.Dataset):

    def __init__(self, file_path, opt_ts):
        super(H5Dataset_level_transfer, self).__init__()
        #print(file_path)
        h5_file = h5py.File(file_path, 'r')
        self.data = h5_file['X_data']
        self.target = h5_file['Y_data']
        self.input_len = self.data[0].shape[0]
        self.tree_len = self.target[0].shape[0]
        self.opt_data, self.opt_target = self.find_node_data(opt_ts)



    def find_node_data(self, opt_ts):

        file = 'temp_h5.hdf5'
        if os.path.exists(file):
            os.remove(file)
        first = True
        pool = [0 for c in range(0,len(opt_ts[0]))]
        for opt_t in opt_ts:

            for i in range(0, self.data.shape[0]):
                mod_target = []
                append = False
                
                for ind in range(0,len(opt_t)):
                    #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                    temp_ind = self.target[i][opt_t[ind][0]:opt_t[ind][1]].tolist()
                    if any(x == 1.0 for x in temp_ind):
                        mod_target.append(1.0)
                        #print(pool)
                        if min(pool) == pool[ind]:
                            pool[ind] += 1
                            append = True
                    else:
                        mod_target.append(0.0)
                    del(temp_ind)
                if append:
                    if first:
                        with h5py.File(file, "w") as temp_h5_file:
                            temp_h5_file.create_dataset("X_data", data=[self.data[i]], maxshape=(None,self.input_len))
                            temp_h5_file.create_dataset("Y_data", data=[np.array(mod_target)], maxshape=(None,len(opt_ts[0])))
                        first = False
                    else:
                        with h5py.File(file, "a") as temp_h5_file:
                            temp_h5_file["X_data"].resize((temp_h5_file["X_data"].shape[0] + 1), axis = 0)
                            temp_h5_file["X_data"][-1] = self.data[i]
                            temp_h5_file["Y_data"].resize((temp_h5_file["Y_data"].shape[0] + 1), axis = 0)
                            temp_h5_file["Y_data"][-1] = np.array(mod_target)

        print(pool)
        temp_h5_file = h5py.File(file, 'r')
        return temp_h5_file['X_data'], temp_h5_file['Y_data']

    def __getitem__(self, index): 
        x_noized = data_noizer_waky(self.opt_data[index])
        return (torch.from_numpy(x_noized).float(),torch.from_numpy(self.opt_target[index]).float())
        #return (torch.from_numpy(self.opt_data[index]).float(),torch.from_numpy(self.opt_target[index]).float())

    def __len__(self):
        return len(self.opt_data)

class H5Dataset_level_validation_transfer(data.Dataset):

    def __init__(self, file_path,file_path_2, opt_ts):
        super(H5Dataset_level_validation_transfer, self).__init__()
        #print('Valid'+file_path)
        h5_file = h5py.File(file_path, 'r')
        h5_file_2 = h5py.File(file_path_2, 'r')
        self.data = h5_file['X_data'] 
        self.data_2 = h5_file_2['X_data']
        self.target = h5_file['Y_data']
        self.target_2 = h5_file_2['Y_data']
        self.input_len = self.data[0].shape[0]
        self.tree_len = self.target[0].shape[0]
        self.opt_data, self.opt_target = self.find_node_data(opt_ts)



    def find_node_data(self, opt_ts):

        file = 'temp_h5_validation.hdf5'
        if os.path.exists(file):
            os.remove(file)
        first = True
        for opt_t in opt_ts:
            for i in range(0, self.data.shape[0]):
                mod_target = []
                append = False
                
                for ind in range(0,len(opt_t)):
                    #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                    temp_ind = self.target[i][opt_t[ind][0]:opt_t[ind][1]].tolist()
                    if any(x == 1.0 for x in temp_ind):
                        mod_target.append(1.0)
                        append = True
                    else:
                        mod_target.append(0.0)
                    del(temp_ind)
                if append:
                    if first:
                        with h5py.File(file, "w") as temp_h5_file:
                            temp_h5_file.create_dataset("X_data", data=[self.data[i]], maxshape=(None,self.input_len))
                            temp_h5_file.create_dataset("Y_data", data=[np.array(mod_target)], maxshape=(None,len(opt_ts[0])))
                        first = False
                    else:
                        with h5py.File(file, "a") as temp_h5_file:
                            temp_h5_file["X_data"].resize((temp_h5_file["X_data"].shape[0] + 1), axis = 0)
                            temp_h5_file["X_data"][-1] = self.data[i]
                            temp_h5_file["Y_data"].resize((temp_h5_file["Y_data"].shape[0] + 1), axis = 0)
                            temp_h5_file["Y_data"][-1] = np.array(mod_target)

            for i in range(0, self.data_2.shape[0]):
                mod_target = []
                append = False
                
                for ind in range(0,len(opt_t)):
                    #temp_ind = target[0][indicies[ind]:indicies[ind+1]].tolist()
                    temp_ind = self.target_2[i][opt_t[ind][0]:opt_t[ind][1]].tolist()
                    if any(x == 1.0 for x in temp_ind):
                        mod_target.append(1.0)
                        append = True
                    else:
                        mod_target.append(0.0)
                    del(temp_ind)
                if append:
                    if first:
                        with h5py.File(file, "w") as temp_h5_file:
                            temp_h5_file.create_dataset("X_data", data=[self.data[i]], maxshape=(None,self.input_len))
                            temp_h5_file.create_dataset("Y_data", data=[np.array(mod_target)], maxshape=(None,len(opt_ts[0])))
                        first = False
                    else:
                        with h5py.File(file, "a") as temp_h5_file:
                            temp_h5_file["X_data"].resize((temp_h5_file["X_data"].shape[0] + 1), axis = 0)
                            temp_h5_file["X_data"][-1] = self.data[i]
                            temp_h5_file["Y_data"].resize((temp_h5_file["Y_data"].shape[0] + 1), axis = 0)
                            temp_h5_file["Y_data"][-1] = np.array(mod_target)
        temp_h5_file = h5py.File(file, 'r')
        return temp_h5_file['X_data'], temp_h5_file['Y_data']

    def __getitem__(self, index): 
        x_noized = data_noizer_waky(self.opt_data[index])
        return (torch.from_numpy(x_noized).float(),torch.from_numpy(self.opt_target[index]).float())
        #return (torch.from_numpy(self.opt_data[index]).float(),torch.from_numpy(self.opt_target[index]).float())

    def __len__(self):
        return len(self.opt_data)

def load_data_depricated(mod_name):
    #X_data = np.load(path+'input_X_1d_np_conf.npy')
    #Y_data = np.load(path+'input_Y_1d_np_conf.npy')

    d_train = []
    d_test = []

    files = glob.glob(path+'input_X_1d_np_'+mod_name+'_noized_*')

    for package in range(0,len(files)):
        X_data = np.load(path+'input_X_1d_np_'+mod_name+'_noized_'+str(package+1)+'.npy')
        Y_data = np.load(path+'input_Y_1d_np_'+mod_name+'_noized_'+str(package+1)+'.npy')
        print(X_data.shape)
        print(Y_data.shape)
        #X_data, Y_data = shuffle(X_data, Y_data)
        input_len = 0
        if package == 0:
            for x in range(TEST_SIZE,TEST_SIZE+10):
                if X_data.shape[1] >= Y_data.shape[1]:
                    d_train.append((torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float)))
                    input_len = X_data.shape[1]
                else:
                    input_len = Y_data.shape[1]
                    d_train.append(( torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float), torch.tensor(Y_data[x],dtype=torch.float)))

            for x in range(0,TEST_SIZE):
                if X_data.shape[1] >= Y_data.shape[1]:
                    d_test.append((torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float)))
                else:
                    d_test.append( ( torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float) , torch.tensor(Y_data[x],dtype=torch.float) ))
        else:
            for x in range(0,X_data.shape[0]):
                if X_data.shape[1] >= Y_data.shape[1]:
                    d_train.append((torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float)))
                    input_len = X_data.shape[1]
                else:
                    input_len = Y_data.shape[1]
                    d_train.append(( torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float), torch.tensor(Y_data[x],dtype=torch.float)))

    print(len(d_test))

    train_loader = torch.utils.data.DataLoader(d_train,batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(d_test,batch_size=BATCH_SIZE, shuffle=True)
    print("Data loaded")
    return (train_loader,test_loader,input_len)

def load_data_as_one_depricated(mod_name):
    #X_data = np.load(path+'input_X_1d_np_conf.npy')
    #Y_data = np.load(path+'input_Y_1d_np_conf.npy')

    d_train = SignDataset_depr([],[])
    d_test = SignDataset_depr([],[])

    files = glob.glob(path+'input_X_1d_np_'+mod_name+'_noized_*')

    for package in range(0,len(files)):
        X_data = np.load(path+'input_X_1d_np_'+mod_name+'_noized_'+str(package+1)+'.npy')
        Y_data = np.load(path+'input_Y_1d_np_'+mod_name+'_noized_'+str(package+1)+'.npy')
        print(X_data.shape)
        print(Y_data.shape)
        #X_data, Y_data = shuffle(X_data, Y_data)
        print("Data shuffeld",end='\r')
        input_len = 0
        if package == 0:
            for x in range(TEST_SIZE,X_data.shape[0]):
                if X_data.shape[1] >= Y_data.shape[1]:
                    d_train.add_data(torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float))
                    input_len = X_data.shape[1]
                else:
                    input_len = Y_data.shape[1]
                    d_train.add_data(torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float), torch.tensor(Y_data[x],dtype=torch.float))

            for x in range(0,TEST_SIZE):
                if X_data.shape[1] >= Y_data.shape[1]:
                    d_test.add_data(torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float))
                else:
                    d_test.add_data(torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float) , torch.tensor(Y_data[x],dtype=torch.float))
        else:
            for x in range(0,X_data.shape[0]):
                if X_data.shape[1] >= Y_data.shape[1]:
                    d_train.add_data(torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float))
                    input_len = X_data.shape[1]
                else:
                    input_len = Y_data.shape[1]
                    d_train.add_data(torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float), torch.tensor(Y_data[x],dtype=torch.float))
        del(X_data,Y_data)
    print('TEST Size: '+ str(d_test.__len__()))
    print('TRAINING SET Size: '+ str(d_train.__len__()))

    train_loader = torch.utils.data.DataLoader(d_train,batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(d_test,batch_size=BATCH_SIZE, shuffle=True)
    print("Data loaded")
    return (train_loader,test_loader,input_len)

def load_data(mod_name):
    d_train = SignDataset(mod_name, '_noized_')
    train_loader = torch.utils.data.DataLoader(d_train,batch_size=BATCH_SIZE, shuffle=True, num_workers = 3)
    print('Training SET Size loaded: '+ str(d_train.__len__()))
   
    d_valid = SignDataset(mod_name, '_valid_')
    test_loader = torch.utils.data.DataLoader(d_valid,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
    print('Valiadation SET Size loaded: '+ str(d_valid.__len__()))
    
    return (train_loader,test_loader,d_train.input_len)

def load_h5_data(mod_name):
    d_train = H5Dataset(path+'lf_train_input_keys_1d_np_'+mod_name+"_h5.hdf5")
    #train_loader = torch.utils.data.DataLoader(d_train,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
    train_loader = []
    print('Training SET Size loaded: '+ str(d_train.__len__()))
   
    d_valid = H5Dataset(path+'lf_test_input_keys_1d_np_'+mod_name+"_h5.hdf5")
    test_loader = torch.utils.data.DataLoader(d_valid,batch_size=1, shuffle=True, num_workers = 1)
    #test_loader = []
    print('Valiadation SET Size loaded: '+ str(d_valid.__len__()))
    
    return (train_loader, test_loader, d_train.input_len, d_train.tree_len)

def load_all_data_in_mem(mod_name):
    #X_data = np.load(path+'input_X_1d_np_conf.npy')
    #Y_data = np.load(path+'input_Y_1d_np_conf.npy')

    d_train = SignDataset_depr([],[])

    files = glob.glob(path+'input_X_1d_np_'+mod_name+'_noized_*')
    input_len = 0
    for package in range(0,len(files)):
    #for package in range(0,1):
        X_data = np.load(path+'input_X_1d_np_'+mod_name+'_noized_'+str(package+1)+'.npy')
        Y_data = np.load(path+'input_Y_1d_np_'+mod_name+'_noized_'+str(package+1)+'.npy')

        #X_data, Y_data = shuffle(X_data, Y_data)
        #print("Training Data shuffeld")

        for x in range(0,X_data.shape[0]):
        #for x in range(0,75):
            if X_data.shape[1] >= Y_data.shape[1]:
                d_train.add_data(torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float))
                if package == 0 and x == 0:
                    input_len = X_data.shape[1]
                    #print(X_data.shape)
                    #print(Y_data.shape)
            else:
                if package == 0 and x == 0:
                    #print(X_data.shape)
                    #print(Y_data.shape)
                    input_len = Y_data.shape[1]
                d_train.add_data(torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float), torch.tensor(Y_data[x],dtype=torch.float))
        del(X_data,Y_data)

    train_loader = torch.utils.data.DataLoader(d_train,batch_size=1, shuffle=True, pin_memory=True, num_workers = 6)
    print('Training SET Size loaded: '+ str(d_train.__len__()))
    test_loader = load_validation_data(mod_name)
    
    return (train_loader,test_loader,input_len)

def load_validation_data(mod_name):
    #X_data = np.load(path+'input_X_1d_np_conf.npy')
    #Y_data = np.load(path+'input_Y_1d_np_conf.npy')

    d_test = SignDataset_depr([],[])

    files = glob.glob(path+'input_X_1d_np_'+mod_name+'_valid_*')
    input_len = 0
    for package in range(0,len(files)):
        X_data = np.load(path+'input_X_1d_np_'+mod_name+'_valid_'+str(package+1)+'.npy')
        Y_data = np.load(path+'input_Y_1d_np_'+mod_name+'_valid_'+str(package+1)+'.npy')

        #X_data, Y_data = shuffle(X_data, Y_data)
        #print("Validation Data shuffeld")
        
        for x in range(0,X_data.shape[0]):
            if X_data.shape[1] >= Y_data.shape[1]:
                d_test.add_data(torch.tensor(X_data[x],dtype=torch.float),torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float))
                if package == 0 and x == 0:
                    input_len = X_data.shape[1]
                    #print("Validation "+str(X_data.shape))
                    #print("Validation "+str(Y_data.shape))
            else:
                if package == 0 and x == 0:
                    input_len = Y_data.shape[1]
                    #print("Validation "+str(X_data.shape))
                    #print("Validation "+str(Y_data.shape))                    
                d_test.add_data(torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float), torch.tensor(Y_data[x],dtype=torch.float))
        del(X_data,Y_data)
   

    test_loader = torch.utils.data.DataLoader(d_test,batch_size = 1, shuffle=True, pin_memory=True)
    print('Validation Set Size loaded: '+ str(d_test.__len__()))
    return test_loader

def level_load_data(file_name, mod_name, d_train, opt_t):


    #if d_train.__len__() > 32:
    #    batch_s = 32
    #if d_train.__len__() > 16 and d_train.__len__() <= 32:
    #    batch_s = 16
    #if d_train.__len__() <= 16:
    #    batch_s = 1

    #batch_s = int(d_train.__len__()*0.25)
    print('TRAINING SET Size: '+ str(d_train.__len__())+ ' BATCH Size : '+str(batch_s)+ ' CLASSES : '+str(len(opt_t)))
    if d_train.__len__()>0:
        train_loader = torch.utils.data.DataLoader(d_train,batch_size=batch_s, shuffle=True, pin_memory=True)
    else:
        train_loader = None
    print("Training data loaded")
    
    return (train_loader, input_len)

def level_load_valid_data(mod_name, d_test, opt_t):

    files = glob.glob(path+'input_X_1d_np_'+mod_name+'_valid_*')
    input_len = 0
    for package in range(0,len(files)):
        X_data = np.load(path+'input_X_1d_np_'+mod_name+'_valid_'+str(package+1)+'.npy')
        Y_data = np.load(path+'input_Y_1d_np_'+mod_name+'_valid_'+str(package+1)+'.npy')

        #X_data, Y_data = shuffle(X_data, Y_data)

        for x in range(0,X_data.shape[0]):
            input = []
            target = []
            if X_data.shape[1] >= Y_data.shape[1]:
                target = torch.tensor(np.hstack((Y_data[x], np.zeros(X_data.shape[1]-Y_data.shape[1], ))),dtype=torch.float)
                input = torch.tensor(X_data[x],dtype=torch.float)
            else:
                target = torch.tensor(Y_data[x],dtype=torch.float)
                input = torch.tensor(np.hstack((X_data[x],np.zeros(Y_data.shape[1]-X_data.shape[1],))),dtype=torch.float)
            
            mod_target = []
            append = False
            for ind in range(0,len(opt_t)):
                temp_ind = target[opt_t[ind][0]:opt_t[ind][1]].tolist()
                if any(x == 1.0 for x in temp_ind):
                    mod_target.append(1.0)
                    append = True
                else:
                    mod_target.append(0.0)
                del(temp_ind)
            if append:
                d_test.add_data(input,torch.tensor(mod_target,dtype=torch.float))
            
        del(X_data,Y_data)

    print('VALIDATION SET Size: '+ str(d_test.__len__()))
    if d_test.__len__()>0:
        test_loader = torch.utils.data.DataLoader(d_test,batch_size=1, shuffle=True, pin_memory=True)
    else:
        test_loader = None
    print("Validation data loaded")
    
    return test_loader

def count_classes(tree):
    if tree.is_leaf():
        return
    else:
        choicer = False

        for i in range(0, (len(tree.children))):
            if tree.children[i].name.find('OPT') != -1:
                choicer = True

        if choicer:
                #if hasattr(tree, 'model')==True:
                 #   if MAX_LEVEL <= tree.level:
                        options = []
                        
                        for i in range(0, (len(tree.children))):
                            if tree.children[i].name.find('OPT') != -1 and tree.children[i].name != "NON_OPT":
                                    options.append(tree.children[i])

                        if len(options)>1:
                            global submodels_total 
                            submodels_total += 1
                        del(options)
                        for o_i in range(0,len(tree.children)):
                            count_classes(tree.children[o_i])
        else:
            for o_i in range(0,len(tree.children)):
                count_classes(tree.children[o_i])

def get_model_names_and_inds(tree, start_ind, model_names):
    if tree.is_leaf():
        return
    else:
        choicer = False

        indicies = [start_ind]

        for i in range(1, (len(tree.children)+1)): #Cumulative indicies 
            tmp_leaves = list(tree.children[i-1].iter_leaves())
            indicies.append(indicies[i-1]+len(tmp_leaves))
            del(tmp_leaves)


        for i in range(0, (len(tree.children))):
            if tree.children[i].name.find('OPT') != -1:
                choicer = True
        for i in range(0, (len(tree.children))):
            if tree.children[i].name.find('OPT') != -1:
                choicer = True

        if choicer:
            options = []
            
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
                has =False
                for n in model_names:
                    if n[0] == tree.name:
                        n[1].append(opt_target)
                        has = True
                if has == False:
                    model_names.append((tree.name,[opt_target]))

                global submodels_total 
                submodels_total += 1
            del(options)
            for o_i in range(0,len(tree.children)):
                get_model_names_and_inds(tree.children[o_i],indicies[o_i],model_names)
        else:
            for o_i in range(0,len(tree.children)):
                get_model_names_and_inds(tree.children[o_i],indicies[o_i],model_names)

def train_transfer(mod_name, m_names):
    for m in m_names:
        try: 
            valid_dataset = H5Dataset_level_validation_transfer(path+'test_input_keys_1d_np_'+mod_name+"_h5.hdf5", path+'rs_test_input_keys_1d_np_'+mod_name+"_h5.hdf5",m[1])
            test_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=1, shuffle=True, num_workers = 1)
        except:
            test_dataloader = None
            print('No Test Samples found')
        try:
            node_dataset = H5Dataset_level_transfer(path+'train_input_keys_1d_np_'+mod_name+"_h5.hdf5",m[1])
            train_dataloader = torch.utils.data.DataLoader(node_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
        except:
            train_dataloader = None
            print('No Train Samples found')
        if train_dataloader != None:
            model_t = Sequential(node_dataset.input_len,len(m[1][0])).to(DEVICE)
            model_t.apply(init_weights)
            optimizer_t = optim.Adam(model_t.parameters(),lr= LR)
            train_result = train_one_node_full(model_t, DEVICE, train_dataloader, optimizer_t, test_dataloader)
            joblib.dump(model_t, path+"../models/transfer_models/"+m[0]+".joblib")
            del(model_t,node_dataset,train_dataloader,optimizer_t)

def print_results(tree):
    def Average(lst): 
        average = sum(lst) / len(lst) 
        return round(average, 2)
    levels = []
    res = []

    for node in tree.traverse("levelorder"):
        if hasattr(node, 'train_result'):
            if node.train_result != None:
                levels.append(node.level)
                res.append( ( node.level, node.samples_per_class, node.classes, node.train_result[1] ) )
                #print('Level : '+str(node.level)+' Samp/Class: '+str(node.samples_per_class)+' Classes: '+str(node.classes)+' Result '+str(node.train_result))
    levels_set  = sorted(set(levels))
    del(levels)
    for level in levels_set:
        level = (level, Average(list([s[1] for s in res if s[0]==level])), Average(list([s[2] for s in res if s[0]==level])),Average(list([s[3] for s in res if s[0]==level])), len(list([s for s in res if s[0]==level])))
        print(level)

def multi_node_train(tree):
    #print('Train node: '+str(tree.number))
    start_time = datetime.datetime.now()
    valid_dataset = None
    node_dataset = None
    global lock_train_data, lock_val_data
    while node_dataset == None:
        if lock_train_data.value == 0:
            try:
                lock_train_data.value = 1
                node_dataset = H5Dataset_level(path+'lf_train_input_keys_1d_np_'+mod_name+"_h5.hdf5",tree.opt_target)
                train_dataloader = torch.utils.data.DataLoader(node_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
            except:
                train_dataloader = None
        else: pass
            #  print('No Train Samples found')
    lock_train_data.value = 0

    while valid_dataset == None:
        if lock_val_data.value == 0:
            try: 
                lock_val_data.value = 1
                valid_dataset = H5Dataset_level_validation(path+'lf_test_input_keys_1d_np_'+mod_name+"_h5.hdf5", path+'lf_rs_test_input_keys_1d_np_'+mod_name+"_h5.hdf5",tree.opt_target)
                test_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers = 1)
            except:
                test_dataloader = None
        else: pass

                #print('No Test Samples found')
    lock_val_data.value = 0

    if train_dataloader != None and test_dataloader != None:
        #if :
        if valid_dataset:
            print('Node',tree.number,'Train smp: ',node_dataset.pool,' Valid smp : ',valid_dataset.pool)
        else:
            print('Node',tree.number,'Train smp: ',node_dataset.pool,' Valid smp : ',None)
        attemts_res = []
        #print(str(i) + ' LR :'+str(LR/((10)**i)))
        model_t = Sequential(loader[2],len(tree.opt_target)).to(DEVICE,non_blocking=True)
        model_t.apply(init_weights)
        #optimizer_t = optim.Adam(model_t.parameters(),lr=LR,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
        optimizer_t = torch.optim.Adam(model_t.parameters(),LR)
        #optimizer_t = optim.SGD(model_t.parameters(),lr=LR, momentum= 0.9, nesterov=True)
        scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'min')
        #scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'min',cooldown=2,patience=4)
        train_result = train_one_node_full(model_t, DEVICE, train_dataloader,optimizer_t, scheduler_t, test_dataloader)
        attemts_res.append((model_t,optimizer_t,LR,train_result))
        model_t = None
        optimizer_t = None
        train_result = None
        max_r = list([res[3][1] for res in attemts_res])
        for r in attemts_res:
            if r[3][1] == max(max_r) and model_t == None:
                #print('Best result : '+str(r[3])+' LR: '+str(r[2]))
                model_t = r[0]
                optimizer_t = r[1]
                train_result = r[3]
        del(attemts_res)
    else:
        model_t = Sequential(loader[2],len(tree.opt_target)).to(DEVICE,non_blocking=True)
        model_t.apply(init_weights)
        optimizer_t = torch.optim.Adam(model_t.parameters(),LR)
        #optimizer_t = optim.Adam(model_t.parameters(),lr=LR,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
        #optimizer_t = optim.SGD(model_t.parameters(),lr=LR, momentum= 0.9, nesterov=True)

        scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'min')
        #scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'min',cooldown=2,patience=4)
        train_result = None

    tree.add_feature('model',model_t)
    tree.add_feature('scheduler',scheduler_t)
    tree.add_feature('train_result',train_result)
    tree.add_feature('samples_per_class',int(node_dataset.__len__()/len(tree.opt_target)))
    tree.add_feature('optimizer',optimizer_t)
    del(model_t,node_dataset,train_dataloader,optimizer_t)
    #del(model_t,optimizer_t)
    tree.done_epochs += EPOCHS
    tree.add_feature('done_training',True)
    joblib.dump(tree, path+"../models/nodes/"+str(tree.number)+".joblib")
    global submodels_done
    global submodels_total
    submodels_done.value += 1
    time_ = datetime.datetime.now() - start_time
    print(str(int(submodels_done.value/submodels_total*100))+'% trained '+str(submodels_done.value)+'/'+str(submodels_total)+' Hours left: '+str((time_*(submodels_total-submodels_done.value)/(multiprocessing.cpu_count()-1))))
    return

def mask_samples(orig):
    mask = orig[0]
    for i in range(0,orig.shape[0]):
        #print(mask)
        mask = torch.eq(mask,orig[i])
    mask = torch.logical_not(mask).type(torch.ByteTensor)
    return torch.mul(orig,mask), mask

#batch = torch.tensor([[[1,2],[1,3]],[[1,2],[1,2]],[[1,1],[1,2]],[[1,1],[2,3]]]).type(torch.float)

BATCH_SIZE=32
EPOCHS=300
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda")#use that for multiprocessing
LR = 0.00321
#MAX_LEVEL = 5
#TEST_SIZE = 1100

parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')
args = parser.parse_args()
path = args.output_dest[0]

print("NEURAL NETWORK TREE")
#print("BATCH SIZE: "+str(BATCH_SIZE)+" EPOCHS: "+str(EPOCHS)+" DEVICE: "+str(DEVICE)+" LR: "+str(LR))
print("EPOCHS: "+str(EPOCHS)+" DEVICE: "+str(DEVICE)+" LR: "+str(LR))
#print("TRAIN ALL KEY POINTS")

print("\n")
print('HAND CONFIGURATION:')
big_loss = 0
base = 0

submodels_total = 0
submodels_done = 0

loader = load_h5_data('lr_hand_conf') #'all_h_conf'
vec = vector2sigml.v2s.Vec2sigml(np.ones_like(loader[3]))
count_classes(Tree(vec.h_conf_tree_path,format=1))
#level_train(vec.h_conf_tree_path,'lr_hand_conf','lr_hand_conf',loader[2],loader[3])
#t = joblib.load(path+"../models/nn_tree_"+'lr_hand_conf'+".joblib",)


#def multi():
t = Tree(vec.h_conf_tree_path,format=1)
prepare_for_multitrain('multi_train',t,0,loader[2],0)

#joblib.dump(t, path+"../models/1_nn_tree_"+'multi_train'+".joblib")
#print('Prepared')
mod_name = 'lr_hand_conf'


multi_nodes = list([node for node in t.traverse("levelorder") if hasattr(node, 'opt_target')])

for node in t.traverse("levelorder"):
    if hasattr(node, 'opt_target'):
        for i in range(0,len(multi_nodes)):
            if multi_nodes[i]==node:
                node.add_feature('number',i)

#joblib.dump(t, path+"../models/2_nn_tree_"+'multi_train'+".joblib")

multi_nodes = list([node for node in t.traverse("levelorder") if hasattr(node, 'opt_target')])

print('Collected nodes :'+str(len(multi_nodes)))
#try:
#     set_start_method('spawn')
#except RuntimeError:
#    pass

submodels_done = multiprocessing.Value('i', 0)
lock_train_data = multiprocessing.Value('i',0)
lock_val_data = multiprocessing.Value('i',0)
#use one less process to be a little more stable
#p = MyPool(processes = multiprocessing.cpu_count()-1)
#p = MyPool(processes = 9)

#timing it...
#start = time.time()
#print('Start Training')
#p.map(multi_node_train, multi_nodes)

#multi_node_train(multi_nodes[3])
#p.close()
#p.join()
#print("Node Train Complete")
#end = time.time()
#print('total time (s)= ' + str(end-start))
#joblib.dump(t, path+"../models/2_nn_tree_"+'multi_train'+".joblib")
#t = joblib.load(path+"../models/2_nn_tree_"+'multi_train'+".joblib",)

#for node in t.traverse("levelorder"):
#    if hasattr(node, 'opt_target'):
#        node_2 = joblib.load(path+"../models/nodes/"+str(node.number)+".joblib",)
#        node.add_feature('model',node_2.model)
#        node.add_feature('level',node_2.level)
#        node.add_feature('train_result',node_2.train_result)
#        node.add_feature('optimizer',node_2.optimizer)
#        node.add_feature('samples_per_class',node_2.samples_per_class) #done_epochs
#        node.add_feature('done_epochs',node_2.done_epochs)
#        print(str(node.number),end='\r')

#joblib.dump(t, path+"../models/3_nn_tree_"+'multi_train'+".joblib")
#print('Done training HAND CONFIGURATION')

#test('multi_train', loader[1])
#t = joblib.load(path+"../models/3_nn_tree_"+'multi_train'+".joblib",)
annotate("../models/3_nn_tree_"+'multi_train'+".joblib","greek_lf_test_input_keys_1d_np_lr_hand_conf_h5.hdf5",)
#print_results(t)
#test('lr_hand_conf', loader[1])
del(loader,vec)

#print("\n")
#print('HAND ORIENTATION:')
#big_loss = 0
#base = 0
#loader = load_h5_data('lr_hand_orient')
#vec = vector2sigml.v2s.Vec2sigml(np.ones_like(loader[3]))
#level_train(vec.h_or_tree_path,'or',loader[2])
#train(vec.h_or_tree_path, 'lr_hand_orient' ,loader[0],loader[1],loader[2])

#test('lr_hand_orient', loader[1])
#evel_test('or_all', loader[1])
#print('Done training HAND ORIENTATION')
#del(loader,vec)

#print("\n")
#print('HAND LOCATION:')
#big_loss = 0
#base = 0
#loader = load_data('loc_all')
#vec = vector2sigml.v2s.Vec2sigml(np.ones_like(loader[2]))
#level_train(vec.h_loc_tree_path, str(MAX_LEVEL)+'_level_lr_loc_all','loc_all',loader[2])
#test('loc_all', loader[1])
#print('Done training HAND LOCATION')
#del(loader,vec)

