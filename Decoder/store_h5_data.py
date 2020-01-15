import glob
import pathlib
import argparse
import numpy as np
import random
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')

args = parser.parse_args()

path = args.output_dest[0]

dirs = glob.glob(path+'/*/')
dirs2 = glob.glob('add_rs_training_10000/'+'/*/')
dirs = dirs+dirs2
#dirs.sort()
for i in range(5):
    random.shuffle(dirs)

VALID_SET = 6000
print('Validation set lenght: '+str(VALID_SET))
print('Training set lenght: '+str(len(dirs)-VALID_SET))

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

def data_noizer_waky(X_data):
    noise = np.random.normal(1, 0.1, len(X_data))
    #rand = random.uniform(0.9,1.1)
    #noise  = np.empty(len(X_data))
    #noise.fill(rand)
    for x in range(0,len(X_data)):
        if X_data[x] == 1:
            noise[x] = 1
    X_noized_data = X_data * noise
    return X_noized_data

def create_h5(file, X_data, Y_data):
    x_d_np = np.load(X_data)
    y_d_np = np.load(Y_data)
    with h5py.File(file, "w") as h5f:
        h5f.create_dataset("X_data", data=[x_d_np], maxshape=(None,x_d_np.shape[0]))
        h5f.create_dataset("Y_data", data=[y_d_np], maxshape=(None,y_d_np.shape[0]))
    return

def add_to_h5(file, X_data, Y_data):
    with h5py.File(file, "a") as h5f:
        h5f["X_data"].resize((h5f["X_data"].shape[0] + 1), axis = 0)
        h5f["X_data"][-1] = X_data
        h5f["Y_data"].resize((h5f["Y_data"].shape[0] + 1), axis = 0)
        h5f["Y_data"][-1] = Y_data
    return

def h5py_data_appender(h5py_filename, X_name, Y_name, validation = False):
    x_d_np = np.load(X_name)
    y_d_np = np.load(Y_name)
    add_to_h5(h5py_filename,x_d_np,y_d_np)
    return

"""
print("Saving CONF_ALL")
#data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy')),X_data_conf_list,Y_data_conf_list,True)
for d in range(0,len(dirs)):
    p = pathlib.PurePath(dirs[d])
    if d == 0:
        create_h5(path+"test_input_keys_1d_np_all_h_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'))
    if d < VALID_SET and d > 0 :
        h5py_data_appender(path+"test_input_keys_1d_np_all_h_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'),True)
    if d==VALID_SET:
        create_h5(path+"train_input_keys_1d_np_all_h_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'))
    if d > VALID_SET:
        h5py_data_appender(path+"train_input_keys_1d_np_all_h_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'),False)
    print("Done:",round(100. * d/len(dirs),2), end='\r')

print("Saving ORIENT_ALL")
#data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy')),X_data_or_list,Y_data_or_list,True)
for d in range(0,len(dirs)):
    p = pathlib.PurePath(dirs[d])
    if d == 0:
        create_h5(path+"test_input_keys_1d_np_all_h_or_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'))
    if d < VALID_SET and d > 0 :
        h5py_data_appender(path+"test_input_keys_1d_np_all_h_or_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'),True)
    if d==VALID_SET:
        create_h5(path+"train_input_keys_1d_np_all_h_or_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'))
    if d > VALID_SET: 
        h5py_data_appender(path+"train_input_keys_1d_np_all_h_or_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'),False)
    print("Done:",round(100. * d/len(dirs),2), end='\r')

print("Saving LOCATION_ALL")
#data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_loc_list,Y_data_loc_list,True)
for d in range(0,len(dirs)):
    p = pathlib.PurePath(dirs[d])
    if d == 0:
        create_h5(path+"test_input_keys_1d_np_all_h_loc_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'))
    if d < VALID_SET and d > 0 :
        h5py_data_appender(path+"test_input_keys_1d_np_all_h_loc_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'),True)
    if d==VALID_SET:
        create_h5(path+"train_input_keys_1d_np_all_h_loc_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'))
    if d > VALID_SET:
        h5py_data_appender(path+"train_input_keys_1d_np_all_h_loc_h5.hdf5",(dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'),False)
    print("Done:",round(100. * d/len(dirs),2), end='\r')
"""

print("Saving CONF")
#data_appender((dirs[d]+'input_keys_1d_np_lr_hand_conf_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy')),X_data_conf_list,Y_data_conf_list,True)
for d in range(0,len(dirs)):
    p = pathlib.PurePath(dirs[d])
    if d == 0:
        create_h5(path+"test_input_keys_1d_np_lr_hand_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_conf_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'))
    if d < VALID_SET and d > 0 :
        h5py_data_appender(path+"test_input_keys_1d_np_lr_hand_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_conf_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'),True)
    if d==VALID_SET:
        create_h5(path+"train_input_keys_1d_np_lr_hand_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_conf_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'))
    if d > VALID_SET:
        h5py_data_appender(path+"train_input_keys_1d_np_lr_hand_conf_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_conf_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'),False)
    print("Done:",round(100. * d/len(dirs),2), end='\r')

"""
print("Saving LOCATION")
#data_appender((dirs[d]+'input_keys_1d_np_lr_hand_location_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_loc_list,Y_data_loc_list,True)
for d in range(0,len(dirs)):
    p = pathlib.PurePath(dirs[d])
    if d == 0:
        create_h5(path+"test_input_keys_1d_np_lr_hand_location_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_location_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'))
    if d < VALID_SET and d > 0 :
        h5py_data_appender(path+"test_input_keys_1d_np_lr_hand_location_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_location_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'),True)
    if d==VALID_SET:
        create_h5(path+"train_input_keys_1d_np_lr_hand_location_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_location_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'))
    if d > VALID_SET:  
        h5py_data_appender(path+"train_input_keys_1d_np_lr_hand_location_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_location_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy'),False)
    print("Done:",round(100. * d/len(dirs),2), end='\r')

print("Saving ORIENT")
#    data_appender((dirs[d]+'input_keys_1d_np_lr_hand_orient_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy')),X_data_or_list,Y_data_or_list,True)
for d in range(0,len(dirs)):
    p = pathlib.PurePath(dirs[d])
    if d == 0:
        create_h5(path+"test_input_keys_1d_np_lr_hand_orient_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_orient_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'))
    if d < VALID_SET and d > 0 :
        h5py_data_appender(path+"test_input_keys_1d_np_lr_hand_orient_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_orient_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'),True)
    if d==VALID_SET:
        create_h5(path+"train_input_keys_1d_np_lr_hand_orient_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_orient_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'))
    if d > VALID_SET:
        h5py_data_appender(path+"train_input_keys_1d_np_lr_hand_orient_h5.hdf5",(dirs[d]+'input_keys_1d_np_lr_hand_orient_'+p.parts[-1]+'.npy'),(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'),False)
    print("Done:",round(100. * d/len(dirs),2), end='\r')
"""