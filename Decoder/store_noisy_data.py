import glob
import pathlib
import argparse
import numpy as np
import psutil
import random
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')

args = parser.parse_args()

path = args.output_dest[0]

dirs = glob.glob(path+'/*/')
dirs.sort()

MEMORY_LIMIT = 60
VALID_SET_COEFF = 0.1

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

def data_appender(X_name, Y_list, X_data, Y_data, validation = False):
    x_d_np = np.load(X_name)
    if validation == False:
        X_data.append(list(x_d_np))
        Y_data.append(Y_list)
        for x in range(0, 3): 
            X_noised_np = data_noizer(x_d_np)
            X_data.append(list(X_noised_np))
            Y_data.append(Y_list)
        for x in range(0, 7): 
            X_noised_np = data_noizer_waky(x_d_np)
            X_data.append(list(X_noised_np))
            Y_data.append(Y_list)
    else:
        #X_noised_np = data_noizer_waky(x_d_np)
        #X_data.append(list(X_noised_np))
        X_data.append(list(x_d_np))
        Y_data.append(Y_list)
    return
h5 = "50000_train.hdf5"

def create_h5(file, X_data, Y_data):
    with h5py.File(file, "w") as h5f:
        h5f.create_dataset("X_data", data=X_data, maxshape=(None,))
        h5f.create_dataset("Y_data", data=Y_data, maxshape=(None,))
    return

def add_to_h5(file, X_data, Y_data):
    with h5py.File(file, "a") as h5f:
        h5f["X_data"].resize((h5f["X_data"].shape[0] + X_data.shape[0]), axis = 0)
        h5f["X_data"][-X_data.shape[0]:] = X_data

        h5f["Y_data"].resize((h5f["Y_data"].shape[0] + Y_data.shape[0]), axis = 0)
        h5f["Y_data"][-Y_data.shape[0]:] = Y_data
    return

def h5py_data_appender(h5py_filename, X_name, Y_name, validation = False):
    x_d_np = np.load(X_name)
    y_d_np = np.load(Y_name)
    if validation == False:
        add_to_h5(h5py_filename,x_d_np,y_d_np)
        for x in range(0, 3): 
            X_noised_np = data_noizer(x_d_np)
            add_to_h5(h5py_filename,x_d_np,y_d_np)
        for x in range(0, 7): 
            X_noised_np = data_noizer_waky(x_d_np)
            add_to_h5(h5py_filename,x_d_np,y_d_np)
    else:
        add_to_h5(h5py_filename,x_d_np,y_d_np)
    return


X_data_list = []
Y_data_list = []

X_data_conf_list = []
X_data_or_list = []
X_data_loc_list = []

Y_data_conf_list = []
Y_data_or_list = []
Y_data_loc_list = []


store_all = False

if store_all == True:
    print("Saving All")
    print('Validation set lenght: '+str(len(dirs)*VALID_SET_COEFF))
    package = 1
    for d in range(0,len(dirs)):

        if d < (int(len(dirs)*VALID_SET_COEFF)):
            p = pathlib.PurePath(dirs[d])
            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'))+list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'))+list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_list,Y_data_list,True)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == ((int(len(dirs)*VALID_SET_COEFF))-1):
                X_data = np.array(X_data_list)
                np.save(path+'input_X_1d_np_all_valid_'+str(package)+'.npy',X_data)
                X_data_list = []
                Y_data = np.array(Y_data_list)
                np.save(path+'input_Y_1d_np_all_valid_'+str(package)+'.npy',Y_data)
                Y_data_list = []
                #print('Saved as: '+str(path+'input_Y_1d_np_all_valid_'+str(package)+'.npy'))
                #print('X shape, Y shape'+str(X_data.shape)+','+str(Y_data.shape))
                if d != ((int(len(dirs)*VALID_SET_COEFF))-1):
                    package += 1
                else:
                    package = 1
                del(X_data,Y_data)
        else:     
            p = pathlib.PurePath(dirs[d])
            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy'))+list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy'))+list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_list,Y_data_list)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == (len(dirs)-1):
                X_data = np.array(X_data_list)
                np.save(path+'input_X_1d_np_all_noized_'+str(package)+'.npy',X_data)
                X_data_list = []
                Y_data = np.array(Y_data_list)
                np.save(path+'input_Y_1d_np_all_noized_'+str(package)+'.npy',Y_data)
                Y_data_list = []
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data.shape)+','+str(Y_data.shape))
                package += 1
                del(X_data,Y_data)
            
        print("Done : "+str(100. * d/len(dirs)), end='\r')
else:
    print("Saving CONF_ALL")
    package = 1
    for d in range(0,len(dirs)):

        if d < (int(len(dirs)*VALID_SET_COEFF)):
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy')),X_data_conf_list,Y_data_conf_list,True)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == ((int(len(dirs)*VALID_SET_COEFF))-1):
                X_data_conf = np.array(X_data_conf_list)
                np.save(path+'input_X_1d_np_conf_all_valid_'+str(package)+'.npy',X_data_conf)
                X_data_conf_list = []

                Y_data_conf = np.array(Y_data_conf_list)
                np.save(path+'input_Y_1d_np_conf_all_valid_'+str(package)+'.npy',Y_data_conf)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_conf.shape)+','+str(Y_data_conf.shape))
                Y_data_conf_list = []
                if d != ((int(len(dirs)*VALID_SET_COEFF))-1):
                    package += 1
                else:
                    package = 1
                del(X_data_conf,Y_data_conf)
        else:
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy')),X_data_conf_list,Y_data_conf_list)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == (len(dirs)-1):
                X_data_conf = np.array(X_data_conf_list)
                np.save(path+'input_X_1d_np_conf_all_noized_'+str(package)+'.npy',X_data_conf)
                X_data_conf_list = []

                Y_data_conf = np.array(Y_data_conf_list)
                np.save(path+'input_Y_1d_np_conf_all_noized_'+str(package)+'.npy',Y_data_conf)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_conf.shape)+','+str(Y_data_conf.shape))
                Y_data_conf_list = []
                package += 1
                del(X_data_conf,Y_data_conf)
            
        print("Done : "+str(100. * d/len(dirs)), end='\r')

    print("Saving ORIENT_ALL")
    package = 1
    for d in range(0,len(dirs)):
        if d < (int(len(dirs)*VALID_SET_COEFF)):
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy')),X_data_or_list,Y_data_or_list,True)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == ((int(len(dirs)*VALID_SET_COEFF))-1):
                X_data_or = np.array(X_data_or_list)
                np.save(path+'input_X_1d_np_or_all_valid_'+str(package)+'.npy',X_data_or)
                X_data_or_list = []

                Y_data_or = np.array(Y_data_or_list)
                np.save(path+'input_Y_1d_np_or_all_valid_'+str(package)+'.npy',Y_data_or)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_or.shape)+','+str(Y_data_or.shape))
                Y_data_or_list = []
                if d != ((int(len(dirs)*VALID_SET_COEFF))-1):
                    package += 1
                else:
                    package = 1
                del(X_data_or,Y_data_or)
        else:
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy')),X_data_or_list,Y_data_or_list)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == (len(dirs)-1):
                X_data_or = np.array(X_data_or_list)
                np.save(path+'input_X_1d_np_or_all_noized_'+str(package)+'.npy',X_data_or)
                X_data_or_list = []

                Y_data_or = np.array(Y_data_or_list)
                np.save(path+'input_Y_1d_np_or_all_noized_'+str(package)+'.npy',Y_data_or)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_or.shape)+','+str(Y_data_or.shape))
                Y_data_or_list = []
                package += 1
                del(X_data_or,Y_data_or)
            
        print("Done : "+str(100. * d/len(dirs)), end='\r')

    print("Saving LOCATION_ALL")
    package = 1
    for d in range(0,len(dirs)):
        if d < (int(len(dirs)*VALID_SET_COEFF)):
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_loc_list,Y_data_loc_list,True)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == ((int(len(dirs)*VALID_SET_COEFF))-1):
                X_data_loc = np.array(X_data_loc_list)
                np.save(path+'input_X_1d_np_loc_all_valid_'+str(package)+'.npy',X_data_loc)
                X_data_loc_list = []

                Y_data_loc = np.array(Y_data_loc_list)
                np.save(path+'input_Y_1d_np_loc_all_valid_'+str(package)+'.npy',Y_data_loc)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_loc.shape)+','+str(Y_data_loc.shape))
                Y_data_loc_list = []
                if d != ((int(len(dirs)*VALID_SET_COEFF))-1):
                    package += 1
                else:
                    package = 1
                del(X_data_loc,Y_data_loc)
            
        else:
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_all_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_loc_list,Y_data_loc_list)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == (len(dirs)-1):
                X_data_loc = np.array(X_data_loc_list)
                np.save(path+'input_X_1d_np_loc_all_noized_'+str(package)+'.npy',X_data_loc)
                X_data_loc_list = []

                Y_data_loc = np.array(Y_data_loc_list)
                np.save(path+'input_Y_1d_np_loc_all_noized_'+str(package)+'.npy',Y_data_loc)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_loc.shape)+','+str(Y_data_loc.shape))
                Y_data_loc_list = []
                package += 1
                del(X_data_loc,Y_data_loc)
            
        print("Done : "+str(100. * d/len(dirs)), end='\r')

    print("Saving CONF")
    package = 1
    for d in range(0,len(dirs)):

        if d < (int(len(dirs)*VALID_SET_COEFF)):
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_lr_hand_conf_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy')),X_data_conf_list,Y_data_conf_list,True)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == ((int(len(dirs)*VALID_SET_COEFF))-1):
                X_data_conf = np.array(X_data_conf_list)
                np.save(path+'input_X_1d_np_conf_valid_'+str(package)+'.npy',X_data_conf)
                X_data_conf_list = []

                Y_data_conf = np.array(Y_data_conf_list)
                np.save(path+'input_Y_1d_np_conf_valid_'+str(package)+'.npy',Y_data_conf)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_conf.shape)+','+str(Y_data_conf.shape))
                Y_data_conf_list = []
                if d != ((int(len(dirs)*VALID_SET_COEFF))-1):
                    package += 1
                else:
                    package = 1
                del(X_data_conf,Y_data_conf)
        else:
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_lr_hand_conf_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_conf_'+p.parts[-1]+'.npy')),X_data_conf_list,Y_data_conf_list)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == (len(dirs)-1):
                X_data_conf = np.array(X_data_conf_list)
                np.save(path+'input_X_1d_np_conf_noized_'+str(package)+'.npy',X_data_conf)
                X_data_conf_list = []

                Y_data_conf = np.array(Y_data_conf_list)
                np.save(path+'input_Y_1d_np_conf_noized_'+str(package)+'.npy',Y_data_conf)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_conf.shape)+','+str(Y_data_conf.shape))
                Y_data_conf_list = []
                package += 1
                del(X_data_conf,Y_data_conf)
            
        print("Done : "+str(100. * d/len(dirs)), end='\r')

    print("Saving ORIENT")
    package = 1
    for d in range(0,len(dirs)):
        if d < (int(len(dirs)*VALID_SET_COEFF)):
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_lr_hand_orient_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy')),X_data_or_list,Y_data_or_list,True)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == ((int(len(dirs)*VALID_SET_COEFF))-1):
                X_data_or = np.array(X_data_or_list)
                np.save(path+'input_X_1d_np_or_valid_'+str(package)+'.npy',X_data_or)
                X_data_or_list = []

                Y_data_or = np.array(Y_data_or_list)
                np.save(path+'input_Y_1d_np_or_valid_'+str(package)+'.npy',Y_data_or)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_or.shape)+','+str(Y_data_or.shape))
                Y_data_or_list = []
                if d != ((int(len(dirs)*VALID_SET_COEFF))-1):
                    package += 1
                else:
                    package = 1
                del(X_data_or,Y_data_or)
        else:
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_lr_hand_orient_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_or_'+p.parts[-1]+'.npy')),X_data_or_list,Y_data_or_list)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == (len(dirs)-1):
                X_data_or = np.array(X_data_or_list)
                np.save(path+'input_X_1d_np_or_noized_'+str(package)+'.npy',X_data_or)
                X_data_or_list = []

                Y_data_or = np.array(Y_data_or_list)
                np.save(path+'input_Y_1d_np_or_noized_'+str(package)+'.npy',Y_data_or)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_or.shape)+','+str(Y_data_or.shape))
                Y_data_or_list = []
                package += 1
                del(X_data_or,Y_data_or)
            
        print("Done : "+str(100. * d/len(dirs)), end='\r')

    print("Saving LOCATION")
    package = 1
    for d in range(0,len(dirs)):
        if d < (int(len(dirs)*VALID_SET_COEFF)):
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_lr_hand_location_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_loc_list,Y_data_loc_list,True)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == ((int(len(dirs)*VALID_SET_COEFF))-1):
                X_data_loc = np.array(X_data_loc_list)
                np.save(path+'input_X_1d_np_loc_valid_'+str(package)+'.npy',X_data_loc)
                X_data_loc_list = []

                Y_data_loc = np.array(Y_data_loc_list)
                np.save(path+'input_Y_1d_np_loc_valid_'+str(package)+'.npy',Y_data_loc)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_loc.shape)+','+str(Y_data_loc.shape))
                Y_data_loc_list = []
                if d != ((int(len(dirs)*VALID_SET_COEFF))-1):
                    package += 1
                else:
                    package = 1
                del(X_data_loc,Y_data_loc)
            
        else:
            p = pathlib.PurePath(dirs[d])

            data_appender((dirs[d]+'input_keys_1d_np_lr_hand_location_'+p.parts[-1]+'.npy'),list(np.load(dirs[d]+'input_vector_h_loc_'+p.parts[-1]+'.npy')),X_data_loc_list,Y_data_loc_list)
            
            if psutil.virtual_memory()[2] > MEMORY_LIMIT or d == (len(dirs)-1):
                X_data_loc = np.array(X_data_loc_list)
                np.save(path+'input_X_1d_np_loc_noized_'+str(package)+'.npy',X_data_loc)
                X_data_loc_list = []

                Y_data_loc = np.array(Y_data_loc_list)
                np.save(path+'input_Y_1d_np_loc_noized_'+str(package)+'.npy',Y_data_loc)
                #print('Saved as: '+str())
                #print('X shape, Y shape'+str(X_data_loc.shape)+','+str(Y_data_loc.shape))
                Y_data_loc_list = []
                package += 1
                del(X_data_loc,Y_data_loc)
            
        print("Done : "+str(100. * d/len(dirs)), end='\r')
