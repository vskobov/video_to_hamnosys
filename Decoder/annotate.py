
import argparse
import vector2sigml.v2s
from ete3 import Tree
import joblib
import h5py
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('model_path', metavar='MODEL_PATH', type=str, nargs=1, help='path to trained model (ex : 3_nn_multi_train.joblib)')
parser.add_argument('real_data_path', metavar='REAL_DATA_PATH', type=str, nargs=1, help='path to HDF5 file with real data(ex : sl_corpus.hdf5)')
args = parser.parse_args()
model_path = args.model_path[0]
data_path = args.real_data_path[0]


def annotate(tree_path, data_path):
    print('Making annotations..')
    tree = joblib.load(path+tree_path)
    data_file = h5py.File(path+data_path, 'r')
    for vec in range(len(data_file['X_data'])):
        test_vec  = np.array(get_test_np_vector(tree,torch.from_numpy(data_file['X_data'][vec]).float(),False))
        test_v = vector2sigml.v2s.Vec2sigml(test_vec)
        test_v.save_sigml('./output_greek/pred_saved_all_nn_tree_'+str(vec)+'.txt',str(vec))
    return

annotate(model_path,data_path)