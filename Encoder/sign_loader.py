"""
Copyright 2020 Victor Skobov

Email: v.skobov@fuji.waseda.jp
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import argparse
import sys
import random
import re
import os
from lxml import etree
from ete3 import Tree
import pprint
import time
import math
import numpy as np
import glob
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')

args = parser.parse_args()

path = args.output_dest[0]


def get_symbols(file_p):
    s_list = []    
    with open(file_p, mode='r') as file:
        for line in file.readlines():
                line_objects = [splits for splits in line.rstrip('\n').split("\t") if splits is not ""]
                if line_objects[1] != 'UNUSED':
                        if len(line_objects)> 3:
                                if line_objects[3] != '(OBSOLETE)':
                                        s_list.append(line_objects[1])
                        else:
                                s_list.append(line_objects[1])
    return s_list

def vector_to_sign_list(vector,tt):
    global ham_symbols
    tree = tt
    r_list =[]
    #print("Vec l: "+str(len(vector)))
    #print("Tree l: "+str(len(list([i for i in tree.iter_leaves()]))))
    i = 0
    for leaf in tree.iter_leaves():
        if vector[i] == 1:
            if leaf.name.lower() in ham_symbols:
                r_list.append(leaf.name.lower())
        i = i + 1
    return r_list

def save_sigml_(list,path, name_addition,i):
    sigml = etree.Element("sigml")
    hns_sign = etree.Element("hns_sign")
    hns_sign.set("gloss","test_"+name_addition+"_"+str(i))
    hamnosys_nonmanual = etree.SubElement(hns_sign,"hamnosys_nonmanual")
    hamnosys_nonmanual.text = ""
    #sign_length = 0
    hamnosys_manual = etree.SubElement(hns_sign,"hamnosys_manual")
    for sign in list:
        etree.SubElement(hamnosys_manual, sign)
    sigml.append(hns_sign)
    sigml_signs = etree.ElementTree(sigml)
    sigml_signs.write(path+'/'+'output_sign_'+name_addition+"_"+str(i)+'.sigml', pretty_print=True, encoding='utf-8')
    print('Sign '+name_addition+"_"+str(i)+'Saved : '+path+'/'+'output_sign_'+name_addition+"_"+str(i)+'.sigml')
    return
    
ham_symbols = get_symbols("./HamSymbols.txt")
#tree_path = './Created_Trees/handshape_tree.nw'

h_conf_tree_path = './Created_Trees/handshape_tree.nw'
h_or_tree_path = './Created_Trees/handpos_tree.nw'
h_loc_tree_path = './Created_Trees/handlocation_tree.nw'
action_tree_path = './Created_Trees/action_tree.nw'
symm_tree_path = './Created_Trees/symm_tree.nw'
print('Loading trees .. ')
h_conf_tree = Tree(h_conf_tree_path,format=1)
h_or_tree = Tree(h_or_tree_path,format=1)
h_loc_tree = Tree(h_loc_tree_path,format=1)
action_tree = Tree(action_tree_path,format=1)
#symm_tree = Tree(action_tree_path,format=1)

dirs = glob.glob(path+'/*/')
dirs.sort()
for d in dirs:
        p = pathlib.PurePath(d)
        p.parts[-1]
        #print (p)
        print('Processing vectors .. ')
        
        h_conf_symbols = vector_to_sign_list(list(np.load(str(p) +'/' +'input_vector_'+'h_conf_'+str(p.parts[-1])+'.npy')),h_conf_tree)
        h_or_symbols = vector_to_sign_list(list(np.load(str(p) +'/' +'input_vector_'+'h_or_'+str(p.parts[-1])+'.npy')),h_or_tree)
        h_loc_symbols = vector_to_sign_list(list(np.load(str(p) +'/' +'input_vector_'+'h_loc_'+str(p.parts[-1])+'.npy')),h_loc_tree)
        h_action_symbols =vector_to_sign_list(list(np.load(str(p) +'/' +'input_vector_'+'h_action_'+str(p.parts[-1])+'.npy')),action_tree)

        print('Saving Signs ..')

        save_sigml_(h_conf_symbols,str(p),"h_conf",p.parts[-1])
        save_sigml_(h_conf_symbols+h_or_symbols,str(p),"h_conf_or",p.parts[-1])
        save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols,str(p),"h_conf_or_loc",p.parts[-1])
        save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols+h_action_symbols,str(p),"h_conf_or_loc_action",p.parts[-1])



#vector = np.load('./'+args.output_dest+'/'+str(i)+'/' +'input_vector_'+str(i)+'.npy')

#tr = Tree(tree_path,format=1)
#gen_symbols = list([s.name.lower() for s in vector_to_sign_list(vector,tr) if s.name.lower() in ham_symbols])