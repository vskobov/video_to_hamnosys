#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
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



class Vec2sigml:
    def __init__(self, vector):
        self.vector = vector
        self.h_conf_tree_path = str(pathlib.Path(__file__).parent)+'/Created_Trees/handshape_tree.nw'
        self.h_or_tree_path = str(pathlib.Path(__file__).parent)+'/Created_Trees/handpos_tree.nw'
        self.h_loc_tree_path = str(pathlib.Path(__file__).parent)+'/Created_Trees/handlocation_tree.nw'

    def save_sigml(self, path, name_add):
        #print('sigml2vec: saving...  to :' + path)
        def vector_to_sign_list(self,vector):
            def get_symbols(self):
                s_list = []    
                with open(str(pathlib.Path(__file__).parent)+"/HamSymbols.txt", mode='r') as file:
                    for line in file.readlines():
                            line_objects = [splits for splits in line.rstrip('\n').split("\t") if splits is not ""]
                            if line_objects[1] != 'UNUSED':
                                    if len(line_objects)> 3:
                                            if line_objects[3] != '(OBSOLETE)':
                                                    s_list.append(line_objects[1])
                                    else:
                                            s_list.append(line_objects[1])
                return s_list
            
            r_list =[]
            trees = []
            ham_symbols = get_symbols(self)
            #h_conf_tree_path = str(pathlib.Path(__file__).parent)+'/Created_Trees/handshape_tree.nw'
            #h_or_tree_path = str(pathlib.Path(__file__).parent)+'/Created_Trees/handpos_tree.nw'
            #h_loc_tree_path = str(pathlib.Path(__file__).parent)+'/Created_Trees/handlocation_tree.nw'
            #action_tree_path = './Created_Trees/action_tree.nw'
            #symm_tree_path = './Created_Trees/symm_tree.nw'

            if len(vector) == 6537:
                trees.append(Tree(self.h_conf_tree_path,format=1))
                trees.append(Tree(self.h_or_tree_path,format=1))
                trees.append(Tree(self.h_loc_tree_path,format=1))
                print('sigml2vec: trees are loaded')
                #action_tree = Tree(action_tree_path,format=1)
                i = 0
                for tree in trees:
                    for leaf in tree.iter_leaves():
                        if vector[i] == 1:
                            if leaf.name.lower() in ham_symbols:
                                r_list.append(leaf.name.lower())
                        i = i + 1
                    
            if len(vector) == 5763:
                trees.append(Tree(self.h_conf_tree_path,format=1))
                print('sigml2vec: trees are loaded')
                #action_tree = Tree(action_tree_path,format=1)
                for tree in trees:
                    i = 0
                    for leaf in tree.iter_leaves():
                        if vector[i] == 1:
                            if leaf.name.lower() in ham_symbols:
                                r_list.append(leaf.name.lower())
                        i = i + 1
            if len(vector) == 92 : #92
                trees.append(Tree(self.h_or_tree_path,format=1))
                print('sigml2vec: trees are loaded')
                #action_tree = Tree(action_tree_path,format=1)
                for tree in trees:
                    i = 0
                    for leaf in tree.iter_leaves():
                        if vector[i] == 1:
                            if leaf.name.lower() in ham_symbols:
                                r_list.append(leaf.name.lower())
                                #print('Added' + str(i))
                        i = i + 1
            if len(vector) == 682:
                trees.append(Tree(self.h_loc_tree_path,format=1))
                print('sigml2vec: trees are loaded')
                #action_tree = Tree(action_tree_path,format=1)
                for tree in trees:
                    i = 0
                    for leaf in tree.iter_leaves():
                        if vector[i] == 1:
                            if leaf.name.lower() in ham_symbols:
                                r_list.append(leaf.name.lower())
                        i = i + 1
            #print("Vec l: "+str(len(vector)))
            #print("Tree l: "+str(len(list([i for i in tree.iter_leaves()]))))
            print('sigml2vec: symbols are loaded')
            return r_list

        s_list = vector_to_sign_list(self,self.vector)
        sigml = etree.Element("sigml")
        hns_sign = etree.Element("hns_sign")
        hns_sign.set("gloss","from_vec_"+name_add)
        hamnosys_nonmanual = etree.SubElement(hns_sign,"hamnosys_nonmanual")
        hamnosys_nonmanual.text = ""
        #sign_length = 0
        hamnosys_manual = etree.SubElement(hns_sign,"hamnosys_manual")
        for sign in s_list:
            etree.SubElement(hamnosys_manual, sign)
        sigml.append(hns_sign)
        sigml_signs = etree.ElementTree(sigml)
        #sigml_signs.write(path+'/'+'output_sign_'+name_addition+"_"+str(i)+'.sigml', pretty_print=True, encoding='utf-8')
        sigml_signs.write(path, pretty_print=True, encoding='utf-8')
        print('sigml2vec: sign saved to : '+path)

        #print('Sign '+name_addition+"_"+str(i)+'Saved : '+path+'/'+'output_sign_'+name_addition+"_"+str(i)+'.sigml')
        return