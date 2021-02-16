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

import sys
import pprint
import os
from ete3 import Tree
import re, random
import argparse

parser = argparse.ArgumentParser(prog='SiGML generation Tree builder',
description='This program builds generation Tree from the Rules in "Tree_Rules" folder',
epilog="Please read README.txt file for further descriptions",
add_help=True)

parser.add_argument('-v','--version', action='version', version='%(prog)s 0.1')

def load_saved_nodes(pathtofolder):
        nodes = []
        files = os.listdir(pathtofolder)
        for f in files:
                try:
                        node = Tree(pathtofolder+"/"+f, format=1)
                        node.name = f[:-3]
                        nodes.append(node)
                except:
                        print(str(f))
        #for n in nodes:
                #print(n.get_ascii(show_internal=True))
        return nodes

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

ham_symbols = get_symbols("./HamSymbols.txt")

def write_to_file(file, line):
    f= open(file,'a')
    f.write(line+'\n')
    f.close()
    return

def add_children_over_nodes(run,filepath):
        t = Tree(filepath, format=1)
        init = False
        
        nod = 0
        leafs = [i for i in t.iter_leaves() if i.name.lower() not in ham_symbols]
        leafs_len = len(leafs)
        for n in nodes:
                nod +=1
                leevv = 0
                for leaf in leafs:
                        leevv += 1
                        print("                                                                                                                                            ", end='\r')
                        print('Leaf '+leaf.name+' Leaf '+str(leevv)+'/'+str(leafs_len)+' Node '+n.name + ' Nodes run: ' + str(nod) + ' Tree Run: '+ str(run) , end='\r')
                        if leaf.name == n.name:
                                init= True
                                for c in n.children:
                                        leaf.add_child(c)
                                #print("                                                                                          ", end='\r')
                                print('Leaf '+leaf.name+' Leaf '+str(leevv)+'/'+str(leafs_len)+' Node '+n.name + ' Nodes run: ' + str(nod) + ' Tree Run: '+ str(run) )

        run += 1
        del(leafs)
        #os.remove("saved_newick_tree.nw")
        os.remove(filepath)
        #t.write(format=1, outfile="saved_newick_tree.nw")
        t.write(format=1, outfile=filepath)
        print("Saved")

        if init == False:
                print("Tree created")
                tree_leaves = len(list([ln for ln in t.iter_leaves()]))
                write_to_file('tree_create_output.txt',filepath+" Number of leaves : "+ str(tree_leaves))
                return 
        else: 
                return add_children_over_nodes(run,filepath)

def create_tree(seed_node_path, tree_name):
        seed_node = Tree(seed_node_path, format=1)
        seed_node.name = seed_node_path[:-3]
        seed_node.write(format=1, outfile=tree_name)

        init_run = 0
        add_children_over_nodes(init_run,tree_name)

        return

nodes = load_saved_nodes("./Tree_Rules/")

#ONE BIG GENERATION TREE 
#create_tree("./Tree_Rules/sign.nw","./Created_Trees/hamnosys_generation_tree.nw")

#ONE HAND
create_tree("./Tree_Rules/hand_symm.nw","./Created_Trees/symm_tree.nw")
create_tree("./Tree_Rules/basichandshape.nw","./Created_Trees/handshape_tree.nw")
create_tree("./Tree_Rules/hand_pos.nw","./Created_Trees/handpos_tree.nw")
create_tree("./Tree_Rules/location1.nw","./Created_Trees/handlocation_tree.nw")
#create_tree("./Tree_Rules/a1tlist.nw","./Created_Trees/action_tree.nw")

#DOUBLE HAND TREES
create_tree("./Tree_Rules/double_hand_symm.nw","./Created_Trees/d_symm_tree.nw")
create_tree("./Tree_Rules/double_handshape.nw","./Created_Trees/d_handshape_tree.nw")
create_tree("./Tree_Rules/double_handposition.nw","./Created_Trees/d_handposition_tree.nw")
create_tree("./Tree_Rules/double_handlocation.nw","./Created_Trees/d_handlocation_tree.nw")
create_tree("./Tree_Rules/double_action.nw","./Created_Trees/d_action_tree.nw")

