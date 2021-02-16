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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser(prog='SiGML Sign Generator',
description='This program computes generates valid HamNoSys 4.0 Signs and outputs them in SiGML files',
epilog="Please read README.txt file for further descriptions",
add_help=True)

parser.add_argument('number_of_signs', metavar='SIGNS_NUM', type=str, help='Number of Signs to generate')
parser.add_argument('signs_in_the_file', metavar='SIGNS_BUNCH', type=str, help='Amount of signs stored in one SiGML file', default='100')
parser.add_argument('-wrs', dest='weighted_random_selection', action='store_true',
                    default=False,
                    help='Use this option to apply Weighted Random Selection of optional nodes during the generation')
parser.add_argument('-rs', dest='random_selection', action='store_true',
                    default=False,
                    help='Use this option to apply Random Selection of optional nodes during the generation, it is set as default method')
parser.add_argument('-mix', dest='mixed_random_selection', action='store_true',
                    default=False,
                    help='Use this option to apply mix of Weighted Random Selection and Random Selection of optional nodes during the generation')
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs='?',default='SiGML_Output', help='Output file directory (ex : ./folder/)')
parser.add_argument('-v','--version', action='version', version='%(prog)s 0.1')
parser.add_argument('-p','--plot', dest='plot', action='store_true',
                    default=False,
                    help='Use this option to output a descriptive statistics plot')
parser.add_argument('-s','--symbols', dest='symbols', action='store_true',
                    default=False,
                    help='Use this option to print symbols usage and mentions')

args = parser.parse_args()
plot = args.plot
symbols = args.symbols

WRS = args.weighted_random_selection
RS = args.random_selection
mix = args.mixed_random_selection

method = 0

if WRS == True and mix == False and RS == False:
    method = 2
    print("Weighted Random Selelection is chosen") 
if WRS == False and mix == True and RS == False:
    method = 3
    print("Mixed Selelection RWS & RS is chosen") 
if WRS == False and mix == False and RS == True:
    method = 1
    print("Random Selelection is chosen") 
if WRS == False and mix == False and RS == False:
        method = 1
        print("Random Selelection is chosen as default") 

if method == 0:
    print("Please use only one method at once (-rs) or (-wrs) or (-mix) ")
    sys.exit()





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

def randomized(list):
    ran_nums = sorted(list, key=lambda *args: random.random())
    for num in ran_nums:
         yield num

def random_tree_detach(tree):
    if tree.is_leaf():
        #l_counter(tree)
        tree.name = tree.name +'*'
        return

    choicer = False

    for o in tree.children:
        if o.name.find('OPT') != -1:
            choicer = True

    if choicer:
        options = []
        random_choice = Tree()
        for o in tree.children:
            if o.name[-4:] == "_OPT" and o.name != "NON_OPT":
                    options.append(o)
        if len(options)>0:
            random_choice = next(randomized(options))

        for child in tree.children:
            if child == random_choice or child.name == "NON_OPT":
                random_tree_detach(child)
    else:
        for child in tree.children:
            random_tree_detach(child)

def weighted_walk(tree):

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
            random_choice = Tree()
            for o in tree.children:
                if o.name.find('OPT') != -1 and o.name != "NON_OPT":
                        options.append(o)
            if len(options)>0:
                for o in options:
                    weights.append(float(o.dist))
                random_choice = random.choices(population = options,weights = weights,k=1)[0]
            del(options)
            del(weights)
            for child in tree.children:
                if child == random_choice or child.name == "NON_OPT":
                    weighted_walk(child)
        else:
            for child in tree.children:
                
                weighted_walk(child)

def precomputation_eqalizer(tree):

    if tree.is_leaf():
        return
    else:
        choicer = False

        for o in tree.children:
            if o.name.find('OPT') != -1:
                choicer = True

        if choicer:
            options = []
            weights = []
            leaves = []
            for o in tree.children:
                if o.name[-4:] == "_OPT" and o.name != "NON_OPT":
                        options.append(o)   
            if len(options)>0:
                for o in options: 
                    leaves.append(len(set(list([get_option_parent(l) for l in o.iter_leaves()]))))
                sum_leaves = len(set(list([get_option_parent(l) for l in tree.iter_leaves()])))
                for wl in range(0,len(leaves)):
                    weights.append(float(leaves[wl]/sum_leaves))
                for o in options:
                    o.dist = weights[options.index(o)]
            del(leaves)
            del(options)
            del(weights)
            for child in tree.children:
                    precomputation_eqalizer(child)
        else:
            for child in tree.children:
                precomputation_eqalizer(child)

def get_option_parent(l):
    if l.name[-4:] == "_OPT" and l != "NON_OPT":
        return l
    else:
        if l.up.name[-4:] == "_OPT" and l.up.name != "NON_OPT":
            return l.up
        else: 
            return l.up.up

def get_equal_list(tt):
    tree = tt
    weighted_walk(tree)
    r_list =[]
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_list.append(leaf)
    return r_list

def get_equal_np_vector(tt):
    tree = tt
    tree_leaves = list([ln for ln in tree.iter_leaves()])
    weighted_walk(tree)
    r_vector =[0 for i in range(0,len(tree_leaves))]
    i = 0
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_vector[i] = 1
        i = i + 1
    return r_vector

def get_random_np_vector(tt):
    tree = tt
    tree_leaves = list([ln for ln in tree.iter_leaves()])
    random_tree_detach(tree)
    r_vector =[0 for i in range(0,len(tree_leaves))]
    i = 0
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_vector[i] = 1
        i = i + 1
    return r_vector

def vector_to_sign_list(vector,tt):
    tree = tt
    r_list =[]
    i = 0
    for leaf in tree.iter_leaves():
        if vector[i] == 1:
            r_list.append(leaf)
        i = i + 1
    return r_list

def get_random_list(tt):
    tree = tt
    random_tree_detach(tree)
    r_list =[]
    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_list.append(leaf)
    return r_list

def big_direct_wrs_way(tree,name_addition,i):
    symbols = []
    symbols = one_big_wrs_tree(tree,name_addition,i)
    signs = []

    for n in symbols:
        for s in n:
            signs.append(s)
    #print(signs)
    del(symbols)
    return signs



def big_direct_rs_way(tree,name_addition,i):
    symbols = []
    symbols = one_big_rs_tree(tree,name_addition,i)
    signs = []

    for n in symbols:
        for s in n:
            signs.append(s)
    #print(signs)
    del(symbols)
    return signs
def one_big_wrs_tree(tree,name_addition,i):
    one_big_tree_list = []
    #one_big_tree_list = get_equal_list(tree)
    vector = get_equal_np_vector(tree)

    np_vector = np.array(vector)
    np.save('./'+args.output_dest+'/'+str(i)+'/' +'input_vector_'+name_addition+"_"+str(i)+'.npy',np_vector)

    one_big_tree_list = vector_to_sign_list(vector,tree)

    return one_big_tree_list
def one_big_rs_tree(tree,name_addition,i):
    one_big_tree_list = []
    one_big_tree_list = get_random_list(tree)
    vector = get_random_np_vector(tree)

    np_vector = np.array(vector)
    np.save('./'+args.output_dest+'/'+str(i)+'/' +'input_vector_'+name_addition+"_"+str(i)+'.npy',np_vector)

    one_big_tree_list = vector_to_sign_list(vector,tree)
    return one_big_tree_list

ham_symbols = get_symbols("./HamSymbols.txt")
symbol_counter = [0 for x in range(0,len(ham_symbols))]

len_list = []
tree_leaves = []
leaves_counter = []

def counter(sign_symbol):
    symbol_counter[ham_symbols.index(sign_symbol)] += 1
    return 

def l_counter(leaf):
    global tree_leaves
    global leaves_counter
    leaves_counter[tree_leaves.index(leaf)] += 1
    return 

def generate_sigml(count, bunch):
    global tree_leaves
    global leaves_counter
    global plot

    if not os.path.exists(args.output_dest):
        os.mkdir(args.output_dest)

    #print('Loading the Generation Tree from '+tree_path+' ...' )
    #tr = Tree(tree_path,format=1)
    h_conf_tree_path = './Created_Trees/handshape_tree.nw'
    h_or_tree_path = './Created_Trees/handpos_tree.nw'
    h_loc_tree_path = './Created_Trees/handlocation_tree.nw'
    #action_tree_path = './Created_Trees/action_tree.nw'
    #symm_tree_path = './Created_Trees/symm_tree.nw'
    
    tr_h_conf = Tree(h_conf_tree_path,format=1)
    tr_h_or = Tree(h_or_tree_path,format=1)
    tr_h_loc = Tree(h_loc_tree_path,format=1)
    #tr_h_action = Tree(action_tree_path,format=1)
    #tr_h_symm = Tree(symm_tree_path,format=1)

    print('Generation Tree loaded' )

    if method > 1:
        print("Precomputing the weights ... ")
        #precomputation_eqalizer(tr)
        
        #precomputation_eqalizer(tr_h_symm)
        precomputation_eqalizer(tr_h_conf)
        precomputation_eqalizer(tr_h_or)
        precomputation_eqalizer(tr_h_loc)
        #precomputation_eqalizer(tr_h_action)

        print("Precomputation is done")

    #if you wish to save the tree with precomputated weights
    #tr.write(format=1, outfile="./Created_Trees/preprocessed_equal_tree.nw")
    
    #tree_leaves = list([ln for ln in tr.iter_leaves()])
    #leaves_counter = [0 for x in range(0,len(tree_leaves))]
    
    print('Starting the generation of '+str(count)+' random signs' )
    start_time  = time.time() 
    #hundred = 0
    #sigml = etree.Element("sigml")
    def save_sigml_(list,name_addition, i):
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
        sigml_signs.write('./'+args.output_dest+'/'+str(i)+'/'+'input_sign_'+name_addition+"_"+str(math.floor(i/bunch))+'.sigml', pretty_print=True, encoding='utf-8')
        return

    for i in range(0,count):
        #print("                                                                                                                                            ", end='\r')
        print('Signs done '+str(i)+'/'+str(count), end='\r')
        if not os.path.exists('./'+args.output_dest+'/'+str(i)+'/'):
            os.mkdir('./'+args.output_dest+'/'+str(i)+'/')
        #hns_sign = etree.Element("hns_sign")
        #hns_sign.set("gloss","test_"+str(hundred))
        #hamnosys_nonmanual = etree.SubElement(hns_sign,"hamnosys_nonmanual")
        #hamnosys_nonmanual.text = ""
        #sign_length = 0
        #hamnosys_manual = etree.SubElement(hns_sign,"hamnosys_manual")

        if method == 1:
            #gen_symbols = list([s.name.lower() for s in big_direct_rs_way(tr,i) if s.name.lower() in ham_symbols])

            #symm_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_symm,i) if s.name.lower() in ham_symbols])
            h_conf_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_conf,"h_conf",i,) if s.name.lower() in ham_symbols])
            h_or_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_or,"h_or",i) if s.name.lower() in ham_symbols])
            h_loc_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_loc,"h_loc",i) if s.name.lower() in ham_symbols])
            #h_action_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_action,"h_action",i) if s.name.lower() in ham_symbols])

            #save_sigml_(h_conf_symbols,"h_conf",i)
            #save_sigml_(h_conf_symbols+h_or_symbols,"h_conf_or",i)
            save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols,"h_conf_or_loc",i)
            #save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols+h_action_symbols,"h_conf_or_loc_action",i)



        if method == 2:
            #gen_symbols = list([s.name.lower() for s in big_direct_wrs_way(tr,i) if s.name.lower() in ham_symbols])
            h_conf_symbols = list([s.name.lower() for s in big_direct_wrs_way(tr_h_conf,"h_conf",i) if s.name.lower() in ham_symbols])
            h_or_symbols = list([s.name.lower() for s in big_direct_wrs_way(tr_h_or,"h_or",i) if s.name.lower() in ham_symbols])
            h_loc_symbols = list([s.name.lower() for s in big_direct_wrs_way(tr_h_loc,"h_loc",i) if s.name.lower() in ham_symbols])
            #h_action_symbols = list([s.name.lower() for s in big_direct_wrs_way(tr_h_action,"h_action",i) if s.name.lower() in ham_symbols])

            #save_sigml_(h_conf_symbols,"h_conf",i)
            #save_sigml_(h_conf_symbols+h_or_symbols,"h_conf_or",i)
            save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols,"h_conf_or_loc",i)
            #save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols+h_action_symbols,"h_conf_or_loc_action",i)
        if method == 3:
            if (i % 2) == 0:
                #gen_symbols = list([s.name.lower() for s in big_direct_rs_way(tr,i) if s.name.lower() in ham_symbols])
                h_conf_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_conf,"h_conf",i) if s.name.lower() in ham_symbols])
                h_or_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_or,"h_or",i) if s.name.lower() in ham_symbols])
                h_loc_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_loc,"h_loc",i) if s.name.lower() in ham_symbols])
                #h_action_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_action,"h_action",i) if s.name.lower() in ham_symbols])

                #save_sigml_(h_conf_symbols,"h_conf",i)
                #save_sigml_(h_conf_symbols+h_or_symbols,"h_conf_or",i)
                save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols,"h_conf_or_loc",i)
                #save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols+h_action_symbols,"h_conf_or_loc_action",i)
            else:
                #gen_symbols = list([s.name.lower() for s in big_direct_wrs_way(tr,i) if s.name.lower() in ham_symbols])
                h_conf_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_conf,"h_conf",i) if s.name.lower() in ham_symbols])
                h_or_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_or,"h_or",i) if s.name.lower() in ham_symbols])
                h_loc_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_loc,"h_loc",i) if s.name.lower() in ham_symbols])
                #h_action_symbols = list([s.name.lower() for s in big_direct_rs_way(tr_h_action,"h_action",i) if s.name.lower() in ham_symbols])

                #save_sigml_(h_conf_symbols,"h_conf",i)
                #save_sigml_(h_conf_symbols+h_or_symbols,"h_conf_or",i)
                save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols,"h_conf_or_loc",i)
                #save_sigml_(h_conf_symbols+h_or_symbols+h_loc_symbols+h_action_symbols,"h_conf_or_loc_action",i)

        #for sign in gen_symbols:
        #    sign_length += 1
        #    counter(sign)
        #    etree.SubElement(hamnosys_manual, sign)
        #len_list.append(sign_length)
        #del(gen_symbols)

        del(h_conf_symbols)
        del(h_or_symbols)
        del(h_loc_symbols)
        #del(h_action_symbols)

        #del(sign_length)
        #sigml.append(hns_sign)
        #if hundred == bunch -1 or i == (count-1): 
        #    hundred = 0
        #    sigml_signs = etree.ElementTree(sigml)
        #    sigml_signs.write('./'+args.output_dest+'/'+str(i)+'/'+'input_sign_'+str(math.floor(i/bunch))+'.sigml', pretty_print=True, encoding='utf-8')
            #print('Saved to '+args.output_dest +'signs_'+str(math.floor(i/100))+'.sigml')
        #    del(sigml)
        #    sigml = etree.Element("sigml")
        #else:
        #    hundred += 1

    end_time = time.time()
    print('Done generating of '+str(count)+' Signs in '+ str(round((end_time-start_time),2))+' seconds')
    return

def plot_it(name='plot.png'):
    plt.figure(figsize=(6, 12))
    bar = plt.subplot(211)
    bar.set_title('Tree leaves usage')
    bar.scatter([i for i in range(0,len(tree_leaves))], leaves_counter, marker='+')

    distr_len = plt.subplot(212)
    distr_len.set_title('Sign Length Distrubution')
    sns.distplot(len_list)
    
    plt.savefig(name, dpi=350)

sign_to_generate = int(args.number_of_signs)
bunch = int(args.signs_in_the_file)


generate_sigml(sign_to_generate,bunch)

if symbols:
    f= open('./'+args.output_dest+'/' + 'stats.txt','w+')
    f.write("Options avalable : "+ str(len(tree_leaves))+'\n')
    f.write('Single Sign Length Distrubution \n')
    f.write(str(list([min(len_list),(sum(len_list)/len(len_list)),max(len_list)])))
    f.write('\n')

    f.write('Symbols not mentioned :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] == 0]))
    f.write('\n')
    f.write('Popular symbols :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] > sign_to_generate*0.7]))
    f.write('\n')
    f.write('Percentage of unused leaves: '+ str(round((len(list([i for i in leaves_counter if i == 0]))/len(leaves_counter)*100),2))+ "%")

    f.close()

    counter = open('./'+args.output_dest+'/'+'leaves_couter.txt','w+')
    for l in leaves_counter:
        counter.write(str(l)+'\n')
    counter.close()

    print('Symbols not mentioned :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] == 0]))
    print('Popular symbols :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] > sign_to_generate*0.7]))

if plot:
    plot_it('./'+args.output_dest+'/' +'plot.png')
