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
from ete3 import Tree
import random
import re
import os
from lxml import etree
import pprint
import matplotlib.pyplot as plt
import time
import math

# get arguments

parser = argparse.ArgumentParser(prog='SiGML Sign Generator',
description='This programm computes generates valid HamNoSys 4.0 Signs and outputs them in SiGML files',
epilog="Please contact me via v.skobov@fuji.waseda.jp for more information",
add_help=True)

parser.add_argument('number_of_signs', metavar='SIGNS_NUM', type=str, help='Number of Signs to generate')
parser.add_argument('signs_in_the_file', metavar='SIGNS_BUNCH', type=str, help='Amount of signs stored in one SiGML file', default='100')
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs='?',default='./SiGML_Fast_output/', help='Output file directory (ex : ./folder/)')
parser.add_argument('-v','--version', action='version', version='%(prog)s 0.1')
parser.add_argument('-p','--plot', dest='plot', action='store_true',
                    default=False,
                    help='Use this option to output a discriptive statistics plot')
parser.add_argument('-s','--symbols', dest='symbols', action='store_true',
                    default=False,
                    help='Use this option to print symbols usage and mentions')

args = parser.parse_args()
plot = args.plot
symbols = args.symbols

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


def weighted_walk(tree):

    if tree.is_leaf():
        l_counter(tree)
        tree.name = tree.name +'*'
        
        return
    else:
        choicer = False

        for o in tree.children:
            if o.name.find('OPT') != -1:
            #if o.name[-4:] == "_OPT" or o.name == "NON_OPT":
                choicer = True

        if choicer:
            #print("Choicer :"+tree.name+" Children: " + str(list([o.name for o in tree.children])))
            options = []
            weights = []
            random_choice = Tree()
            for o in tree.children:
                if o.name.find('OPT') != -1 and o.name != "NON_OPT":
                        #print(o.name)
                        options.append(o)
            if len(options)>0:
                for o in options:
                    #weights.append(float(o.name[o.name.find('OPT')+3:]))
                    weights.append(float(o.dist))

                random_choice = random.choices(population = options,weights = weights,k=1)[0]
                #print(random_choice.name)

            del(options)
            del(weights)
            for child in tree.children:
                if child == random_choice or child.name == "NON_OPT":
                    #print("Pass further from "+tree.name+" to :" + child.name)
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
            #if o.name[-4:] == "_OPT" or o.name == "NON_OPT":
                choicer = True

        if choicer:
            #print("Choicer :"+tree.name+" Children: " + str(list([o.name for o in tree.children])))
            options = []
            weights = []
            leaves = []
            for o in tree.children:
                if o.name[-4:] == "_OPT" and o.name != "NON_OPT":
                        #print(o.name)
                        options.append(o)
            
            if len(options)>0:
                for o in options: 
                    leaves.append(len(set(list([get_option_parent(l) for l in o.iter_leaves()]))))
                #print(leaves)
            
                sum_leaves = len(set(list([get_option_parent(l) for l in tree.iter_leaves()])))

                for wl in range(0,len(leaves)):
                    weights.append(float(leaves[wl]/sum_leaves))

                for o in options:
                    #o.name = o.name + str(weights[options.index(o)])
                    o.dist = weights[options.index(o)]
                #print(random_choice.name)
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
    #print("finding lists")
    #print(tree.get_ascii(show_internal=True))

    for leaf in tree.iter_leaves():
        if leaf.is_leaf() and leaf.name[-1]=='*':
            leaf.name = leaf.name.rstrip('*')
            r_list.append(leaf)
            #print(leaf.name)
            
            #print(leaf.name)
    return r_list

def clean_upper(l):
    clean = []
    for el in l:
        if el.name.isupper():
            #print('Upper '+el)
            l.remove(el)
        else: 
            #print('Lower '+el)
            clean.append(el)
    return clean

def big_direct_way(tree):
    symbols = []
    symbols = one_big_tree(tree)
    signs = []

    for n in symbols:
        for s in n:
            signs.append(s)
    #print(signs)
    del(symbols)
    return signs

def one_big_tree(tree):
    one_big_tree_list = []
    one_big_tree_list = get_equal_list(tree)
    return one_big_tree_list


ham_symbols = get_symbols("../HamSymbols.txt")
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
    print('Loading the Generation AST ...' )
    tr = Tree("./Created_Trees/hamnosys_generation_tree.nw",format=1)
    #tr = Tree("./Single_trees/symm_tree.nw",format=1)
    print('Generation AST loaded' )
    print("Precomputing the weights ... ")
    precomputation_eqalizer(tr)
    print("Precomputation done")
    #tr.write(format=1, outfile="./Single_trees/preprocessed_equal_tree.nw")
    tree_leaves = list([ln for ln in tr.iter_leaves()])
    print("Options avalable : "+ str(len(tree_leaves)))
    #print(type(tree_leaves[0]))
    leaves_counter = [0 for x in range(0,len(tree_leaves))]
    
    print('Starting the generation of '+str(count)+' random signs' )
    start_time  = time.time() 
    hundred = 0
    sigml = etree.Element("sigml")
    for i in range(0,count):
        print("                                                                                                                                            ", end='\r')
        print('Signs done '+str(i)+'/'+str(count), end='\r')
        
        hns_sign = etree.Element("hns_sign")
        hns_sign.set("gloss","test_"+str(hundred))
        hamnosys_nonmanual = etree.SubElement(hns_sign,"hamnosys_nonmanual")
        hamnosys_nonmanual.text = ""
        sign_length = 0
        hamnosys_manual = etree.SubElement(hns_sign,"hamnosys_manual")

        gen_symbols = list([s.name.lower() for s in big_direct_way(tr) if s.name.lower() in ham_symbols])

        for sign in gen_symbols:
            sign_length += 1
            etree.SubElement(hamnosys_manual, sign)
        len_list.append(sign_length)
        del(gen_symbols)
        del(sign_length)
        sigml.append(hns_sign)
        if hundred == bunch -1 or i == (count-1): 
            hundred = 0
            sigml_signs = etree.ElementTree(sigml)
            sigml_signs.write(args.output_dest +'fast_equal_'+str(math.floor(i/bunch))+'.sigml', pretty_print=True, encoding='utf-8')
            #print('Saved to '+args.output_dest +'fast_equal_'+str(math.floor(i/100))+'.sigml')
            del(sigml)
            sigml = etree.Element("sigml")
        else:
            hundred += 1

    end_time = time.time()
    print('Done generating of '+str(count)+' Signs in '+ str(round((end_time-start_time),2))+' seconds')
    return
#os.remove("output.sigml")

def plot_it(name='fast_equal_len.png'):
    bar = plt.subplot(121)
    bar.set_title('Tree leaves usage')
    bar.scatter([i for i in range(0,len(tree_leaves))], leaves_counter, marker='o')
    length = plt.subplot(122)
    length.set_title('Single Sign Length Distrubution')
    length.bar( ['min','average','max'] ,[min(len_list),(sum(len_list)/len(len_list)),max(len_list)])
    plt.savefig(name)


#pp = pprint.PrettyPrinter(indent=4)
#tr = Tree("./Single_trees/two_hands_tree.nw",format=1)
#m = tr.search_nodes(name="")
#pp.pprint(list([list([a.name for a in n.get_ancestors()]) for n in m]))


sign_to_generate = int(args.number_of_signs)
bunch = int(args.signs_in_the_file)


generate_sigml(sign_to_generate,bunch)

if plot:
    plot_it(args.output_dest +'fast_equal_len.png')

if symbols:
    print('Symbols not mentioned :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] == 0]))
    print('Popular symbols :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] > sign_to_generate*0.7]))

