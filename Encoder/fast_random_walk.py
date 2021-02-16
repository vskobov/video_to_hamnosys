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

# get arguments

parser = argparse.ArgumentParser(prog='SiGML Sign Generator',
description='This programm computes generates valid HamNoSys 4.0 Signs and outputs them in SiGML file',
epilog="Please contatcact via v.skobov@fuji.waseda.jp for more information",
add_help=True)

parser.add_argument('number_of_signs', metavar='SIGNS_NUM', type=str, help='Number of Signs to generate')
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

def random_tree_detach(tree):
    #print("Start")
    #print([n.name for n in tree.children])
    if tree.is_leaf():
        #print("Return :"+tree.name)
        l_counter(tree)
        tree.name = tree.name +'*'
        #print("Return :"+tree.name)
        return

    choicer = False

    for o in tree.children:
        if o.name.find('OPT') != -1:
        #if o.name[-4:] == "_OPT" or o.name == "NON_OPT":
            choicer = True
        

    if choicer:
        #print("Choicer :"+tree.name+" Children: " + str(list([o.name for o in tree.children])))
        options = []
        random_choice = Tree()
        for o in tree.children:
            if o.name[-4:] == "_OPT" and o.name != "NON_OPT":
                    #print(o.name)
                    options.append(o)


        if len(options)>0:
            random_choice = next(randomized(options))

        for child in tree.children:
            if child == random_choice or child.name == "NON_OPT":
                #print("Pass further from "+tree.name+" to :" + child.name)
                random_tree_detach(child)

        
    else:
        #print("NOT a Choicer :"+tree.name+" Children: " + str(list([o.name for o in tree.children])))
        for child in tree.children:
                    #print(child.name)
            #print("Pass further from "+tree.name+" to :" + child.name)
            random_tree_detach(child)

                


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

def get_random_list(tt):
    tree = tt
    random_tree_detach(tree)
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



pruned_out = []

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
    one_big_tree_list = get_random_list(tree)
    return one_big_tree_list


ham_symbols = get_symbols("../HamSymbols.txt")
symbol_counter = [0 for x in range(0,len(ham_symbols))]
len_list = []


def counter(sign_symbol):
    symbol_counter[ham_symbols.index(sign_symbol)] += 1
    return 

tree_leaves = []
leaves_counter = []

def l_counter(leaf):
    global tree_leaves
    global leaves_counter
    leaves_counter[tree_leaves.index(leaf)] += 1
    return 

def sigml(count):
    global tree_leaves
    global leaves_counter
    sigml = etree.Element("sigml")
    print('Starting the generation of '+str(count)+' random signs' )
    tr = Tree("./Created_Trees/hamnosys_generation_tree.nw",format=1)
    #tr = Tree("./Single_trees/symm_tree.nw",format=1)
    tree_leaves = list([ln for ln in tr.iter_leaves()])
    #print(type(tree_leaves[0]))
    leaves_counter = [0 for x in range(0,len(tree_leaves))]
    print('Generation AST loaded' )
    start_time  = time.time()  
    for i in range(0,count):
        print("                                                                                                                                            ", end='\r')
        print('Signs '+str(i)+'/'+str(count), end='\r')
        hns_sign = etree.SubElement(sigml, "hns_sign")
        hns_sign.set("gloss","test_"+str(i))
        hamnosys_nonmanual = etree.SubElement(hns_sign,"hamnosys_nonmanual")
        hamnosys_nonmanual.text = ""
        sign_length = 0
        hamnosys_manual = etree.SubElement(hns_sign,"hamnosys_manual")

        gen_symbols = list([i.name.lower() for i in big_direct_way(tr) if i.name.lower() in ham_symbols])

        for sign in gen_symbols:
            sign_length += 1
            counter(sign)
            etree.SubElement(hamnosys_manual, sign)
        len_list.append(sign_length)
        del(gen_symbols)
        del(sign_length)
        tree = etree.ElementTree(sigml)
    end_time = time.time()
    tree.write(args.output_dest, pretty_print=True, encoding='utf-8')
    print('Saved to '+args.output_dest)
    print('Done generating of '+str(count)+' Signs in '+ str(round((end_time-start_time),2))+' seconds')
    return
#os.remove("output.sigml")


def save_sigml(sing_symbols):
    #sigml = etree.Element("sigml")
    
        #hns_sign = etree.SubElement(sigml, "hns_sign")
        #hns_sign.set("gloss","test_"+str(i))
        #hamnosys_nonmanual = etree.SubElement(hns_sign,"hamnosys_nonmanual")
        #hamnosys_nonmanual.text = ""
        #hamnosys_manual = etree.SubElement(hns_sign,"hamnosys_manual")
            #etree.SubElement(hamnosys_manual, sign)
                    #tree = etree.ElementTree(sigml)
    #tree.write("output.sigml", pretty_print=True, encoding='utf-8')
    return

sign_to_generate = int(args.number_of_signs)

#pp = pprint.PrettyPrinter(indent=4)
#tr = Tree("./Single_trees/two_hands_tree.nw",format=1)
#m = tr.search_nodes(name="")
#pp.pprint(list([list([a.name for a in n.get_ancestors()]) for n in m]))

sigml(sign_to_generate)
if plot:
    bar = plt.subplot(121)
    bar.set_title('Tree leaves usage')
    bar.scatter([i for i in range(0,len(tree_leaves))], leaves_counter, marker='o')

    length = plt.subplot(122)
    length.set_title('Single Sign Length Distrubution')
    length.bar( ['min','average','max'] ,[min(len_list),(sum(len_list)/len(len_list)),max(len_list)])

    plt.savefig('fast_len.png')

if symbols:
    print('Symbols not mentioned :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] == 0]))
    print('Popular symbols :' + str([ham_symbols[i] for i in range(0,len(symbol_counter)) if symbol_counter[i] > sign_to_generate*0.7]))

