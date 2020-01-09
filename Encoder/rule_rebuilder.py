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

#nodes = load_saved_nodes("./Tree_Rules_Depricated/")
nodes = load_saved_nodes("./Tree_Rules/")
new_nodes = []

names = []
def name_swap(node):
    #print(node.get_ascii(show_internal=True))
    for l in node.iter_leaves():
        names.append((node.name,l.name))
        #print('Got name : '+ l.name)
    return

def lonely_option(node):
    #print(node.get_ascii(show_internal=True))
    empty_opt = Tree()
    empty_opt.name = 'empty'
    non_opt = Tree()
    non_opt.name = 'NON_OPT'
    
    if len(node.children)>1:
        for c in node.children:
            if c.name[-4:] == '_OPT' and c.name != 'NON_OPT':
                first_opt = c.copy()
                
                ccs = list([cc for cc in c.children])
                for cc in range(0, len(c.children)):
                    c.remove_child(ccs[cc])

                c.name = 'NON_OPT'
                second_opt= Tree()
                second_opt.name = '2_OPT'
                c.add_child(first_opt)
                second_opt.add_child(empty_opt)
                c.add_child(second_opt)
    if len(node.children)==1 and node.children[0].name[-4:] == '_OPT' and node.children[0].name != 'NON_OPT':
        node.children[0].delete()
        
            #node.add_child(non_opt)

    #print(node.get_ascii(show_internal=True))
    return

for n in nodes:
    pr = False
    if len(list(n.iter_leaves())) ==1: 
        #print(" Name Swap for : " + n.name)
        name_swap(n)
    else:
        for t in n.traverse():
            if len(list([c for c in t.children if c.name[-4:] == '_OPT' and c.name != 'NON_OPT'])) == 1:
                #print(" Lonely option for : " + n.name)
                #lonely_option(t)
                pr = True
        if pr:
            #print(n.get_ascii(show_internal=True))
            #n.write(format=1, outfile="./Tree_Rules_nwk/"+n.name+".nwk")
            for t in n.traverse():
                if len(list([c for c in t.children if c.name[-4:] == '_OPT' and c.name != 'NON_OPT'])) == 1:
                    #lonely_option(t)
                    c = 1
                    #print(n.get_ascii(show_internal=True))
            #print(n.get_ascii(show_internal=True))


for i in range(0, len(nodes)):
    for l in nodes[i].iter_leaves():
        for name in names:
            if l.name == name[0]:
                l.name = name[1]

    new_nodes.append(nodes[i])


#print("Double check")
for i in range(0, len(nodes)):
    #print(nodes[i].get_ascii(show_internal=True))
    if len(list(new_nodes[i].iter_leaves()))>1:
        #new_nodes[i].write(format=1, outfile="./Tree_Rules/"+new_nodes[i].name+".nw")
        print(new_nodes[i].get_ascii(show_internal=True))
        c= 1
    #