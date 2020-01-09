#!/usr/local/bin/python3
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')

args = parser.parse_args()

path = args.output_dest[0]
#out_path="/home/Skobov_Victor/train_set_hand_conf_wrs_50000/"
print(path)
files = glob.glob(path+'/All_Keys/*.json')
files.sort()
length = len(files)
print(length)
s = 0
for f in files:
    f_name = os.path.basename(f)[:os.path.basename(f).find('.')]
    frame_s = f_name.find('_')
    sign = f_name[:frame_s]
    frame = f_name[frame_s+1:]
    if os.path.exists(path+'/'+sign+'/keys_'+sign)==False:
        os.mkdir(path+'/'+sign+'/keys_'+sign)
    os.rename(f,path+'/'+sign+'/keys_'+sign+'/'+frame+'.json')
    s+=1
    print(str(int(s/length)*100)+' %', end = '\r')
    #print('Make Dir: '+ path+'/'+sign+'/keys_'+sign)
    #print('mv '+f+ ' '+path+'/'+sign+'/keys_'+sign+'/'+frame+'.json')

