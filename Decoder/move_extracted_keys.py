#!/usr/local/bin/python3

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

