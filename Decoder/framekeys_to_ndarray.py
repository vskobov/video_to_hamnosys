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


import os
import numpy as np
import sys
import json
import math
import glob
import matplotlib.pyplot as plt
import pathlib
import argparse
# get arguments

parser = argparse.ArgumentParser()
parser.add_argument('output_dest', metavar='OUTPUT_DEST', type=str, nargs=1, help='Output file directory (ex : ./folder/)')

args = parser.parse_args()

path = args.output_dest[0]

#returns Numpy Euclidean distance matrix, values are normalized against base distanse of shoulders. matrix has to be a hollow
def normalize_keys(body_keys=[],hand_keys=[], right_hand = False):
    norm_keys = []
    keys= body_keys+hand_keys
    coords = []
    for i in range(0, int(len(keys)/3)):
        coords.append((keys[i*3],keys[i*3+1]))

    # Used for Hand Location features
    if len(coords) == 25:
        #print('Body_25 model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(coords[2], coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove = [8,9,12,10,11,23,22,24,13,21,19,20,14] # legs

    if len(coords) == 18:
        #print('COCO model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(coords[2], coords[5])]))
        #print('Base distance: '+ str(base_distance))
        if right_hand:
            points_to_remove = [8,11,12,13,9,10,6,7] #leaving only head shoulders and right hand
        else:
            points_to_remove = [8,11,12,13,9,10,3,4] #leaving only head shoulders and left hand

    if len(coords) == 15:
        print('MPI model normilization')
        
    # Used for Hand configuration features
    if len(coords) == 21: 
        #print('HAND model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(coords[0], coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove =[]

    # Used for Palm orientation features
    if len(coords) == 36:
        print('MPI and HAND model normilization')
        
    if len(coords) == 46:
        #print('Body_25 and HAND model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(coords[2], coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove = [8,9,12,10,11,23,22,24,13,21,19,20,14,3,6,4,7]
        hand_points_to_remove = [20,19,16,15,12,11,7,8,4]
        for h_p in hand_points_to_remove:
            points_to_remove.append(h_p+25)
    if len(coords) == 39:
        #print('COCO and HAND model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(coords[2], coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove = [8,11,12,13,9,10,3,6,4,7] #head and legs
        hand_points_to_remove = [20,19,16,15,12,11,7,8,4]
        for h_p in hand_points_to_remove:
            points_to_remove.append(h_p+18)

    vals = []
    for r_p in points_to_remove:
        vals.append(coords[r_p])
    for val in vals:
        coords.remove(val)
    #print('coords lenght: '+str(len(coords)))
    M = []
    dimensions = 1
    #print(len(coords))
    for x in range(0,len(coords)):
        #y_array = []
        for y in range(dimensions,len(coords)):
            if any(coords[y]) == 0 or any(coords[x]) == 0 or base_distance == 0:
                norm_dist = 0
            else:
                dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(coords[y], coords[x])]))
                norm_dist = dist/base_distance
            M.append(norm_dist)
        #M.append(y_array)
        dimensions += 1
    
    np_M = np.array(M)
    return np_M

def normalize_keys_as_one(body_keys,hand_l_keys,hand_r_keys):
    norm_keys = []
    hand_keys = hand_l_keys + hand_r_keys 
    hand_coords = []
    for i in range(0, int(len(hand_keys)/3)):
        hand_coords.append((hand_keys[i*3],hand_keys[i*3+1]))

    body_coords = []
    for i in range(0, int(len(body_keys)/3)):
        body_coords.append((body_keys[i*3],body_keys[i*3+1]))

    if len(body_coords) == 25:
        #print('Body_25 model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(body_coords[2], body_coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove = [8,9,12,10,11,23,22,24,13,21,19,20,14] #legs
        for r_p in points_to_remove:
            body_coords.remove(body_coords[r_p])

    if len(body_coords) == 18:
        #print('COCO model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(body_coords[2], body_coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove = [8,11,12,13,9,10] #legs
        vals = []
        for r_p in points_to_remove:
            vals.append(body_coords[r_p])
        for val in vals:
            body_coords.remove(val)
    #if len(body_coords) == 15:
        #print('MPI model normilization')
    all_coords = body_coords + hand_coords
    #print('coords lenght: '+str(len(all_coords)))
    M = []
    dimensions = 1
    #print(len(all_coords))
    for x in range(0,len(all_coords)):
        #y_array = []
        for y in range(dimensions,len(all_coords)):
            if any(all_coords[y]) == 0 or any(all_coords[x]) == 0:
                norm_dist = 0
            else:
                dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(all_coords[y], all_coords[x])]))
                norm_dist = dist/base_distance
            M.append(norm_dist)
        #M.append(y_array)
        dimensions += 1
    
    np_M = np.array(M)
    #print(np_M.shape)

    #print(np_M)
    return np_M

def coords_normalize_keys_as_one(body_keys,hand_l_keys,hand_r_keys):
    norm_keys = []
    hand_keys = hand_l_keys + hand_r_keys 
    hand_coords = []
    for i in range(0, int(len(hand_keys)/3)):
        hand_coords.append((hand_keys[i*3],hand_keys[i*3+1]))

    body_coords = []
    for i in range(0, int(len(body_keys)/3)):
        body_coords.append((body_keys[i*3],body_keys[i*3+1]))

    if len(body_coords) == 25:
        #print('Body_25 model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(body_coords[2], body_coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove = [8,9,12,10,11,23,22,24,13,21,19,20,14] #legs
        for r_p in points_to_remove:
            body_coords.remove(body_coords[r_p])

    if len(body_coords) == 18:
        #print('COCO model normilization')
        base_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(body_coords[2], body_coords[5])]))
        #print('Base distance: '+ str(base_distance))
        points_to_remove = [8,11,12,13,9,10] #legs
        vals = []
        for r_p in points_to_remove:
            vals.append(body_coords[r_p])
        for val in vals:
            body_coords.remove(val)
    #if len(body_coords) == 15:
        #print('MPI model normilization')
    all_coords = body_coords + hand_coords
    #print('coords lenght: '+str(len(all_coords)))
    M = []
    dimensions = 1
    #print(len(all_coords))
    for x in range(0,len(all_coords)):
        M.append(all_coords[x][0])
        M.append(all_coords[x][1])

    np_M = np.array(M)
    #print(np_M.shape)

    #print(np_M)
    return np_M

def concat_to_one_dim(two_dim_array):
    shape = two_dim_array.shape
    
    one_dim_array = two_dim_array.reshape(shape[0]*shape[1],)
    #print(one_dim_array)
    if two_dim_array.size != one_dim_array.size:
        print("Concatenation error")
    return one_dim_array

def frame_handle(frame_name, path, plot = False, text_out = False):
    frame_keys = glob.glob(path+'keys_'+frame_name+'/*.json')
    #print(path+'input_video_keys_'+frame_name+'/*.json')
    frame_keys.sort()

    all_in_matrix = []

    l_hand_conf_matrix = []
    l_hand_orient_matrix = []
    l_hand_location_matrix = [] 

    r_hand_conf_matrix = []
    r_hand_orient_matrix = []
    r_hand_location_matrix = [] 


    for f in frame_keys:
        with open(f) as json_file:
            data = json.load(json_file)
            all_in_matrix.append(normalize_keys_as_one(data['people'][0]['pose_keypoints_2d'],data['people'][0]['hand_left_keypoints_2d'],data['people'][0]['hand_right_keypoints_2d']))
            
            l_hand_conf_matrix.append(normalize_keys([],data['people'][0]['hand_left_keypoints_2d'])) 
            l_hand_orient_matrix.append(normalize_keys(data['people'][0]['pose_keypoints_2d'],data['people'][0]['hand_left_keypoints_2d'])) 
            l_hand_location_matrix.append(normalize_keys(data['people'][0]['pose_keypoints_2d'],[])) 
            
            r_hand_conf_matrix.append(normalize_keys([],data['people'][0]['hand_right_keypoints_2d'])) 
            r_hand_orient_matrix.append(normalize_keys(data['people'][0]['pose_keypoints_2d'],data['people'][0]['hand_right_keypoints_2d'])) 
            r_hand_location_matrix.append(normalize_keys(data['people'][0]['pose_keypoints_2d'],[],True)) 

    np_movement_matrix = np.array(all_in_matrix)

    np_l_hand_conf_matrix = np.array(l_hand_conf_matrix)
    np_l_hand_orient_matrix = np.array(l_hand_orient_matrix)
    np_l_hand_location_matrix = np.array(l_hand_location_matrix)

    np_r_hand_conf_matrix = np.array(r_hand_conf_matrix)
    np_r_hand_orient_matrix = np.array(r_hand_orient_matrix)
    np_r_hand_location_matrix = np.array(r_hand_location_matrix)

    #print(np_movement_matrix.shape)
    #ADD LINEAR INTERPOLATION
    #np_movement_matrix = fill_zeros_with_avg(np_movement_matrix)

    #np_l_hand_conf_matrix = fill_zeros_with_avg(np_l_hand_conf_matrix)
    #np_l_hand_orient_matrix = fill_zeros_with_avg(np_l_hand_orient_matrix)
    #np_l_hand_location_matrix = fill_zeros_with_avg(np_l_hand_location_matrix)

    #np_r_hand_conf_matrix = fill_zeros_with_avg(np_r_hand_conf_matrix)
    #np_r_hand_orient_matrix = fill_zeros_with_avg(np_r_hand_orient_matrix)
    #np_r_hand_location_matrix = fill_zeros_with_avg(np_r_hand_location_matrix)

    def plot_the_move(mat, frames, lenght, sign_name):
        for i in range (0,lenght):
            x = []
            for m in range(0,len(mat)):
                x.append(mat[m][i])
            plt.plot([i for i in range (0,frames)],x, label=str(i))

        plt.title('Movement '+sign_name)
        plt.xlabel('Frames')
        plt.ylabel('Head-to-Point Distances')
        plt.savefig(path+'Movement_plot_'+sign_name+'.png')
        plt.clf()
        
        return
    
    if plot:
        plot_the_move(np_movement_matrix,len(frame_keys),50, frame_name+'_all' )

        plot_the_move(np_l_hand_conf_matrix,len(frame_keys),21, frame_name+'_l_h_conf')
        plot_the_move(np_l_hand_orient_matrix,len(frame_keys),9, frame_name+'_l_h_or')
        plot_the_move(np_l_hand_location_matrix,len(frame_keys),6, frame_name+'_l_h_loc')

        plot_the_move(np_r_hand_conf_matrix,len(frame_keys),21, frame_name+'_r_h_conf')
        plot_the_move(np_r_hand_orient_matrix,len(frame_keys),9, frame_name+'_r_h_or')
        plot_the_move(np_r_hand_location_matrix,len(frame_keys),6, frame_name+'_r_h_loc')
    
    if text_out:
        text_file = open(path+"np_movement_matrix_"+frame_name+".txt", "w")
        text_file.write(np.array2string(np_movement_matrix, threshold=sys.maxsize, max_line_width=sys.maxsize))
        text_file.close()

        text_file = open(path+"np_l_hand_conf_matrix_"+frame_name+".txt", "w")
        text_file.write(np.array2string(np_l_hand_conf_matrix, threshold=sys.maxsize, max_line_width=sys.maxsize))
        text_file.close()
        text_file = open(path+"np_l_hand_orient_matrix_"+frame_name+".txt", "w")
        text_file.write(np.array2string(np_l_hand_orient_matrix, threshold=sys.maxsize, max_line_width=sys.maxsize))
        text_file.close()
        text_file = open(path+"np_l_hand_location_matrix_"+frame_name+".txt", "w")
        text_file.write(np.array2string(np_l_hand_location_matrix, threshold=sys.maxsize, max_line_width=sys.maxsize))
        text_file.close()

        text_file = open(path+"np_r_hand_conf_matrix_"+frame_name+".txt", "w")
        text_file.write(np.array2string(np_r_hand_conf_matrix, threshold=sys.maxsize, max_line_width=sys.maxsize))
        text_file.close()
        text_file = open(path+"np_r_hand_orient_matrix_"+frame_name+".txt", "w")
        text_file.write(np.array2string(np_r_hand_orient_matrix, threshold=sys.maxsize, max_line_width=sys.maxsize))
        text_file.close()
        text_file = open(path+"np_r_hand_location_matrix_"+frame_name+".txt", "w")
        text_file.write(np.array2string(np_r_hand_location_matrix, threshold=sys.maxsize, max_line_width=sys.maxsize))
        text_file.close()

    #np.save(path+'input_keys_2d_np_all_'+frame_name, np_movement_matrix)

    #np.save(path+'input_keys_2d_np_l_hand_conf_'+frame_name, np_l_hand_conf_matrix)
    #np.save(path+'input_keys_2d_np_l_hand_orient_'+frame_name, np_l_hand_orient_matrix)
    #np.save(path+'input_keys_2d_np_l_hand_location_'+frame_name, np_l_hand_location_matrix)

    #np.save(path+'input_keys_2d_np_r_hand_conf_'+frame_name, np_r_hand_conf_matrix)
    #np.save(path+'input_keys_2d_np_r_hand_orient_'+frame_name, np_r_hand_orient_matrix)
   # np.save(path+'input_keys_2d_np_r_hand_location_'+frame_name, np_r_hand_location_matrix)

    np.save(path+'input_keys_1d_np_all_'+frame_name, concat_to_one_dim(np_movement_matrix))

#    np.save(path+'input_keys_1d_np_l_hand_conf_'+frame_name, concat_to_one_dim(np_l_hand_conf_matrix))
#    np.save(path+'input_keys_1d_np_l_hand_orient_'+frame_name, concat_to_one_dim(np_l_hand_orient_matrix))
#    np.save(path+'input_keys_1d_np_l_hand_location_'+frame_name, concat_to_one_dim(np_l_hand_location_matrix))

#    np.save(path+'input_keys_1d_np_r_hand_conf_'+frame_name,concat_to_one_dim(np_r_hand_conf_matrix))
#    np.save(path+'input_keys_1d_np_r_hand_orient_'+frame_name, concat_to_one_dim(np_r_hand_orient_matrix))
#    np.save(path+'input_keys_1d_np_r_hand_location_'+frame_name, concat_to_one_dim(np_r_hand_location_matrix))

    np.save(path+'input_keys_1d_np_lr_hand_conf_'+frame_name, np.concatenate((concat_to_one_dim(np_l_hand_conf_matrix), concat_to_one_dim(np_r_hand_conf_matrix)), axis=0) )
    np.save(path+'input_keys_1d_np_lr_hand_orient_'+frame_name, np.concatenate((concat_to_one_dim(np_l_hand_orient_matrix), concat_to_one_dim(np_r_hand_orient_matrix)), axis=0) )
    np.save(path+'input_keys_1d_np_lr_hand_location_'+frame_name, np.concatenate((concat_to_one_dim(np_l_hand_location_matrix), concat_to_one_dim(np_r_hand_location_matrix)), axis=0) )


    
    

    return

def fill_zeros_with_avg(martix):
    t_m = martix.T
    for col in t_m:
        zeros = np.flatnonzero(col == 0)
        if len(zeros)>0:
            ok = np.flatnonzero(col > 0)
            fp = col[ok]
            xp = np.where(col > 0)[0]
            x  = zeros
            #print(fp)
            #print(xp)
            #print(x)
            for zero in zeros:
                col[zero] = np.interp(zero, xp, fp)
            #print(col)
    filled_matrix = t_m.T
    return filled_matrix



dirs = glob.glob(path+'/*/')
dirs.sort()
lenght = len(dirs)
s = 0 
for d in dirs:
    p = pathlib.PurePath(d)
    try:
        frame_handle(p.parts[-1],d,False,False)
    except:
        print('Missed '+ d)
    print(str(int(s/lenght*100))+' %', end = '\r')
    s += 1 





