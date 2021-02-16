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


import random
import os
import zlib
from PIL import Image
import socket, select
from time import gmtime, strftime
from random import randint
import cv2
import numpy as np
import subprocess, signal

HOST = '127.0.0.1'
PORT = 41414

def int_to_bytes(value, length):
    result = []
    for i in range(0, length):
        result.append(value[i] >> (i * 8) & 0xff)
    return int.from_bytes(result, byteorder='little')

connected_clients_sockets = []
check_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
check_socket.bind((HOST,41415))
server_socket.listen(2)
check_socket.listen(2)

def readInt(ins):

    ival = ins[3] << 8;
    ival |= ins[2] & 0xFF;ival <<= 8;
    ival |= ins[1] & 0xFF;ival <<= 8;
    ival |= ins[0] & 0xFF;
    return ival
def read_pixel(ins):
    b= ins[0] & 0xFF
    g = ins[1] & 0xFF
    r =ins[2] & 0xFF
    return (r,g,b)
def readString(data):
    sbuf = ''
    for i in range(5,int(len(data)/4+1)):
        sbuf = sbuf + chr(readInt(d[i*4:(i*4 + 4)]) & 0xFFFF)
        #print('i : '+str(i)+" "+ sbuf)
    #print(sbuf)
    return sbuf

def readpixels(dir_name,frames,pixels,w,h):
    print("Writing pixels...")

    for f in range(0,frames):
        im= Image.new('RGB', (W, H))
        frame_pixels = pixels[f*W*H:f*W*H+W*H]
        #frame_pixels.reverse()
        
        y = h-1
        x = 0

        #print(len(frame_pixels))
        for p in frame_pixels:
            im.putpixel((x,y),p)
            #print(p)
            x = x + 1
            if x == w:
                y = y - 1
                x = 0
                #print((x,y))
        im.save(dir_name+'/'+str(f)+'.png')
    print("Writing pixels DONE")
    return

print('Ready for the next sign')
data = "".encode("utf-8")
ints = []
pixel_ints = [] 
connectionSocket, addr = server_socket.accept()
check_conn_socket, addr = check_socket.accept()
d = []

while True:
    
    #un = connectionSocket.recv(4)
    #print(str(un))
    #t = zlib.decompress(connectionSocket.recv(4), wbits = 18)

    t = connectionSocket.recv(4096)

    if t:
        data += t
        #print(int_to_bytes(t,len(t)))
        #print(readInt(t))
    else:

        try:
            d = zlib.decompress(data[10:-8],wbits = -15)

            #print('decompressed')
            #print(len(d))
            
            for i in range(0,5):
                integ = readInt(d[i*4:(i*4 + 4)])
                ints.append(integ)
                #print(integ)
                
            pixels_start = ints[4]
            #print(pixels_start)
            sign_name = readString(d[5*4:int((5*4+pixels_start*4))])
            
            frames = ints[0]
            W = ints[1]
            H = ints[2]
            fps = ints[3]
            #print('Frames: '+str(frames)+' Widht: '+str(W)+' Height: '+str(H)+' FPS: '+str(fps) + ' Name: '+sign_name)
            
            #if not os.path.exists(sign_name):
            #    os.mkdir(sign_name)
                #print("Directory " , sign_name ,  " Created ")
            #else:    
            #    print("Directory " , sign_name ,  " already exists")
            
            p_data = d[(5*4+pixels_start*4):]
            
            #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #print(sign_name+'.mp4')
            #video = cv2.VideoWriter(sign_name+'.mp4', fourcc, fps, (W,H))
            #print('Video ...')
            os.popen('pkill -f JASApp.jar')
            print('JASAPP.jar killed, doing Frames ...')
            for i in range(0,frames):
                #print("Video f")
                array = p_data[i*3*W*H:(i*3*W*H + 3*W*H)]
                im= Image.new('RGB', (W, H))
                #frame_pixels.reverse()
                
                y = H-1
                x = 0
                #print("col pix start")
                #print(str(len(array)/3))
                for p in range(0,int(len(array)/3)):
                    im.putpixel((x,y),(array[p*3+2]& 0xFF,array[p*3+1]& 0xFF,array[p*3]& 0xFF))
                    #im.putpixel((x,y),(array[p*3]& 0xFF,array[p*3+1]& 0xFF,array[p*3+2]& 0xFF))

                    #print(p)
                    x = x + 1
                    if x == W:
                        y = y - 1
                        x = 0
                #print("col pix end")
                #m.save(sign_name+'.png')
                    #print(str(array))
                #im = cv2.imdecode(array,1)
                #im.save('/'+sign_name+'/'+str(i)+'.png', compress_level=1)
                im.save(sign_name+'_'+str(i)+'.png', compress_level=1)
                del(im)
                #print("saved")
                #video.write(np.array(im))
            #cv2.destroyAllWindows()
            #video.release()
            #print('Video SAVED')
            print('Frames of '+sign_name+' SAVED')
            del(p_data)
            del(d)

            print('loaded')

            #del(video)
            
            #for i in range(0,int(len(p_data)/3)):
            #    pix = read_pixel(p_data[i*3:(i*3 + 3)])
            #    pixel_ints.append(pix)
        except:
            print("Bad stream -- ")
        connectionSocket.close()
        check_conn_socket.close()
        del(data)
        del(t)

        #os.system("killall -9 SiGMLPlayer");
        #os.popen("kill -9 $(ps aux | grep  JASApp | awk '{print $2}')")


        #p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        #out, err = p.communicate()
        #for line in out.splitlines():
        #    if 'SiGMLPlayer' in line:
        ##        pid = int(line.split(None, 1)[0])
        #        os.kill(pid, signal.SIGKILL)
        #readpixels(sign_name,frames,pixel_ints,W,H)
        #print(len(pixel_ints))
        print('Ready for the next sign')
        data = "".encode("utf-8")
        ints = []
        pixel_ints = [] 
        connectionSocket, addr = server_socket.accept()
        check_conn_socket, addr = check_socket.accept()
        d = []
        