#!/usr/bin/env python3

import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import time
import select
import socket
import struct
import numpy as np
import cv2 as cv


DATA_PORT = 3125

data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(("0.0.0.0", DATA_PORT))


depth_scale = 0.001

coords = np.empty((848, 2), np.float64)

#left_xrange = (np.arange(848) - rgb0_intrinsics.ppx) / rgb0_intrinsics.fx  # this doesn't change
#left_xrange = (np.arange(848) - 424) / 615.0
left_xrange = (np.arange(848) - 432) / 615.5

theta = np.radians(69.3/ 2.0)

#avoidance_areas = np.ones((800, 800, 3), np.uint8) * 255
#avoidance_areas = cv.imread('avoidance_areas/avoidance_areas.png')
avoidance_areas = cv.imread('avoidance_areas/very_large.png')

vid_out = None
WRITE_VIDEO = False
if WRITE_VIDEO:
    vid_out = cv.VideoWriter('map_video.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (800, 800))


amap = np.copy(avoidance_areas)
cv.imshow('a map', amap)
cv.moveWindow('a map', 560, 0)
cv.waitKey(1)

while True:
    inputs, outputs, errors = select.select([data_sock], [], [])
    for oneInput in inputs:
        if oneInput == data_sock:

            data = None

            # empty out the recv buffer (in case fig.canvas.draw is slower than udp packets)
            data_sock.setblocking(0)
            cont=True
            while cont:
                try:
                    tmpData, addr = data_sock.recvfrom(65535)
                except Exception as ee:
                    #print(ee)
                    cont=False
                else:
                    if tmpData:
                        if data is not None:
                            print('throwing away a packet (GUI is too slow)')
                        data = tmpData
                    else:
                        cont=False
            data_sock.setblocking(1)

            #print(len(data))
            scanline = np.frombuffer(data, dtype='<u2')
            coords[:,1] = scanline * depth_scale
            coords[:,0] = left_xrange * coords[:,1]

        # Draw a real-time map on a white background
        #  (each pixel represents 1cm x 1cm in real space)
        amap = np.copy(avoidance_areas)

        for pt in coords:
            x = int(round(pt[0]*100) + 400) # graphical coords, in cm
            y = int(800 - round(pt[1]*100)) # graphical coords, in cm
            if (5 <= x < 795) and (5 <= y < 795):
                #amap[y, x] = [255, 0, 0]
                cv.circle(amap, (x,y), 3, [255,0,0], -1)

        cv.imshow('a map', amap)
        cv.waitKey(1)

        if WRITE_VIDEO:
            vid_out.write(amap)


