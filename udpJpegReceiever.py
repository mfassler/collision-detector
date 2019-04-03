#!/usr/bin/env python3

import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


import select
import socket
import struct
import time
import numpy as np
import cv2



WRITE_VIDEO = False
vid_out = None
if WRITE_VIDEO:
    vid_out = cv2.VideoWriter('cam_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (848, 480))


START_MAGIC = b"__HylPnaJY_START_JPG "
STOP_MAGIC = b"_g1nC_EOF"

DATA_PORT = 3101
IMAGE_PORT = 3201

data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(("0.0.0.0", DATA_PORT))

image_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
image_sock.bind(("0.0.0.0", IMAGE_PORT))


STOP_DISTANCE = 3.0  # meters
WARN_DISTANCE = 4.0  # meters


def showImage():
    global _last_stop_time
    global WRITE_VIDEO
    global vid_out
    closest_host_idx = -1
    closest_bbox_idx = -1
    closest_distance = 10000.00  # realsense can only go to ~10 meters.
    ts = time.time()
    hostnames = list(rx_data.keys())
    for i, key in enumerate(hostnames):
        #if ts - rx_data[key]['ts'] > 5.0:
        #    rx_data[key]['nboxes'] = 0
        #else:
            for j in range(rx_data[key]['nboxes']):
                if rx_data[key]['distances'][j] < closest_distance:
                    closest_distance = rx_data[key]['distances'][j]
                    closest_bbox_idx = j
                    closest_host_idx = i

    if closest_host_idx == -1:
        closest_host_idx = 0  # default value
        closest_bbox_idx = 0  # default value

    hostname = hostnames[closest_host_idx]
    #im = rx_jpgs[hostname]['jpg']
    nboxes = rx_data[hostname]['nboxes']
    bboxes = rx_data[hostname]['bboxes']
    distances = rx_data[hostname]['distances']

    #print("********* ", nboxes)
    #aa = np.frombuffer(rx_jpgs[hostname]['jpgData'], np.uint8)
    im = cv2.imdecode(rx_jpgs[hostname]['jpgData'], cv2.IMREAD_UNCHANGED)
    for i in range(nboxes):
        if distances[i] < STOP_DISTANCE:
            cimg2 = cv2.rectangle(im, (bboxes[i][0], bboxes[i][2]), (bboxes[i][1], bboxes[i][3]),
                (0,0,255), 4)
        elif distances[i] < WARN_DISTANCE:
            cimg2 = cv2.rectangle(im, (bboxes[i][0], bboxes[i][2]), (bboxes[i][1], bboxes[i][3]),
                (0,255,255), 4)
        else:
            cimg2 = cv2.rectangle(im, (bboxes[i][0], bboxes[i][2]), (bboxes[i][1], bboxes[i][3]),
                (255,255,255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if closest_distance < WARN_DISTANCE:
        text = "%.02f m" % (closest_distance)
        cv2.putText(im, text, (19,119), font, 4, (0,0,0), 8)  # black shadow
        cv2.putText(im, text, (10,110), font, 4, (255,255,255), 8)  # white text

    #cv2.setWindowProperty('RealSense', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('RealSense', im)
    cv2.waitKey(1)
    if WRITE_VIDEO:
        vid_out.write(im)





cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)

rx_jpgs = {}

rx_data = {}
nboxes = 0
distances = np.empty(16, np.float)
bboxes = np.empty((16,4), np.int32)

while True:
    inputs, outputs, errors = select.select([data_sock, image_sock], [], [])
    for oneInput in inputs:
        if oneInput == image_sock:
            data, addr = image_sock.recvfrom(65535)
            if addr[0] not in rx_jpgs:
                rx_jpgs[addr[0]] = {
                    'size': 0,
                    'packets': [],
                    'inBand': False
                }
            #print(addr)

            if rx_jpgs[addr[0]]['inBand']:
                if data == STOP_MAGIC:
                    rx_jpgs[addr[0]]['inBand'] = False
                    if len(rx_jpgs[addr[0]]['packets']) > 1:
                        jpgData = b''.join(rx_jpgs[addr[0]]['packets'])
                        if rx_jpgs[addr[0]]['size'] == len(jpgData):
                            rx_jpgs[addr[0]]['jpgData'] = np.frombuffer(jpgData, np.uint8)
                            #rx_jpgs[addr[0]]['jpg'] = cv2.imdecode(aa, cv2.IMREAD_UNCHANGED)
                        else:
                            print('image size doesn\'t match')
                        try:
                            showImage()
                        except Exception as e:
                            print(e)
                    else:
                        print('no data')
                else:
                    rx_jpgs[addr[0]]['packets'].append(data)
        
            else:
                if data.startswith(START_MAGIC):
                    rx_jpgs[addr[0]]['size'] = int(data[-10:], 10)
                    rx_jpgs[addr[0]]['packets'] = []
                    rx_jpgs[addr[0]]['inBand'] = True

        if oneInput == data_sock:
            data, addr = data_sock.recvfrom(65535)
            if addr[0] not in rx_data:
                rx_data[addr[0]] = {
                    'remote_id': 0,
                    'nboxes': 0,
                    'distances': np.empty(16, np.float),
                    'bboxes': np.empty((16, 4), np.int32),
                    'ts': time.time()
                }
            #print(addr[0])
            rx_data[addr[0]]['remote_id'] = data[2]
            print('remote_id:', rx_data[addr[0]]['remote_id'])
            rx_data[addr[0]]['nboxes'] = data[3]
            rx_data[addr[0]]['ts'] = time.time()
            for i in range(rx_data[addr[0]]['nboxes']):
                iStart = i*20 + 4
                iStop = iStart + 20
                #distances[i], bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3] = \
                a, b, c, d, e = struct.unpack('fiiii', data[iStart:iStop])
                rx_data[addr[0]]['distances'][i] = a
                rx_data[addr[0]]['bboxes'][i][0] = b
                rx_data[addr[0]]['bboxes'][i][1] = c
                rx_data[addr[0]]['bboxes'][i][2] = d
                rx_data[addr[0]]['bboxes'][i][3] = e
                #print(a,b,c,d,e)
                #print("%d  %f, %d %d %d %d" % (i, distances[i], bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3] ))
            #print(distances[:nboxes])



