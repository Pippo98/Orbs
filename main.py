#!/usr/bin/env python3

import cv2
import numpy as np

mov = cv2.VideoCapture(0)

orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)

while(True):
    ret, frame = mov.read()
    
    
    orbs = orb.detect(frame,None)

    print("l")

    kp, bla = orb.compute(frame, orbs)
    print("l")

    framme = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
    print("l")
    
    cv2.imshow("image", framme)
    cv2.waitKey(1)
