import cv2
import numpy as np

cv2.namedWindow("Matching result", cv2.WINDOW_NORMAL)

img1 = cv2.imread("images/img1.JPG",cv2.IMREAD_COLOR)
img2 = cv2.imread("images/img1_crop.JPG",cv2.IMREAD_COLOR)

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
'''
img1 = cv2.resize(img1, (960, 640))
img2 = cv2.resize(img2, (960, 640))
'''

cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
