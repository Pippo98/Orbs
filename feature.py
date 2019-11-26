import cv2
import numpy as np
import math
from skimage.measure import ransac
from PIL import Image, ImageFilter
#from helpers import add_ones, poseRt, fundamentalToRt, normalize, EssentialMatrixTransform, myjet

CULLING_ERR_THRES = 0.02

RANSAC_RESIDUAL_THRES = 0.02
RANSAC_MAX_TRIALS = 100

MIN_MATCHES_REQUIRED = 10

movie = cv2.VideoCapture(0)
cv2.namedWindow("Feature Tracking", cv2.WINDOW_NORMAL)

flann_params = dict(algorithm = 6,trees = 4)    
matcher = cv2.FlannBasedMatcher(flann_params, {})

def dist(p1, p2):
    distance = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
    distance = int(distance)
    return distance


ret, frame = movie.read()
frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
previous_frame = frame

H, W = frame.shape

previous_x = 0
previous_y = 0

process_this_frame = 2
frame_count = 0

while True:
  ret, frame = movie.read()
  frame_count += 1
  #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

  '''
  kernel = np.ones((5,5),np.float32)/25
  frame2 = cv2.filter2D(frame,-1,kernel)
  frame = frame2 - frame
  #frame = cv2.filter2D(frame,-1,kernel)
  '''

  if ret != True:
      break

  img1 = previous_frame
  img2 = frame

  
  # ORB Detector
  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(img1, None)
  kp2, des2 = orb.detectAndCompute(img2, None)

  # BFMatcher with default params
  bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches=bf.match(des1,des2)
  matches=sorted(matches, key= lambda x:x.distance)

  if len(matches) < MIN_MATCHES_REQUIRED:
    continue

  # Apply ratio test
  # Initialize lists
  list_kp1 = []
  list_kp2 = []

  # For each match...
  for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append((int(x1), int(y1)))
    list_kp2.append((int(x2), int(y2)))

  #img2 = cv2.drawMatches(img2, kp1, img2, kp2, matches, None, flags=2)
  vector = []
  img3 = img2
  for i in range(len(list_kp1)):
    if (i < len(list_kp1) and i < 100):
      x1, y1 = list_kp1[i]
      x2, y2 = list_kp2[i]
      y = float(float(y2)-float(y1))
      x = float(float(x2)-float(x1))
      if x == 0 or y == 0:
        continue

      modulo = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
      if abs(modulo) > W/4:
        continue

      direction_x = 1 if (x2-x1) >= 0 else -1
      direction_y = 1 if (y2-y1) >= 0 else -1

      verso = math.degrees(math.atan(y/x))
      
      if direction_x*-1 < 0 and direction_y > 0:
        verso += 90
      elif direction_x*-1 < 0 and direction_y < 0:
        verso += 180
      elif direction_x*-1 > 0 and direction_y < 0:
        verso += 270
      
      vector.append(
         {
          "modulo": modulo,
          "dir_x": direction_x,
          "dir_y": direction_y,
          "verso": verso
         }
      )
      #print(vector[-1]["dir_x"], vector[-1]["dir_y"])
      img3 = cv2.line(img2, list_kp1[i], list_kp2[i], (0,255,255), 1)


  modulo_range = 0.5
  verso_range = 0.1

  idxs = []
  for idx1, v1 in enumerate(vector):
    buffer = []
    for idx2, v2 in enumerate(vector):
      if v1 != v2:
        if v1["dir_x"] == v2["dir_x"] and v1["dir_y"] == v2["dir_y"]:
          if abs(v1["modulo"]) < W/4:
            if abs(v1["modulo"]) > abs(v2["modulo"] * (1-modulo_range)) and abs(v1["modulo"]) < abs(v2["modulo"] * (1+modulo_range)):
              if v1["verso"] > v2["verso"] * (1-verso_range) and v1["verso"] < v2["verso"] * (1+verso_range):
                buffer.append(idx2)
    if not buffer == [] and len(buffer) > MIN_MATCHES_REQUIRED:
      idxs.append(buffer)

  idxs.sort(key=len)
  #print(idxs)
  if(len(idxs) > 0):
    if (len(idxs[-1]) > 0):
      motion = {
        "modulo": 0,
        "dir_x": 0,
        "dir_y": 0,
        "verso": 0
      }
      try:
        for idx in idxs[-1]:
          motion["modulo"] += vector[idx]["modulo"]
          motion["verso"] += vector[idx]["verso"]
          motion["dir_x"] += vector[idx]["dir_x"]
          motion["dir_y"] += vector[idx]["dir_y"]
          #print(vector[idx]["verso"])
      except:
        pass
      
      motion["modulo"] /= len(idxs[-1])
      motion["verso"] /= len(idxs[-1])
      motion["dir_x"] = 1 if motion["dir_x"] >= 0 else -1
      motion["dir_y"] = 1 if motion["dir_y"] >= 0 else -1

      scale = 2
      x1, y1 = int(W/2), int(H/2)
      x2 = int(W/2 + motion["modulo"] * scale * math.sin(math.radians(motion["verso"])))
      y2 = int(H/2 - motion["modulo"] * scale * math.cos(math.radians(motion["verso"])))
      cv2.arrowedLine(img3, (x1,y1), (x2,y2), (0,0,255), 2)
      #print(motion["dir_x"], motion["dir_y"])

  
  
  
  
  
  
  if frame_count >= process_this_frame:
    previous_frame = frame
    frame_count = 0

  img3 = cv2.drawKeypoints(img3, kp1, None, (255,0,0))
  
  cv2.imshow("Feature Tracking", img3)

  key = cv2.waitKey(1)
  if key == 27:
    cv2.destroyAllWindows()
    exit(0)

    