#!/usr/bin/env python

# Test script is used only for testing the algorithm with only one video, no batch. For batch videos see py-cloud-opencv.
from os.path import basename, splitext
import numpy as np
import time
import cv2
import sys
from seamcarving import seam_carving, progress_bar
from video_helper import save_video_caps
from image_helper import local_path
from utils.seams import print_seams

progress_bar(False)


deleteNumberW = -1
counting_frames = 10
filename = '../assets/car.m4v'
suffix = ''
for i in xrange(len(sys.argv) - 1):
  if sys.argv[i] == '-s':
    deleteNumberW = int(sys.argv[i + 1])
  elif sys.argv[i] == '-f':
    counting_frames = int(sys.argv[i + 1])

# onlyfiles = glob.glob(path)

makeNewDecData = False

debug = False
saveBMP = True

cap = cv2.VideoCapture(local_path(filename))
size = '_enlarge' if deleteNumberW < 0 else '_reduce'
size += str(-deleteNumberW) if deleteNumberW < 0 else str(deleteNumberW)

name = splitext(basename(filename))[0] + suffix + '_' + size + '_' + str(int(time.time()))

frames_count, fps, width, height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT), cap.get(cv2.cv.CV_CAP_PROP_FPS), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
frames_count = frames_count if counting_frames is None else counting_frames

video = np.empty((frames_count, height, width, 3))

i = 0
while cap.isOpened() and i < frames_count:
  ret, X = cap.read()
  if not ret:
    break
  video[i] = X
  i += 1

result, seams = seam_carving(video, deleteNumberW, False)

A = print_seams(video, result, seams, deleteNumberW)
A = np.clip(A * 0.8 + video, 0, 255).astype(np.uint8)

result = np.clip(result, 0, 255).astype(np.uint8)
save_video_caps(result, local_path('results/') + name + '_')
save_video_caps(A, local_path('results/') + name + '_seams_')
cap.release()
print 'Finished file: ' + basename(filename)
