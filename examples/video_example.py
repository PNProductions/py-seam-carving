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

progress_bar(False)


def generate_step(I, img):
  result = np.empty((img.shape[0], img.shape[1], img.shape[2]))
  r = np.arange(img.shape[2])
  for i in xrange(img.shape[0]):
    result[i] = r > np.vstack(I[i])
  return result.astype(np.uint64)

def print_seams(result, seams):
  print "printing seam"
  seams = seams.astype(np.uint64)
  A = np.zeros_like(result)
  correction = np.zeros((result.shape[0], result.shape[1], result.shape[2])).astype(np.uint64)
  for i in xrange(seams.shape[0]):
    X, Y = np.mgrid[:result.shape[0], :result.shape[1]]
    I = seams[i]
    I = I + correction[X, Y, I]
    color = np.random.rand(3) * 255
    A[X, Y, I] = color
    correction = correction + generate_step(I, result)
  return A

deleteNumberW = 1
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

A = print_seams(video, seams)
A = np.clip(A * 0.8 + video, 0, 255).astype(np.uint8)

result = np.clip(result, 0, 255).astype(np.uint8)
save_video_caps(result, local_path('results/') + name + '_')
save_video_caps(A, local_path('results/') + name + '_seams_')
cap.release()
print 'Finished file: ' + basename(filename)
