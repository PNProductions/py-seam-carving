from image_helper import image_open, local_path
from seamcarving import seam_carving
import cv2
from numpy import size
import time
import os

makeNewDecData = False
debug = False
saveBMP = True

file_suffix = '_small'
folder_name = 'results'

X = image_open(local_path('../assets/seam_carving' + file_suffix + '.bmp'))

deleteNumberW = -size(X, 1) / 2

img = seam_carving(X, deleteNumberW, False)

size = '_reduce' if deleteNumberW < 0 else '_enlarge'
size += str(-deleteNumberW) if deleteNumberW < 0 else str(deleteNumberW)
name = 'result_mod_' + size + '_' + str(int(time.time()))
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
cv2.imwrite(local_path('./' + folder_name + '/' + name + '.bmp'), img)
