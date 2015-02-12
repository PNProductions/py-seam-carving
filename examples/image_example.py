from image_helper import image_open, local_path
from seamcarving import seam_carving
import cv2
from numpy import size
from utils.seams import print_seams
import time
import os

makeNewDecData = False
debug = False
saveBMP = True

file_suffix = '_small'
folder_name = 'results'

X = image_open(local_path('../assets/seam_carving' + file_suffix + '.bmp'))

deleteNumberW = -size(X, 1) / 2

img, seams = seam_carving(X, deleteNumberW, False)

seams = print_seams(X, img, seams, deleteNumberW)

size = '_reduce' if deleteNumberW < 0 else '_enlarge'
size += str(-deleteNumberW) if deleteNumberW < 0 else str(deleteNumberW)
name = 'result_mod_' + size + '_' + str(int(time.time()))
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
cv2.imwrite(local_path('./' + folder_name + '/' + name + '.png'), img)

cv2.imwrite(local_path('./' + folder_name + '/' + name + '_seams_.png'), seams)
