from image_helper import image_open, local_path
from seamcarving import seam_carving
import cv2
from numpy import size, float64, array, abs
import time
import os

alpha = 0.5
betaEn = 0.5

iterTV = 80
makeNewDecData = False

debug = False
saveBMP = True

file_suffix = '_small'

folder_name = 'results'

X = image_open(local_path('../assets/seam_carving' + file_suffix + '.bmp'))


deleteNumberW = -size(X, 1) / 2
deleteNumberH = 0

y = cv2.cvtColor(X, cv2.COLOR_BGR2YCR_CB)
y = y.astype(float64)

importance = y
kernel = array([[0, 0, 0],
                [1, 0, -1],
                [0, 0, 0]
                ])
importance = abs(cv2.filter2D(y[:, :, 0], -1, kernel, borderType=cv2.BORDER_REPLICATE)) + abs(cv2.filter2D(y[:, :, 0], -1, kernel.T, borderType=cv2.BORDER_REPLICATE))

img = seam_carving(X, importance, deleteNumberW, alpha, betaEn, False)

size = '_reduce' if deleteNumberW < 0 else '_enlarge'
size += str(-deleteNumberW) if deleteNumberW < 0 else str(deleteNumberW)
name = 'result_mod_' + size + '_' + str(int(time.time()))
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
cv2.imwrite(local_path('./' + folder_name + '/' + name + '.bmp'), img)
