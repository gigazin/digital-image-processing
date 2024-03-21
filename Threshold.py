from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np


max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'


def Threshold_Demo(val):
 #0: Binary
 #1: Binary Inverted
 #2: Threshold Truncated
 #3: Threshold to Zero
 #4: Threshold to Zero Inverted
 threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
 threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
 _, dst = cv.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
 cv.imshow(window_name, dst)


src = cv.imread('images/digital.jpg')
if src is None:
 print('Could not open or find the image: ', src)
 exit(0)
# Convert the image to Gray
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


rgb_planes = cv.split(src)
histSize = 256
histRange = (0, 256)
accumulate = False

b_hist = cv.calcHist(rgb_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv.calcHist(rgb_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv.calcHist(rgb_planes, [2], None, [histSize], histRange, accumulate=accumulate)

hist_w = 512
hist_h = 400
bin_w = int(round( hist_w/histSize ))

histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)


for i in range(1, histSize):
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(b_hist[i]) ),
            ( 255, 0, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(g_hist[i]) ),
            ( 0, 255, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(r_hist[i]) ),
            ( 0, 0, 255), thickness=2)
           
           
           
cv.namedWindow(window_name)


cv.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold_Demo)
# Create Trackbar to choose Threshold value
cv.createTrackbar(trackbar_value, window_name , 0, max_value, Threshold_Demo)

cv.imshow('Histogram', histImage)

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv.waitKey()