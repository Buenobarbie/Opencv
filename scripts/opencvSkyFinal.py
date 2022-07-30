import cv2
from cv2 import erode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def nothing(x):
    pass

def get_mask(hsv , lower_color , upper_color):
    lower = np.array(lower_color)
    upper = np.array(upper_color)
    
    mask = cv2.inRange(hsv , lower, upper)

    return mask

capture = cv2.VideoCapture(0)
skyratsImg = cv2.imread("images/skyrats_logo.jpg")
skyratsImgResize = cv2.resize(skyratsImg, (300, 300))

cv2.namedWindow('Parâmetros')

cv2.namedWindow('Camera')
cv2.createTrackbar('HMin', 'Parâmetros', 0, 179, nothing)
cv2.createTrackbar('SMin', 'Parâmetros', 0, 255, nothing)
cv2.createTrackbar('VMin', 'Parâmetros', 0, 255, nothing)
cv2.createTrackbar('HMax', 'Parâmetros', 179, 179, nothing)
cv2.createTrackbar('SMax', 'Parâmetros', 255, 255, nothing)
cv2.createTrackbar('VMax', 'Parâmetros', 255, 255, nothing)
cv2.createTrackbar('Erode', 'Parâmetros', 0, 100, nothing)
cv2.createTrackbar('Dilate', 'Parâmetros', 0, 100, nothing)


while True: 
    success, frame = capture.read()
    if success == False:
        raise ConnectionError


    hMin = cv2.getTrackbarPos('HMin', 'Parâmetros')
    sMin = cv2.getTrackbarPos('SMin', 'Parâmetros')
    vMin = cv2.getTrackbarPos('VMin', 'Parâmetros')
    hMax = cv2.getTrackbarPos('HMax', 'Parâmetros')
    sMax = cv2.getTrackbarPos('SMax', 'Parâmetros')
    vMax = cv2.getTrackbarPos('VMax', 'Parâmetros')

    lower = [hMin, sMin, vMin]
    upper = [hMax, sMax, vMax]

    # get_mask(hsv, [160, 100, 20], [179, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = get_mask(hsv, lower, upper)

    
    result = cv2.bitwise_and(frame , frame , mask= mask)
    
    #plotting
    erode_size = cv2.getTrackbarPos('Erode', 'Parâmetros')
    dilate_size = cv2.getTrackbarPos('Dilate', 'Parâmetros')

    erode_kernel = np.ones((erode_size, erode_size), np.float32)
    dilate_kernel = np.ones((dilate_size, dilate_size), np.float32)
    
    result = cv2.dilate(result, dilate_kernel)
    result = cv2.erode(result, erode_kernel)
    
    cv2.imshow('Parâmetros', skyratsImgResize)
    cv2.imshow('Camera', result)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

cv2.waitKey(0)