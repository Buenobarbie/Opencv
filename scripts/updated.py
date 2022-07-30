import cv2
from cv2 import erode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#U
def nothing(x):
    pass

def get_mask(hsv , lower_color , upper_color):
    lower = np.array(lower_color)
    upper = np.array(upper_color)
    
    mask = cv2.inRange(hsv , lower, upper)

    return mask

capture = cv2.VideoCapture(0)
cv2.namedWindow('frame')
cv2.namedWindow('frame2')
cv2.createTrackbar('HMin', 'frame', 0, 179, nothing)
cv2.createTrackbar('SMin', 'frame', 0, 255, nothing)
cv2.createTrackbar('VMin', 'frame', 0, 255, nothing)
cv2.createTrackbar('HMax', 'frame', 179, 179, nothing)
cv2.createTrackbar('SMax', 'frame', 255, 255, nothing)
cv2.createTrackbar('VMax', 'frame', 255, 255, nothing)
cv2.createTrackbar('Erode', 'frame', 0, 50, nothing)
cv2.createTrackbar('Dilate', 'frame', 0, 50, nothing)


hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while True: 
    success, frame = capture.read()
    if success == False:
        raise ConnectionError


    hMin = cv2.getTrackbarPos('HMin', 'frame')
    sMin = cv2.getTrackbarPos('SMin', 'frame')
    vMin = cv2.getTrackbarPos('VMin', 'frame')
    hMax = cv2.getTrackbarPos('HMax', 'frame')
    sMax = cv2.getTrackbarPos('SMax', 'frame')
    vMax = cv2.getTrackbarPos('VMax', 'frame')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = get_mask(hsv, lower, upper)

    
    result = cv2.bitwise_and(frame , frame , mask= mask)
    
    #plotting
    erode_size = cv2.getTrackbarPos('Erode', 'frame')
    dilate_size = cv2.getTrackbarPos('Dilate', 'frame')

    erode_kernel = np.ones((erode_size, erode_size), np.float32)
    dilate_kernel = np.ones((dilate_size, dilate_size), np.float32)
    
    result = cv2.dilate(result, dilate_kernel)
    result = cv2.erode(result, erode_kernel)
    
    

    cv2.imshow('frame2', result)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

cv2.waitKey(0)