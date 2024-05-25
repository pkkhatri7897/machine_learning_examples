import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import argparse
import random as rnd
import time

########################################################### IMAGE MASKING FROM HSV FORMAT TUT ##################################################################

# coating_thickness = {
#     0.33 : 120,
#     0.43 : 156,
#     0.55 : 180,
#     0.60 : 204,
#     0.66 : 215
# }

# img_file = 'data/pic1.avif'
# jpg_img = Image.open(img_file)
# jpg_img.save('data/pic.jpg')


# def nothing(x):
#     pass
    

# cv2.namedWindow('Tracking')
# # create trackbar
# cv2.createTrackbar('LH', 'Tracking', 0, 255, nothing)
# cv2.createTrackbar('UH', 'Tracking', 255, 255, nothing)
# cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing)
# cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)
# cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing)
# cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)

# while True:
#     img = cv2.imread('data/pic.jpg', 1)
#     img = cv2.resize(img, (640, 480))
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # get trackbar value
#     lh = cv2.getTrackbarPos('LH', 'Tracking')
#     ls = cv2.getTrackbarPos('LS', 'Tracking')
#     lv = cv2.getTrackbarPos('LV', 'Tracking')

#     uh = cv2.getTrackbarPos('UH', 'Tracking')
#     us = cv2.getTrackbarPos('US', 'Tracking')
#     uv = cv2.getTrackbarPos('UV', 'Tracking')
    
#     lb = np.array([lh, ls, lv])
#     ub = np.array([uh, us, uv])
#     mask = cv2.inRange(hsv_img, lb, ub)
#     res = cv2.bitwise_and(img, img, mask=mask)

#     cv2.imshow('img', img)
#     cv2.imshow('mask', hsv_img)
#     cv2.imshow('result', res)

#     key = cv2.waitKey(0)
#     if key == 27:
#         break

# cv2.destroyAllWindows()

################################################################################################################################################################



############################################################### HSV IMAGE MASKING EX ###########################################################################

# def nothing(x):
#     pass

# def mouseEventCallback(event, x, y, flags, params):
#     if event == cv.EVENT_RBUTTONDOWN:
#         blue = img[y, x, 0]
#         green = img[y, x, 1]
#         red = img[y, x, 2]
#         strColor = str(blue) + ', ' + str(green) + ', ' + str(red)
#         cv.putText(img, strColor, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
#         cv.imshow("image", img)

# cv.namedWindow('Tracking')
# cv.resizeWindow('Tracking', 360, 240)
# cv.createTrackbar("LH", 'Tracking', 0, 255, nothing)
# cv.createTrackbar("UH", 'Tracking', 255, 255, nothing)
# cv.createTrackbar("LS", 'Tracking', 0, 255, nothing)
# cv.createTrackbar("US", 'Tracking', 255, 255, nothing)
# cv.createTrackbar("LV", 'Tracking', 0, 255, nothing)
# cv.createTrackbar("UV", 'Tracking', 255, 255, nothing)

# while True:
#     # img = cv.imread("data/img5.jpg")
#     # img = cv.imread("data/golf.jpg")
#     # img = cv.imread("data/uv.jpg")
#     # img = cv.imread("data/pcb_coating3.jpeg")
#     # img = cv.resize(img, (360, 620))
#     # img = cv.imread("data/987010_TOP.bmp")
#     # img = cv.resize(img, (1280, 720))
#     # img = cv.imread('data/987010_T_BOT_CP.bmp')
#     # img = cv.resize(img, (1280, 720))
#     # img = cv.imread('data/blue.jpg')
#     img = cv.imread('data/pcb1.bmp')
#     img = cv.resize(img, (700,700))
#     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#     lh = cv.getTrackbarPos('LH', 'Tracking')
#     uh = cv.getTrackbarPos('UH', 'Tracking')
#     ls = cv.getTrackbarPos('LS', 'Tracking')
#     us = cv.getTrackbarPos('US', 'Tracking')
#     lv = cv.getTrackbarPos('LV', 'Tracking')
#     uv = cv.getTrackbarPos('UV', 'Tracking')

#     lb = np.array([lh, ls, lv])
#     ub = np.array([uh, us, uv])

#     mask = cv.inRange(hsv, lb, ub)
#     res = cv.bitwise_and(img, img, mask=mask)

#     # cv.imshow('original', img)
#     cv.imshow("mask", mask)
#     cv.imshow('result', res)
#     cv.setMouseCallback("result", mouseEventCallback)

#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break

# cv.destroyAllWindows()

################################################################################################################################################################



############################################################### MEASURE IMAGE SIZE #############################################################################

# class detObjInHomoBackgrnd():
#     def __init__(self):
#         pass

#     def detect_object(self, frame):
#         # convert image into gray scale
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#         # create a mask with adaptive threshold
#         mask = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 5)

#         # find contours
#         contours, heirarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         cv.imshow('mask', mask)

#         object_contours = []

#         for cnt in contours:
#             area = cv.contourArea(cnt)
#             if area > 2000:
#                 object_contours.append(cnt)

#         return object_contours


# # LOAD ARUCO DETECTOR.
# params = cv.aruco.DetectorParameters()
# aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)

# # LOAD OBJECT DETECTOR
# detectror = detObjInHomoBackgrnd()

# # READ THE IMAGE
# # img = cv.imread('data/img.jpg')
# # img = cv.imread('data/img2.jpg')
# # img = cv.imread('data/phone.jpg')
# img = cv.imread('data/phone_aruco_marker.jpg')
# cap = cv.VideoCapture(0)

# while True:
#     ret, img = cap.read()

#     # get aruco markers
#     corners, _, _ = cv.aruco.detectMarkers(img, aruco_dict, parameters=params)

#     if corners:

#         # draw aruco poly lines.
#         int_cnr = np.int0(corners)
#         cv.polylines(img, int_cnr, True, (0, 255, 0), 2)

#         # aruco perimeters.
#         aruco_peri = cv.arcLength(corners[0], True)
#         # print(aruco_peri) # 20cm means all sides are 5 cm correspondes to 590 pixels it cover.

#         # pixels to cm ratio.
#         pixel_cm_ratio = aruco_peri / 20
#         print("1 cm cover pixels is ", pixel_cm_ratio)


#         contours = detectror.detect_object(img)


#         # Draw oject boundries
#         for cnt in contours:
#             # img is curv is not proper rectangle to detect.
#             # draw polygon
#             # cv.polylines(img, [cnt], True, (0, 0, 255), 2)

#             # get rectangle
#             # x & y is centre point of object.
#             rect = cv.minAreaRect(cnt)
#             (x, y), (w, h), angle = rect

#             # coords should be int.
#             # width & hight of objects by applying ratio. 
#             obj_w = w / pixel_cm_ratio
#             obj_h = h / pixel_cm_ratio

#             # draw rectangle around object instaed of poly line for measuring its dimension.
#             box = cv.boxPoints(rect)
#             box = np.int0(box)

#             # display centre point and box plot on object.
#             cv.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
#             cv.polylines(img, [box], True, (0, 0, 255), 2)
            
#             cv.putText(img, "width {} cm".format(round(obj_w, 1)), (int(x), int(y - 15)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
#             cv.putText(img, "hight {} cm".format(round(obj_h, 1)), (int(x + 10), int(y + 15)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            

#         cv.imshow('img', img)
#         k = cv.waitKey(1)
#         if k == 27:
#             break

# cv.destroyAllWindows()

################################################################################################################################################################


############################################################### MEASURE IMAGE SIZE 1 ###########################################################################

# class detObjInHomoBackgrnd():
#     def __init__(self):
#         pass

#     def detect_object(self, frame):
#         # convert image into gray scale
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#         # create a mask with adaptive threshold
#         mask = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 5)

#         # find contours
#         contours, heirarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         cv.imshow('mask', mask)

#         object_contours = []

#         for cnt in contours:
#             area = cv.contourArea(cnt)
#             if area > 2000:
#                 object_contours.append(cnt)

#         return object_contours


# # LOAD ARUCO DETECTOR.
# params = cv.aruco.DetectorParameters()
# aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)

# # LOAD OBJECT DETECTOR
# detectror = detObjInHomoBackgrnd()

# # READ THE IMAGE
# # img = cv.imread('data/img.jpg')
# # img = cv.imread('data/img2.jpg')
# # img = cv.imread('data/phone.jpg')
# img = cv.imread('data/phone_aruco_marker.jpg')
# # cap = cv.VideoCapture(0)
# # ret, img = cap.read()

# # get aruco markers
# corners, _, _ = cv.aruco.detectMarkers(img, aruco_dict, parameters=params)

# # draw aruco poly lines.
# int_cnr = np.int0(corners)
# cv.polylines(img, int_cnr, True, (0, 255, 0), 2)

# # aruco perimeters.
# aruco_peri = cv.arcLength(corners[0], True)
# # print(aruco_peri) # 20cm means all sides are 5 cm correspondes to 590 pixels it cover.

# # pixels to cm ratio.
# pixel_cm_ratio = aruco_peri / 20
# print("1 cm cover pixels is ", pixel_cm_ratio)


# contours = detectror.detect_object(img)


# # Draw oject boundries
# for cnt in contours:
#     # img is curv is not proper rectangle to detect.
#     # draw polygon
#     # cv.polylines(img, [cnt], True, (0, 0, 255), 2)

#     # get rectangle
#     # x & y is centre point of object.
#     rect = cv.minAreaRect(cnt)
#     (x, y), (w, h), angle = rect

#     # coords should be int.
#     # width & hight of objects by applying ratio. 
#     obj_w = w / pixel_cm_ratio
#     obj_h = h / pixel_cm_ratio

#     # draw rectangle around object instaed of poly line for measuring its dimension.
#     box = cv.boxPoints(rect)
#     box = np.int0(box)

#     # display centre point and box plot on object.
#     cv.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
#     cv.polylines(img, [box], True, (0, 0, 255), 2)
    
#     cv.putText(img, "width {} cm".format(round(obj_w, 1)), (int(x), int(y - 15)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
#     cv.putText(img, "hight {} cm".format(round(obj_h, 1)), (int(x + 10), int(y + 15)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    

# cv.imshow('img', img)
# k = cv.waitKey(0)
# if k == 27:
#     cv.destroyAllWindows()

################################################################################################################################################################



############################################################### DICT OPERATION ON IMG SAT VALUE ################################################################

# coating_thickness_table = {
#     0.33 : 120,
#     0.43 : 156,
#     0.55 : 180,
#     0.60 : 204,
#     0.66 : 215
# }

# def getThickness(dict, sat_val):
#     for key, value in dict.items():
#         if value == sat_val:
#             return key
        
# coat_thick = getThickness(coating_thickness, 156)
# print(coat_thick)

# def get_coating_sat_measure():
#     pass

# def estimateCoatingThickness(img_coating, coating_thickness_table):
#     sat_value = get_coating_sat_measure(img_coating)
#     for key, value in coating_thickness_table.items():
#         if value == sat_value:
#             return key



################################################################################################################################################################



############################################################### COLOR BASED OBJECT DETECTION ###################################################################

# img = cv.imread('data/golf.jpg')
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# lower_limit = np.array([43, 109, 0])
# upper_limit = np.array([70, 255, 255])

# mask = cv.inRange(hsv, lower_limit, upper_limit)
# bbox = cv.boundingRect(mask)
# if bbox is not None:
#     print('object detected')
#     x, y, w, h = bbox
#     cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
# else:
#     print('object not detected')

# cv.imshow('image', img)
# cv.waitKey(0)    

################################################################################################################################################################



############################################################### FINDING CONTOURS ###############################################################################

# def nothing(x):
#     pass

# # def threash_callbk(val):
# #     threshold = val

# source = 'Source'
# cv.namedWindow(source)
# cv.resizeWindow(source, 360, 240)
# thresh = 100
# max_thresh = 255
# cv.createTrackbar('Canny Threshbar', source, thresh, max_thresh, nothing)
# # threash_callbk(thresh)

# while True:
#     # Read source image.
#     # img = cv.imread('data/family.jpg')
#     # img = cv.imread('data/pcb_coating.jpeg')
#     # img = cv.resize(img, (640, 555))
#     # img = cv.imread('data/pcb_coating1.jpg')
#     # img = cv.resize(img, (640, 320))
#     # img = cv.imread('data/pcb_coating2.jpg')
#     img = cv.imread('data/pcb_coating3.jpeg')
#     img = cv.resize(img, (360, 620))

#     # convert image to gray and blur it.
#     grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     blurImag = cv.blur(grayImg, (3,3))

#     # detect edges using canny
#     th = cv.getTrackbarPos('Canny Threshbar', source)
#     # cannyImg = cv.Canny(blurImag, threshold, threshold*2)
#     cannyImg = cv.Canny(blurImag, th, th*2)

#     # find contours
#     contours, hierarchy = cv.findContours(cannyImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     # draw contours
#     drawing = np.zeros((cannyImg.shape[0], cannyImg.shape[1], 3), dtype = np.uint8)
#     for i in range(len(contours)):
#         cv.drawContours(drawing, contours, i, (0, 0, 255), 1, cv.LINE_8, hierarchy, 0)

#     cv.imshow("contours", drawing)
#     cv.imshow('original', img)
#     k = cv.waitKey(0)
#     if k == 27:
#         break

# cv.destroyAllWindows()

################################################################################################################################################################


############################################################### CONVEX HULL CONTOUR ############################################################################


# def nothing(x):
#     pass

# cv.namedWindow('Tracking')
# cv.resizeWindow('Tracking', 360, 240)
# cv.createTrackbar('canny_thresh', 'Tracking', 0, 255, nothing)

# while True:
#     img = cv.imread('data/img6.jpg')
#     img = cv.resize(img, (640, 480))
#     grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     blurImg = cv.blur(grayImg, (3,3))
#     th = cv.getTrackbarPos('canny_thresh', 'Tracking')
#     cannyImg = cv.Canny(blurImg, th, th*2)
#     contours, hierarchy = cv.findContours(cannyImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     hull_list = []
#     for i in range(len(contours)):
#         hull = cv.convexHull(contours[i])
#         hull_list.append(hull)

#     drawing = np.zeros((cannyImg.shape[0], cannyImg.shape[1], 3), dtype=np.uint8)
#     for i in range(len(contours)):
#         cv.drawContours(drawing, contours, i, (0, 255, 0))
#         cv.drawContours(drawing, hull_list, i, (255, 0, 0))

#     cv.imshow("contours", drawing)
#     k = cv.waitKey(0)
#     if k == 27:
#         break

# cv.destroyAllWindows()
    

################################################################################################################################################################


############################################################### ADD TWO IMAGES #################################################################################

# img1 = cv.imread('data/family.jpg')
# img2 = cv.imread('data/img5.jpg')
# dst = cv.addWeighted(img1, 0.5, img2, 0.5, 0)
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

################################################################################################################################################################


############################################################### IMG COATING #################################################################################

# img = cv.imread('data/pcb_coating.jpeg')
# grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('grayImage', grayImg)
# cv.waitKey(0)
# cv.destroyAllWindows()

################################################################################################################################################################


############################################################### GESTURE CONTROL ################################################################################

# cap = cv.VideoCapture(0)
# # addr = "https://192.168.1.33:8080/video"
# # cap.open(addr)
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# pTime = 0
# cTime = 0


# while True:
    
#     ret, frame = cap.read()

#     img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     results = hands.process(img)
#     # print(results.multi_hand_landmarks)

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 # print(id, lm)
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # print(id, cx, cy)
#                 if id == 0:
#                     cv.circle(frame, (cx, cy), 10, (0, 255, 0), cv.FILLED)
#                 if id == 4 or id == 8:
#                     cv.circle(frame, (cx, cy), 15, (0, 255, 0), cv.FILLED)
#             mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)


#     cv.imshow('image', frame)
#     k = cv.waitKey(1)
#     if k == 27:
#         break

# cv.destroyAllWindows()

################################################################################################################################################################


############################################################### HANDLE MOUSE EVENT #############################################################################

# dir have show all the classes and member function inside cv2 package.
# events = [i for i in dir(cv) if 'EVENT' in i]
# events = [i for i in dir(cv)]
# # print(events)

# # # write al; events into text file.
# with open('mouse_events.txt', 'w') as f:
#     for i in events:
#         f.write(i)
#         f.write('\n')

# font = cv.FONT_HERSHEY_SIMPLEX
# prevX, prevY = -1, -1

# # mouse callback event
# def click_event(event, x, y, flags, params):
#     global prevX, prevY
#     if event == cv.EVENT_LBUTTONDOWN:
#         strXY = str(x) + ', ' + str(y)
#         cv.putText(img, strXY, (x, y), font, 0.5, (0, 0, 255), 1)
#         cv.imshow("image", img)

#     if event == cv.EVENT_RBUTTONDOWN:
#         blue = img[y, x, 0]
#         green = img[y, x, 1]
#         red = img[y, x, 2]
#         strColor = str(blue) + ', ' + str(green) + ', ' + str(red)
#         cv.putText(img, strColor, (x, y), font, .5, (0, 255, 0), 1)
#         cv.imshow("image", img)

#     # if event == cv.EVENT_MOUSEMOVE:
#     #     if prevX == -1 and prevY == -1:
#     #         prevX, prevY = x, y
#     #     else:
#     #         cv.line(img, (prevX, prevY), (x, y), (255, 255, 0), 2)
#     #         prevX, prevY = -1, -1
#     #     cv.imshow("image", img)
        

# img = np.zeros((512, 512, 3), np.uint8)
# # img = np.full((512, 512, 3), (255, 120, 0), dtype=np.uint8)
# # img = cv.imread('data/987010_TOP.bmp')
# # img = cv.resize(img, (1280, 720))
# # img = cv.imread('data/987010_T_BOT_CP.bmp')
# # img = cv.resize(img, (1280, 720))
# cv.imshow('image', img)
# cv.setMouseCallback('image', click_event)
# cv.waitKey(0)
# cv.destroyAllWindows()

################################################################################################################################################################


############################################################### MASK IMAGE WITH CUSTOM COORDINATES #############################################################

# circles = np.zeros((4,2), np.int)
# counter = 0

# def mouseEvent(event, x, y, flags, params):
#     global counter
#     if event == cv.EVENT_LBUTTONDOWN:
#         circles[counter] = x, y
#         counter += 1
#         print(circles)

# # img = np.zeros((512, 512, 3), np.int8)
# # img = cv.imread('data/987010_T_BOT_CP.bmp')
# # img = cv.resize(img, (1280, 720))
# img = cv.imread('data/img4.jpg')
# while True:
#     if counter == 4:
#         width, height = 640, 480
#         pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
#         pts2 = np.float32([[0,0], [width,0], [0,height], [width, height]])
#         matrix = cv.getPerspectiveTransform(pts1, pts2)
#         imgOut = cv.warpPerspective(img, matrix, (width, height))
#         cv.imshow('output image', imgOut)

#     for x in range(0, 4):
#         cv.circle(img, (circles[x][0], circles[x][1]), 5, (255, 0, 0), cv.FILLED)

#     cv.imshow('image',img)
#     cv.setMouseCallback('image', mouseEvent)
#     k = cv.waitKey(1)
#     if k == 27:
#         break

# cv.destroyAllWindows()


################################################################################################################################################################



############################################################### COATING THICKNESS. #############################################################################


# # Thickness config:
# Tmin = 0.33
# Tmax = 0.66

# # Color config:
# Cmin  = 0
# Cmax = 221

 

# def mouseEventCallback(event, x, y, flags, params):
#     if event == cv.EVENT_LBUTTONDOWN:
#         blue = img[y, x, 0]
#         green = img[y, x, 1]
#         red = img[y, x, 2]

#         if blue > 200:
#             # Color % calculation:
#             CP = (green - Cmin) / (Cmax - Cmin)
#             # where Cvalue is Color value for the selected pixel

#             # Thickness calculation:
#             TV = ((Tmax - Tmin) * CP) + Tmin
#             TV = round(TV, 2)
#             # where TV is the selected pixel thickness.
#             cv.putText(img, f'depth: {TV}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 221, 254), 2)
#             cv.imshow("image", img)
#         else:
#             TV = 0
#             cv.putText(img, f'depth: {TV}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 221, 254), 2)
#             cv.imshow("image", img)

    
#     if event == cv.EVENT_RBUTTONDOWN:
#         blue = img[y, x, 0]
#         green = img[y, x, 1]
#         red = img[y, x, 2]
#         strColor = str(blue) + ', ' + str(green) + ', ' + str(red)
#         cv.putText(img, strColor, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         cv.imshow("image", img)


# # img = cv.imread('data/blue.jpg')
# # img = cv.imread('data/pcb.bmp')
# # img = cv.resize(img, (720, 720))
# img = cv.imread('data/coatingTest1.jpg')
# # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow("image", img)
# cv.setMouseCallback("image", mouseEventCallback)
# cv.waitKey(0)
# cv.destroyAllWindows()

################################################################################################################################################################


############################################################### GEEKS FOR GEEKS ################################################################################


############### IMAGE READ

# img = cv.imread("data/blue.jpg", cv.IMREAD_COLOR)
# cv.imshow("image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# img = cv.imread("data/blue.jpg")
# imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# plt.imshow(imgRGB)
# plt.waitforbuttonpress()
# plt.close('all')


# img = cv.imread("data/blue.jpg", cv.IMREAD_GRAYSCALE)
# cv.imshow("image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

################# FINDING COUNTOURS
# img = cv.imread("data/slide.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 200)
# contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# print("contours: ", len(contours))
# cv.drawContours(img, contours, -1, (255, 255, 0), 2)

# cv.imshow('Image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()


################################################################################################################################################################


############################################################## CROP IMAGES #####################################################################################

# img = cv.imread('data/blue.jpg')
# crop = img[200:400, 300:600]
# cv.imshow("crop", crop)
# cv.waitKey(0)
# cv.destroyAllWindows()

################################################################################################################################################################



############################################################### DRAWING FUNCTIONS ##############################################################################

# img = np.zeros((512, 512, 3), np.uint8)
# cv.line(img, (0,0), (512, 512), (0, 0, 255), 3)
# cv.line(img, (0,512), (512, 0), (0, 0, 255), 3)
# cv.rectangle(img, (128, 128), (384, 384), (0, 255, 0), 2)
# cv.circle(img, (256, 256), 120, (255,0,0), 2)
# cv.ellipse(img, (256, 256), (80, 20), 0, 0, 360, (255, 255, 0), 2)
# cv.ellipse(img, (256, 256), (20, 80), 0, 0, 360, (255, 255, 0), 2)
# cv.ellipse(img, (256, 256), (20, 80), 45, 0, 360, (255, 255, 0), 2)
# cv.ellipse(img, (256, 256), (20, 80), 135, 0, 360, (255, 255, 0), 2)
# cv.circle(img, (256, 345), 1, (255,255,0), 2)
# cv.circle(img, (256, 167), 1, (255,255,0), 2)
# cv.circle(img, (345, 256), 1, (255,255,0), 2)
# cv.circle(img, (167, 256), 1, (255,255,0), 2)
# cv.circle(img, (318, 318), 1, (255,255,0), 2)
# cv.circle(img, (192, 192), 1, (255,255,0), 2)
# cv.circle(img, (192, 319), 1, (255,255,0), 2)
# cv.circle(img, (318, 193), 1, (255,255,0), 2)
# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img,'DOCTOR STRANGE SPELL',(70,490), font, 1,(255,255,255),2,cv.LINE_AA)
# cv.imshow("image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

################################################################################################################################################################



############################################################### MASK IMAGE WITH CUSTOM COORDINATES##############################################################

ig = cv.imread('data/pcb.bmp')
ig = cv.resize(ig, (700,700))
b, g, r = cv.split(ig)
cv.imshow("blue image", r)
cv.waitKey(0)
cv.destroyAllWindows()

################################################################################################################################################################


############################################################### MASK IMAGE WITH CUSTOM COORDINATES##############################################################
################################################################################################################################################################
############################################################### MASK IMAGE WITH CUSTOM COORDINATES##############################################################
################################################################################################################################################################
############################################################### MASK IMAGE WITH CUSTOM COORDINATES##############################################################
################################################################################################################################################################
############################################################### MASK IMAGE WITH CUSTOM COORDINATES##############################################################
################################################################################################################################################################