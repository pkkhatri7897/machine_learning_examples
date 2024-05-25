import cv2
import numpy as np
import matplotlib.pyplot as plt

####################################### img manupulation ##########################################

# # flags: 1-color, 0:grey scale, -1:nochange with alpha channel
# img = cv2.imread('lena.jpg', -1)
# print(img)

# # showing and distroy image
# cv2.imshow('image', img)
# key = cv2.waitKey(5000)
# if key == ord('q'):
#     cv2.destroyAllWindows()
# elif key == ord('s'):
#     cv2.imwrite('lena_copy.png', img)
#     cv2.destroyAllWindows()

#####################################################################################################

##################################### video capture #################################################

# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# out  = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
# print(cap.isOpened())

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         out.write(frame)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
#         cv2.imshow('frame', gray)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

######################################################################################################

################################## draw geometric shapes #############################################

# img = cv2.imread('lena.jpg', -1)
# img = np.zeros([512, 512, 3], np.uint8)

# draw line
# img = cv2.line(img, (0,0), (255,255), (0,0,255), 5)
# img = cv2.arrowedLine(img, (0,0), (255,255), (0,255,0), 3)
# img = cv2.arrowedLine(img, (0,280), (255,255), (0,255,0), 3)
# img = cv2.rectangle(img, (384,0), (510,128), (255,0,0), 2)
# img = cv2.circle(img, (270,263), 20, (255,0,255), 2)
# img = cv2.circle(img, (335,263), 20, (255,0,255), 2)
# font = cv2.FONT_HERSHEY_COMPLEX
# img = cv2.putText(img, 'lena', (320, 50), font, 2, (255,255,0), 3, cv2.LINE_AA)


# cv2.imshow('image', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######################################################################################################

###################################### setting camera parameter #######################################

# cap = cv2.VideoCapture(0)
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # 3 - cv2.CAP_PROP_FRAME_HEIGHT and 4 - cv2.CAP_PROP_FRAME_WIDTH 
# cap.set(3,5000) 
# cap.set(4,5000)

# print(cap.get(3)) 
# print(cap.get(4))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('frame', gray)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()      

########################################################################################################

####################################### Date & Time on video ###########################################

# import datetime

# cap = cv2.VideoCapture(0)
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text = 'Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
#         date = str(datetime.datetime.now())
#         frame = cv2.putText(frame, date, (10,30), font, 0.5, (255,2500,0), 1, cv2.LINE_AA)
#         frame = cv2.rectangle(frame, (5,35), (267,12), (255,0,0), 1, cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

#######################################################################################################

######################################## Handle mouse events ##########################################

# events = [i for i in dir(cv2) if 'EVENT' in i]

# def click_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, ',', y)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strxy = str(x) + ',' + str(y)
#         cv2.putText(img, strxy, (x,y), font, 0.50, (255,255,0), 2)
#         cv2.imshow('image', img)
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         blue = img[y, x, 0]
#         green = img[y, x, 1]
#         red = img[y, x, 2]
#         font = cv2.FONT_HERSHEY_DUPLEX
#         strBGR = str(blue) + ',' + str(green) + ',' + str(red)
#         cv2.putText(img, strBGR, (x, y), font, 0.5, (0,255,255), 1)
#         cv2.imshow('image', img)
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img, (x, y), 2, (0,0,255), -1)
#         points.append((x,y))
#         if len(points) >= 2 :
#             cv2.line(img, points[-1], points[-2], (255,255,120), 2)
#         cv2.imshow('image', img)
#     if event == cv2.EVENT_LBUTTONDOWN:
#         blue = img[x,y,0]
#         green = img[x,y,1]
#         red = img[x,y,2]
#         cv2.circle(img, (x, y), 3, (255,255,120), -1)
#         mycolorImage = np.zeros((512,512,3), np.uint8)
#         mycolorImage[:] = [blue, green, red]

#         cv2.imshow('color', mycolorImage)
    

# img = np.zeros((512,512,3), np.uint8)
# # img = cv2.imread('lena.jpg')
# cv2.imshow('image', img)
# points = []
# cv2.setMouseCallback('image', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#################################################################################################

################################ Arithmatic and basic func on image #############################

# img = cv2.imread('messi5.jpg')
# img1 = cv2.imread('opencv-logo.png')

# print(img.shape) # return no of rows, columns, channels
# print(img.size) # return num of pixels is accessed
# print(img.dtype) # img datatype is obtained
# b,g,r = cv2.split(img)
# img = cv2.merge((b,g,r))

# ball = img[290:380, 330:390]
# img[283:335, 100:160] = ball

# # ball1 = img[337:388, 288:336]
# # img[65:70, 199:247] = ball1

# def click_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, ',', y)
#         font = cv2.FONT_HERSHEY_TRIPLEX
#         strxy = str(x) + ',' + str(y)
#         cv2.putText(img, strxy, (x, y), font, 1, (255,255,120), 2)
#         cv2.imshow('image', img)

# img = cv2.resize(img, (512,512))
# img1 = cv2.resize(img1, (512,512))
# # dst = cv2.add(img, img1)
# dst = cv2.addWeighted(img, 0.5, img1, 0.5, 0)

# cv2.imshow('image', dst)
# cv2.setMouseCallback('image', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

################################################################################################

###################################### Bitwise Operations ######################################

# img1 = np.zeros((250, 500, 3), np.uint8)
# img1 = cv2.rectangle(img1, (125,62), (375,187), (255,255,255), -1)
# img2 = cv2.imread('image_1.png')

# bitAnd = cv2.bitwise_and(img2, img1)
# bitOr = cv2.bitwise_or(img2, img1)
# bitXor = cv2.bitwise_xor(img2, img1)
# bitNot = cv2.bitwise_not(img2, img1)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('bitAnd', bitAnd)
# cv2.imshow('bitOr', bitOr)
# cv2.imshow('bitXor', bitXor)
# cv2.imshow('bitNot', bitNot)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

###################################################################################################

###################################### Bind Tracker ###############################################

# call back func when it call
# def nothing(x):
#     print(x)

# # img = np.zeros((300, 300, 3), np.uint8)
# cv2.namedWindow('image')

# # cv2.createTrackbar('B', 'image', 0, 255, nothing)
# # cv2.createTrackbar('G', 'image', 0, 255, nothing)
# # cv2.createTrackbar('R', 'image', 0, 255, nothing)
# cv2.createTrackbar('CP', 'image', 0, 400, nothing)
# # switch = '0 : OFF \n1 : ON'
# switch = 'Color/Gray'
# cv2.createTrackbar(switch, 'image', 0, 1, nothing)

# while(1):
#     img = cv2.imread('lena.jpg')
#     pos =  cv2.getTrackbarPos('CP', 'image')
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img, str(pos), (50, 150), font, 4, (255,0,0), 6)

#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('q'):
#         break
#     # b = cv2.getTrackbarPos('B', 'image')
#     # g = cv2.getTrackbarPos('G', 'image')
#     # r = cv2.getTrackbarPos('R', 'image')
#     s = cv2.getTrackbarPos(switch, 'image')
#     if s == 0:
#         pass
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     img = cv2.imshow('image', img)

# cv2.destroyAllWindows()

##################################################################################################

################################## Hue, Saturation & Value #######################################

def nothing(x):
    pass

capture = cv2.VideoCapture(0)
cv2.namedWindow('Tracking')
cv2.createTrackbar('LH', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('UH', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)

while True:
    # frame = cv2.imread('data/smarties.png')
    # convert into hsv
    _, frame = capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('LH', 'Tracking')
    l_s = cv2.getTrackbarPos('LS', 'Tracking')
    l_v = cv2.getTrackbarPos('LV', 'Tracking')

    u_h = cv2.getTrackbarPos('UH', 'Tracking')
    u_s = cv2.getTrackbarPos('US', 'Tracking')
    u_v = cv2.getTrackbarPos('UV', 'Tracking')

    l_b = np.array([l_h, l_s, l_v]) #blue lower color range
    u_b = np.array([u_h, u_s, u_v]) #blue upper color range
    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', res)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()

######################################################################################################

######################################## Image Sturation #############################################

# img = cv2.imread('data/gradient.png', 0)
# _, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# _, th2 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
# _, th3 = cv2.threshold(img, 150, 255, cv2.THRESH_TRUNC)
# _, th4 = cv2.threshold(img, 150, 255, cv2.THRESH_TRIANGLE)
# _, th5 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
# _, th6 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO_INV)

# cv2.imshow('image', img)
# cv2.imshow('th1', th1)
# cv2.imshow('th2', th2)
# cv2.imshow('th3', th3)
# cv2.imshow('th4', th4)
# cv2.imshow('th5', th5)
# cv2.imshow('th6', th6)

# cv2.waitKey(0) & 0xFF == ord('q')
# cv2.destroyAllWindows()

#########################################################################################################

######################################## Adaptive Thresolding ###########################################

# img = cv2.imread('data/sudoku.png', 0)
# _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# cv2.imshow('image', img)
# # cv2.imshow('th1', th1)
# cv2.imshow('th2', th2)
# cv2.imshow('th3', th3)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########################################################################################################

################################### matplotlib with opencv ###############################################

# import matplotlib.pyplot as plt

# # img = cv2.imread('data/lena.jpg', 1)

# # cv2.imshow('image', img)
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img1 = cv2.imread('data/gradient.png', 0)
# _, th1 = cv2.threshold(img1, 50, 255, cv2.THRESH_BINARY)
# _, th2 = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY_INV)
# _, th3 = cv2.threshold(img1, 200, 255, cv2.THRESH_TRUNC)
# _, th4 = cv2.threshold(img1, 200, 255, cv2.THRESH_TOZERO)
# _, th5 = cv2.threshold(img1, 200, 255, cv2.THRESH_TOZERO_INV)


# # plt.imshow(img)
# # Hide ticks at both axis
# # plt.xticks([]), plt.yticks([])
# # plt.show()
# titles = ['original', 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv']
# images = [img1, th1, th2, th3, th4, th5]
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

########################################################################################################

########################################### Morphological Transfomation ################################

# import matplotlib.pyplot as plt

# # img = cv2.imread('data/smarties.png', cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread('data/LinuxLogo.jpg', cv2.IMREAD_GRAYSCALE)
# # _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# kernal = np.ones([2,2], np.uint8)
# dilation = cv2.dilate(mask, kernal, iterations=2)
# erosion = cv2.erode(mask, kernal, iterations=2)
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal) # perform first erosion than dilation
# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal) # perform first dilation than erosion
# gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal) # perform first dilation than erosion
# topHat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal) # perform first dilation than erosion
# blackHat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernal) # perform first dilation than erosion

# # title = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'gradient', 'topHat', 'blackHat']
# # images = [img, mask, dilation, erosion, opening, closing, gradient, topHat, blackHat]
# title = ['mask', 'dilation', 'erosion', 'opening', 'closing', 'gradient', 'topHat', 'blackHat']
# images = [mask, dilation, erosion, opening, closing, gradient, topHat, blackHat]

# for i in range(8):
#     plt.subplot(3,3,i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(title[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()

############################################################################################################

####################################### Smoothing & Blurring ###############################################

# img = cv2.imread('data/opencv-logo.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# kernel = np.ones((5,5), np.float32)/25
# dst = cv2.filter2D(img, -1, kernel)

# titles = ['image', '2D-convo']
# images = [img, dst]

# for i in range(2):
#     plt.subplot(1, 2, i+1), plt.imshow(images[i], 'hsv')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
    
# plt.show()