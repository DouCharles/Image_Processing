import cv2
import time
import numpy as np
"""
10404 cat
666 cat
11702 dog
完全 None image

Q5
有的matrix 裡面有值是none 
"""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
pic_ar=[]
pic_or= []
corners_ar = []
objpoints = []
imgpoints = []
#In_mtx= []
objp = np.zeros((1, 11*8, 3), np.float32)
objp[0,:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
prev_img_shape = None
Extrinsic = []

for i in range(15):
    pic_ar.append(cv2.imread("./Dataset_OpenCvDl_Hw2/Q2_Image/" + str(i + 1) + ".bmp"))
    pic_or.append(cv2.imread("./Dataset_OpenCvDl_Hw2/Q2_Image/" + str(i + 1) + ".bmp"))
    gray = cv2.cvtColor(pic_ar[i], cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (11, 8),flags=cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if (ret == True):
        objpoints.append(objp)
        corner2 = cv2.cornerSubPix(gray, corners,(11,8), (-1, -1), criteria)
        imgpoints.append(corner2)
        corners_ar.append(corners)
        img = cv2.drawChessboardCorners(pic_ar[i], (11, 8), corners, ret)

        h,w = img.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#In_mtx.append(mtx)
        #print(mtx)


for j in range(len(rvecs)):
    r = cv2.Rodrigues(rvecs[j])
    a = tvecs[j]
    Extrinsic.append(np.concatenate((r[0],a),axis = 1))

def Q21():
    for i in range(15):
        pic = cv2.resize(pic_ar[i], (800, 800))
        cv2.imshow("pic", pic)
        cv2.waitKey(500)
        #time.sleep(1)

def Q22():
    print("Intrsic Matrix:")
    #for i in range(len(mtx)):
    print(mtx)
def Q23(index):
    if (index == ""):
        return
    print("Extrinsic Matrix")
    num = int(float(index))-1
    if num < 0:
        num = 0
    elif num > 14:
        num = 14
    print(Extrinsic[num])
    return

def Q24():
    print("Distortion Matrix")
    print(dist)
    return

def Q25():
    for img in pic_or:
        distortion = cv2.undistort(img, mtx, dist, None, mtx)
        img = cv2.resize(img,(800,800))
        distortion = cv2.resize(distortion,(800,800))
        cv2.imshow("distorted",img)
        cv2.imshow("undistorted", distortion)
        cv2.waitKey(500)
    return


