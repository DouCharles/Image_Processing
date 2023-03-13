import cv2
import numpy as np
img = cv2.imread("Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png")
def Q41():
    global img
    img= cv2.resize(img,(256,256))

    cv2.imshow("img", img)

def Q42():
    global img
    shift = np.float32([[1,0,0],[0,1,60]])
    #rows, cols = img.shape[:2]
    res = cv2.warpAffine(img,shift,(400, 300))
    img = res
    cv2.imshow("translate",img)
def Q43():
    global img
    rows , cols = img.shape[:2]
    rotate = cv2.getRotationMatrix2D((rows/2, cols/2),10,0.5)
    res = cv2.warpAffine(img,rotate,(400,300))
    img = res
    cv2.imshow("rotate", img)

def Q44():
    global img
    old = np.float32([[50,50],[200,50],[50,200]])
    new = np.float32([[10,100],[200,50],[100,250]])
    t = cv2.getAffineTransform(old,new)
    res = cv2.warpAffine(img,t,(400,300))
    img = res
    cv2.imshow("Result" , img)

