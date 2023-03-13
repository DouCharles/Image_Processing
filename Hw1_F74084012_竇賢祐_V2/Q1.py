import numpy as np
import cv2
def Q11():
    img = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg")

    cv2.imshow("Hw1-1",img)
    height, width, color = img.shape
    print("Height : {0}\nWidth : {1}".format(height,width))
def Q12():
    img = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg")
    B, G, R = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.imshow("B channel", cv2.merge([B,zeros, zeros]))
    cv2.imshow("G channel", cv2.merge([zeros, G, zeros]))
    cv2.imshow("R channel", cv2.merge([zeros, zeros, R]))
def Q13(): ## problem : no img
    img = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("OpenCv function", gray_img)
    B, G, R = cv2.split(img)
    my_img = (B/3+G/3+R/3)
    my_img = my_img.astype('uint8')
    cv2.imshow("Average weighted",my_img)


def update_img(x):
    #global img_weak, img_strong
    img_weak = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg")
    img_strong = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg")
    blender = cv2.getTrackbarPos('Blend', 'Blend')
    blender /= 255
    # 1 - blender,, blender
    img_weak = cv2.addWeighted(img_strong, 1-blender, img_weak, blender, 0)
    cv2.imshow("Blend", img_weak)

def Q14():
    #global img_weak, img_strong
    img_weak = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg")
    img_strong = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg")
#0~255 bar value
    cv2.namedWindow("Blend", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Blend',800,600)

    cv2.createTrackbar('Blend', 'Blend', 0, 255, update_img)
    cv2.setTrackbarPos('Blend', 'Blend', 50)
    img = cv2.addWeighted(img_strong, 205 / 255, img_weak, 50 / 255, 0)
    cv2.imshow("Blend", img)