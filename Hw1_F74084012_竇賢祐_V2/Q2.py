import numpy as np
import cv2

def Q21():
    img = cv2.imread("Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg")
    cv2.imshow("original", img)
    blur_img = cv2.GaussianBlur(img, (5,5), 0)
    cv2.imshow('Blurred Image', blur_img)

def Q22():
    img = cv2.imread("Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg")
    bilateral_blur = cv2.bilateralFilter(img, 9, 90, 90)
    cv2.imshow("original", img)
    cv2.imshow("blurred", bilateral_blur)

def Q23():
    img = cv2.imread("Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg")
    blurred3 = cv2.medianBlur(img, 3)
    blurred5 = cv2.medianBlur(img, 5)
    cv2.imshow("blurred 3", blurred3)
    cv2.imshow("blurred 5", blurred5)