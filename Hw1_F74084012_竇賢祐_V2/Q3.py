import numpy as np
import cv2
import math
height_G = 0
width_G = 0
gaussian_img = np.zeros((10,10))
sobelX_img = np.zeros((10,10))
sobelY_img = np.zeros((10,10))
def Q31():
    global gaussian_img , width_G, height_G
    img = cv2.imread("Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape
    height_G = height
    width_G = width

    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()

    temp = np.zeros((height+2,width+2))
    temp[1:height+1,1:width+1] = img_gray
    ans = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            for k in range(3):
                for l in range(3):
                    ans[i][j] += gaussian_kernel[k][l] * temp[i+k][j+l]

    ans = ans.astype("uint8")
    gaussian_img = ans
    cv2.imshow("Grayscale", img_gray)
    cv2.imshow("Gaussian Blur",ans)

def Q32():
    global gaussian_img, width_G, height_G, sobelX_img
    if (gaussian_img[0][0] == 0):
        img = cv2.imread("Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape

        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        temp = np.zeros((height + 2, width + 2))
        temp[1:height + 1, 1:width + 1] = img_gray
        ans = np.zeros((height, width))


        for i in range(height):
            for j in range(width):
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        ans[i][j] += gaussian_kernel[k+1][l+1] * temp[1 + i + k][1 + j + l]
    else :
        ans = gaussian_img
        height = height_G
        width = width_G
    temp2 = np.zeros((height + 2, width + 2))
    temp2[1:height + 1, 1:width + 1] = ans
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    final = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            for k in range(-1,2):
                for l in range(-1,2):
                    final[i][j] += sobelx[1 + k][1 + l] * temp2[1 + i + k][1 + j + l]
            # final[i][j] = sqr(final[i][j]**2)
            final[i][j] = math.sqrt(final[i][j]**2)
            if final[i][j] > 255:
                final[i][j] = 255
            elif final[i][j] < 0:
                final[i][j] = 0

    final = final.astype("uint8")
    sobelX_img = final
    cv2.imshow("sobelX",final)

def Q33():
    global width_G, height_G, gaussian_img, sobelY_img
    if gaussian_img[0][0] == 0:
        img = cv2.imread("Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape

        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        temp = np.zeros((height + 2, width + 2))
        temp[1:height + 1, 1:width + 1] = img_gray
        ans = np.zeros((height, width))


        for i in range(height):
            for j in range(width):
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        ans[i][j] += gaussian_kernel[k+1][l+1] * temp[1 + i + k][1 + j + l]
    else:
        ans = gaussian_img
        height = height_G
        width = width_G
    temp2 = np.zeros((height + 2, width + 2))
    temp2[1:height + 1, 1:width + 1] = ans
    sobelx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    final = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            for k in range(-1,2):
                for l in range(-1,2):
                    final[i][j] += sobelx[1 + k][1 + l] * temp2[1 + i + k][1 + j + l]
            # final[i][j] = sqr(final[i][j]**2)
            final[i][j] = math.sqrt(final[i][j]**2)
            if final[i][j] > 255:
                final[i][j] = 255
            elif final[i][j] < 0:
                final[i][j] = 0

    final = final.astype("uint8")
    sobelY_img = final
    cv2.imshow("sobelY",final)

def Q34(): 
    global sobelY_img, sobelX_img, width_G,height_G
    if sobelY_img[0][0] != 0 and sobelX_img[0][0] != 0:
        height = height_G
        width = width_G
        ans = np.zeros((height,width))
        for i in range(height):
            for j in range(width):
                ans[i][j] =math.sqrt(sobelX_img[i][j] **2 + sobelY_img[i][j]**2)
                if ans[i][j] >255:
                    ans[i][j] = 255
                elif ans[i][j] < 0 :
                    ans[i][j] = 0
        ans = ans.astype("uint8")
        cv2.imshow("Magnitude", ans)