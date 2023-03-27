# Image_Processing

## Introduction
課程作業<br>

## hw1
以OpenCV讀取圖片，並對圖片做不同的變換<br>
- Image Processing
    - Load Image : 載入圖片 並顯示
    - Color Seperation : 分出RGB三種色調
    - Color Transformation : 取灰階
    - Blending : 以Bar調整兩張圖片的透明度
- Image Smoothing : 使用blur算法or filter
    - Gaussian Blur 
    - Blateral Filter
    - Median Filter
- Edge Detection
    - Gaussian Blur : 強化邊角
    - Sobel X : 以鉛直線來偵測整張圖片內容
    - Sobel Y : 以水平線來偵測整張圖片內容
    - Magitude : 
- Transformation
    - Resize : 調整圖片大小
    - Translation : 圖片轉正
    - Rotation,Scaling : 圖片旋轉and放大縮小
    - Shearing : 圖片傾斜


## hw1_5
訓練model來辨識圖片內容<br>
資料集 : cifar10<br>

- show Train Images : 隨機選擇9張圖片做預測
- show HyperParameter : 印出learning rate 、 batch size 、 optimizer
- show Model Shortcut : 印出model的架構
- show Accuracy : 印出訓練、預測的準確度
- Test : 自行輸入一個數字，來指定資料集內的某張圖片並做預測


## 環境 packge
- cifar 10
- opencv-python 4.5.2.52
- PyQt5  5.15.1
- 
