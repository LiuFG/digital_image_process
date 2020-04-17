from skimage import io,exposure
import numpy as np
import cv2


def imadjust(img_mono, gamma, low_in=0, high_in=255, out_inverse=0):
    # img_mono单通道图像，[low_in,high_in]到[0,1]映射，out_inverse=1时为到[1,0]的映射
    # 将[low_in,high_in]以外值减掉，向量化
    image_trunc = (img_mono > high_in) * high_in + (img_mono <= high_in) * img_mono
    # cv2.imshow('temp', image_trunc)
    image_trunc = (image_trunc < low_in) * low_in + (image_trunc >= low_in) * image_trunc
    # 归一化0-1
    img_adjust = (image_trunc - image_trunc.min()) / float(image_trunc.max() - image_trunc.min())
    # 反转
    if out_inverse == 1:
        img_adjust = 1 - img_adjust
    # gamma变换
    img_adjust = (img_adjust**gamma)
    return img_adjust


img = cv2.imread('./images/gamma.PNG')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('src', img_gray)
img_ad = imadjust(img_gray, 1, 0, 255, 0)
cv2.imshow('after', img_ad)
cv2.waitKey(0)

