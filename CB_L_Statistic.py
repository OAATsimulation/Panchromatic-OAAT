import numpy as np
import pandas as pd
import colour
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.stats import norm

def convert(rgb):
    return colour.convert(rgb, 'RGB', 'CIE Lab')*100

def rgb2xyz(img):
    for i, j in enumerate(img):
        img[i] = np.apply_along_axis(convert, axis=1, arr=j/255)
    return img

def calCB(test_list, thresh=40):
    root = tk.Tk()
    root.withdraw()
    curdir = filedialog.askdirectory()
    listdir = os.listdir(curdir)
    imgpaths = [i for i in listdir if i[-3:] == 'csv']

    for img1path in imgpaths:
        img1 = pd.read_csv(os.path.join(curdir, img1path), usecols=[2, 3, 4], skiprows=[0], header=None)
        img1 = np.array(img1).astype('float32')

        # 将图片rgb转换为Lab
        img1 = np.apply_along_axis(convert, axis=1, arr=img1 / 255)

        # 以亮度作为条件分割背景和字母
        L = img1[:, 0]
        index = np.where(L < thresh)
        img1[index] = 0
        pix_num = len(L) - len(index[0])

        # 计算字母的平均Lab值
        avgLab = np.array([img1[:, 0].sum() / pix_num, img1[:, 1].sum() / pix_num, img1[:, 2].sum() / pix_num])
        avgLab = avgLab.reshape((1, 3))

        # 用字母的平均Lab值计算所有RGB值的色差
        CB = np.sqrt(np.sum(np.square(test_list - avgLab), axis=1))

        mean = np.mean(CB)
        sigma = np.std(CB)

        n, bins, patches = plt.hist(CB, 40, (0, 200), density=True, edgecolor='black', facecolor='white')
        norm_y = norm.pdf(bins, mean, sigma)

        plt.plot(bins, norm_y, 'r--')
        plt.xlabel('delta E')
        plt.ylabel('Num')
        plt.annotate('mean= {0:.2f}'.format(mean), xycoords='axes fraction', xy=(0.8, 0.95))
        plt.annotate('sigma= {0:.2f}'.format(sigma), xycoords='axes fraction', xy=(0.8, 0.9))
        plt.savefig(img1path[:-4] + '.png', dpi=300)
        plt.close()

def calL(test_list,thresh):
    root = tk.Tk()
    root.withdraw()
    curdir = filedialog.askdirectory()
    listdir = os.listdir(curdir)
    imgpaths = [i for i in listdir if i[-3:] == 'csv']

    for img1path in imgpaths:
        img1 = pd.read_csv(os.path.join(curdir, img1path), usecols=[2, 3, 4], skiprows=[0], header=None)
        img1 = np.array(img1).astype('float32')

        # 将图片rgb转换为Lab
        img1 = np.apply_along_axis(convert, axis=1, arr=img1 / 255)

        # 以亮度作为条件分割背景和字母
        L = img1[:, 0]
        index = np.where(L < thresh)
        L[index] = 0
        pix_num = len(L) - len(index[0])

        # 计算字母的平均Lab值，提取所有Lab的L
        avgL = L.sum()/pix_num
        RGBL = test_list[:,0]

        # 用字母的平均L值计算所有RGB值的亮度差
        deltaL = RGBL-avgL

        mean = np.mean(deltaL)
        sigma = np.std(deltaL)

        n, bins, patches = plt.hist(deltaL, 25, (-25, 75), density=True, edgecolor='black', facecolor='white')
        norm_y = norm.pdf(bins, mean, sigma)

        plt.plot(bins, norm_y, 'r--')
        plt.xlabel('delta E')
        plt.ylabel('Num')
        plt.annotate('mean= {0:.2f}'.format(mean), xycoords='axes fraction', xy=(0.8, 0.95))
        plt.annotate('sigma= {0:.2f}'.format(sigma), xycoords='axes fraction', xy=(0.8, 0.9))
        plt.savefig(img1path[:-4] + '.png', dpi=300)
        plt.close()
# 创建包含所有RGB的list
test_list = pd.read_csv('rgbdata.csv',header=None).to_numpy()

# 色差计算和统计
calCB(test_list)

# 亮度差计算和统计
# calL(test_list,thresh=40)

