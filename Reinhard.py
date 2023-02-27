# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:22:54 2022

Reinhard method - paper: 
Roy, Santanu, et al. "A study about color normalization methods for histopathology images." Micron 114 (2018): 42-61.
@author: Matilde
"""

from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#%% Functions
def read_images(source_path, target_path):
    source_img = io.imread(source_path)
    target_img = io.imread(target_path)
    return source_img, target_img

def convertRGB2LalphaBeta(img):
    return color.rgb2lab(img)

def colorTransfer(source_img, target_img, channels = 3):
    i = 0
    output_img = np.zeros_like(source_img)
    while i<3:
        output_img[:,:,i] = np.mean(target_img[:,:,i]) + (source_img[:,:,i] - np.mean(source_img[:,:,i]))*(np.std(target_img[:,:,i])/np.std(source_img[:,:,i]))
        i += 1
    return output_img
                                                  
def convertLalphaBeta2RGB(img):
    return color.lab2rgb(img)

#%% paths 
source_path = r"./database - immature/center cropped images/"
target_path = r"./database - immature/original images/21.11.png"
save_file = "./database - immature/center cropped color-normalized images/"

#%% 
for path in os.listdir(source_path):
    source_img_path = os.path.join(source_path, path)
    
    source_img_rgb, target_img_rgb = read_images(source_img_path, target_path)
   
    source_img = convertRGB2LalphaBeta(source_img_rgb)
    target_img = convertRGB2LalphaBeta(target_img_rgb)

    modified_output = colorTransfer(source_img, target_img)
    modified_output_rgb = convertLalphaBeta2RGB(modified_output)
    
    plt.imsave((save_file + "\\" + path ), modified_output_rgb)
    

