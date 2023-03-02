# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:10:55 2023

2nd script on the Oocyte quality estimator - image center cropping, using the pre-located oocyte center

@author: Matilde
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

source_path =  "./database - immature/images with center/"
resized_path = "./database - immature/center cropped images/"
df = pd.read_excel('./database - immature/images_info.xlsx', dtype = str)
IMAGE_SIZE = 256

for path in os.listdir(source_path):   
    oocyte = (path.rpartition('-')[2][:-4])
    center_str = path.rpartition(' -')[0][9:]
    x = int(center_str.partition(',')[0])
    y = int(center_str.partition(',')[2])
    df.loc[df['oocyte n°'] == oocyte,'center x'] = x
    df.loc[df['oocyte n°'] == oocyte,'center y'] = y
    if not pd.isna((path)):
        df.loc[df['oocyte n°'] == oocyte,'image_name'] = path
        
    original_image = cv2.imread(source_path + path)
    resized_image = original_image[ (y - int(IMAGE_SIZE/2)) : (y + int(IMAGE_SIZE/2)), (x - int(IMAGE_SIZE/2)) : (x + int(IMAGE_SIZE/2)) ]
    
    cv2.imwrite(resized_path + path,resized_image )
    





