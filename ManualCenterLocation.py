# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:54:39 2022

1st Script on the Oocyte quality estimator - locate the oocyte center, manually.

@author: Matilde
"""

#%% imported libraries 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.io import imread, imshow
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.transform import warp_polar
from skimage.util import img_as_float

#%% User's oocyte center location
# This function checks if the user clicks in their mouse over the image and
# saves the coordinates of the click in the global variable 'center coordinates'.

center_coordinates = []
doubt = []

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
 
    global center_coordinates
    
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print('cordinates of the center:  ', x, ' ', y)
        
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, ".  "+ str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)       
        center_coordinates.append([x,y])
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print('cordinates of the center:  ', x, ' ', y)
        
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, ".  "+ str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)       
        center_coordinates.append([x,y])
    
        
        
#%% main running
# Opens a pop up with image and the user have to point and click in the center 
# of the oocyte. Then, save the same image in another folder with the label 
# of the center

# (This doesn't look to make sense, but it's done in this way because before I 
# was saving the polar image with the center coordinates in the label.
# This code can be updated to save the coordinates in the exel 
# file 'image info'.


# NOTE: this function runs for all the images that you have in the source path. 
if __name__=="__main__":
    
    source_path = "./database - immature/original images/"
    center_path = "./database - immature/images with center/"
    count = 0
    for path in os.listdir(source_path):
        
        print('Selection center image nº ', count)
        
        source_img_path = os.path.join(source_path, path)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('image', 600,600)
    
        # reading the image
        img = cv2.imread(source_img_path)
        img_copy = img.copy()
 
        # displaying the original image
        cv2.imshow('image', img)
 
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', click_event)
 
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
	
        # close the window
        cv2.destroyAllWindows()
        
    
        # displaying the results 
        cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('image', 600,600)
        cv2.imshow('image', img)
        
        cv2.imwrite(center_path +  "center - " + str(center_coordinates[count][0]) + " , " + str(center_coordinates[count][1]) + " -" + path,img_copy)
        
        count = count + 1
   
  



    

