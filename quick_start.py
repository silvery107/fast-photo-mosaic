import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from photo_mosaic import *

# Declear types
types = ["manmade","natural"]

# Data augmentation for fast maching and better performance (optional)
for type in types:
    data_augmentation(type)

# Load target images
target_img_pth = os.path.join(COM_pth,"mao1.jpg")
img_target = cv2.imread(target_img_pth)

# Set tile size
tiles = (32,32)

# Composite
img_composited = mosaic(target_img_pth,tiles,types)

# Show the result with comparation
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(img_composited[:,:,::-1])
plt.subplot(1,2,2)
plt.imshow(img_target[:,:,::-1])
plt.show()