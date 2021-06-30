import cv2
import numpy as np
import matplotlib.pyplot as plt
from photo_mosaic import *

with open('./manmade_training.txt') as manmade_training:
    dir_manmade = manmade_training.read().splitlines()
with open('./natural_training.txt') as natural_training:
    dir_natural = natural_training.read().splitlines()
with open('./resize_image.txt') as resize_image:
    dir_resize = resize_image.read().splitlines()
img_target = cv2.imread(dir_manmade[39])
# img_target = img_target[:,:,::-1]
tiles = (32,32)

# _,img_composite = get_composite(img_target,dir_resize,tiles)
# resize_source(dir_natural,refresh_txt=False)

img_composite = feather_image(img_target,img_target,alpha=1,mode='edge')
# plt.figure(figsize=(15,15))
# plt.subplot(1,2,1)
# plt.imshow(feather[:,:,::-1].astype('uint8'))
# plt.subplot(1,2,2)
# plt.imshow(img_target[:,:,::-1])
# plt.show()

img_composite = crop_image(img_composite,(656,480))
# img_blend = blend_grid(img_target,tiles)
# plt.figure(figsize=(15,15))
# plt.subplot(1,2,1)
# plt.imshow(img_blend[:,:,::-1].astype('uint8'))
# plt.subplot(1,2,2)
# plt.imshow(img_target[:,:,::-1])
# plt.show()

cv2.imshow('win1',img_composite)
cv2.waitKey(0)