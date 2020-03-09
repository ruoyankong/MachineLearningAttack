#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import random
import numpy as np
import math
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ### Collect data
# 
# place your data at the same floder with the github project

# In[2]:


# input path of file
# return data extract from pickle
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# examples of how to get data from the file
# data stores a collection of images in flat mode
# label stores the labels corresponding to each image
DIR = "../../cifar-10-batches-py"
FILE_LIST = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

rawdata = unpickle(DIR + "/" + FILE_LIST[0])

data = rawdata[b'data']
labels = rawdata[b'labels']
batch_label = rawdata[b'batch_label']
filenames = rawdata[b'filenames']


# ### transform the raw data into image
# below gives an example of how to extract the first image from the data
# 
# 
# change index to get more

# In[5]:


# input 2D data
# return BGR image
def flatToImg(flat):
    img = np.reshape(flat, (3, 32, 32))
    img = np.transpose(img, (1, 2, 0))
    return img

# read and show image
flat = data[0]
# img = flatToImg(flat)
# print(img.shape)
# plt.imshow(img)
# plt.show()


# ### extract secret from image and decode

# In[12]:


# input: image
# return bits of image in one-dimention like
def getSecret(image):
    return np.unpackbits(image.flatten())
  
# image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
# getSecret(image)


# In[13]:


# input: secret data s(binary), bool value
# return decimal data, if ture return a BGR image, else 2D value
def decode(s, toImg=False):
    if not toImg:
        return np.packbits(s)
    else:
        return flatToImg(np.packbits(s))

    
# s = getSecret(image)
# img = decode(s, True)
# plt.imshow(img)
# plt.show()


# ## synthesizing malicious data

# In[8]:


def GenData(d1, d2):
    randVal = random.randint(0,255)
    image = np.zeros([d1, d2])
    idx1 = random.randint(0, d1-1)
    idx2 = random.randint(0, d2-1)
    image[idx1, idx2] = randVal
    return image

# test
# x = GenData(5, 5)
# print(x)
# print(x[:2].flatten)


# In[ ]:


def bitToLabel(bits, label_num):
    x = math.ceil(len(bits)/label_num)
    label = [0]*x
    for i in range(x):
        label[i] = int(bits[i*label_num:(i+1)*label_num], 2)
    return label
        
# bits = "0011010011"
# bitToLabel(bits, 3)
    

