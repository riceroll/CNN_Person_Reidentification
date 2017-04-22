import os
import sys
caffe_root = '../'  
sys.path.insert(0, caffe_root + 'python')

import caffe

import numpy as np
import matplotlib.pyplot as plt
import json
import csv


caffe.set_device(1)
caffe.set_mode_gpu()


model_def = './deploy/RIDLT.prototxt'
model_weights = './models/RIDLT_iter_20000.caffemodel'
testing_set = './test.txt'
num_test = 118
length_library = 6817
length_query = 13318-6817
length_feature = 1024

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


testfile=file(testing_set,'rb')
imgs0=[]
imgs1=[]
feat0=[]
feat1=[]
for i in range(num_test):
    imgs0.append([])
    imgs1.append([])
    feat0.append([])
    feat1.append([])
    
for i in range(length_library):
    line=testfile.readline()
    imgs0[int(line.split()[1])].append(line.split()[0]) 

for i in range(length_query):
    line=testfile.readline()
    imgs1[int(line.split()[1])].append(line.split()[0])


feat7_0_per=[]
i=0
for per in imgs0:
    feat7_0_tmp=[]
    batch_size=len(imgs0[i])
    net.blobs['data'].reshape(batch_size, 3, 227, 227) 
    j=0
    for img in imgs0[i]:
        image = caffe.io.load_image(img)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[j] = transformed_image
        j+=1
    output = net.forward()
    for k in range(batch_size):
        feat7=net.blobs['fc7_m'].data[k]
        feat7=feat7.tolist()
        feat7_0_tmp.append(feat7)
    feat7_0_per.append(feat7_0_tmp) 
    i+=1
    print float(i)/num_test


feat7_1_per=[]
i=0
for per in imgs1:
    feat7_1_tmp=[]
    batch_size=len(imgs1[i])
    net.blobs['data'].reshape(batch_size, 3, 227, 227) 
    j=0
    for img in imgs1[i]:
        image = caffe.io.load_image(img)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[j] = transformed_image
        j+=1
    output = net.forward()
    for k in range(batch_size):
        feat7=net.blobs['fc7_m'].data[k]
        feat7=feat7.tolist()
        feat7_1_tmp.append(feat7)
    feat7_1_per.append(feat7_1_tmp)   
    i+=1
    print float(i)/num_test


feat7_0_ave=[]
for i in range(num_test):
    img_sum=[0.0 for j in range(length_feature)]
    img_sum=np.array(img_sum)
    img_num=len(feat7_0_per[i])
    t=0
    for img in feat7_0_per[i]:
        img=np.array(img)
        img_sum+=img
        t+=1    
    img_sum/=img_num
    img_sum=img_sum.tolist()
    feat7_0_ave.append(img_sum)


feat7_1_ave=[]
for i in range(num_test):
    img_sum=[0.0 for j in range(length_feature)]
    img_sum=np.array(img_sum)
    img_num=len(feat7_1_per[i])
    t=0
    for img in feat7_1_per[i]:
        img=np.array(img)
        img_sum+=img
        t+=1    
    img_sum/=img_num
    img_sum=img_sum.tolist()
    feat7_1_ave.append(img_sum)


import scipy.io as sio
sio.savemat('RIDLT_test_feature.mat',{'feat7_0_ave':feat7_0_ave,'feat7_1_ave':feat7_1_ave})
