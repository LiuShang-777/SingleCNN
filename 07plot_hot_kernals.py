#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:02:00 2020

@author: ls
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models  
import numpy as np
import sys
import matplotlib.pyplot as plt

#get input data
model_file='/home/ls/deepfiber_new/01cnn_result/scw_result/model4.h5'
exfile='/home/ls/deepfiber_new/03activated_kernals/scw/scw_ex4.npy'
nofile='/home/ls/deepfiber_new/03activated_kernals/scw/scw_no4.npy'
with open('/home/ls/deepfiber_new/02deeplift/input/scw/scw_utr54.fa','r') as file:
    list_name,list_seq,list_length=[],[],[]
    for line in file:
        line=line.strip()
        if line[0]=='>':
            list_name.append(line)
        else:
            list_seq.append(line)
            list_length.append(len(line)) 
def one_hotshot(array,classes):
    onehot=np.zeros((classes,array.shape[0]))
    for i in range(array.shape[0]):
        onehot[int(array[i]),i]=1
    return onehot
def tansfer_str_to_array(list_input):
    array=np.zeros((len(list_input),len(list_input[0])))
    for i in range(len(list_input)):
        for j in range(len(list_input[i])):
            if list_input[i][j]=='A':
                continue
            elif list_input[i][j]=='T':
                array[i,j]=1
            elif list_input[i][j]=='C':
                array[i,j]=2
            elif list_input[i][j]=='G':
                array[i,j]=3
            elif list_input[i][j]=='N':
                array[i,j]=4
    return array
init_array=tansfer_str_to_array(list_seq)                
list_result=[]
for i in range(init_array.shape[0]):
    list_result.append(one_hotshot(init_array[i,:],5))
list_result=np.array(list_result)
#get expression data
y_list=[]
for i in list_name:
    if 'aw' in i:
        y_list.append(0)
    elif 'ae' in i:
        y_list.append(1)
    else:
        break
def y_onehot(array,classes):
    narray=np.zeros((array.shape[0],classes))
    for i in range(len(array)):
        narray[i,array[i]]=1
    return narray

y_result=y_onehot(np.array(y_list),2)
list_result_idx=[i for i in range(len(list_name))]
list_result_idx=np.array(list_result_idx)
list_result=list_result[:,:,:,np.newaxis]

#get output of first conv layer
model=tf.keras.models.load_model(model_file)
tmp=model.get_layer(index=0).get_weights()
tmp=models.Model(model.input,model.get_layer(index=0).output)
tmp_output=tmp.predict(list_result)
ex,no=[],[]
for i,j in zip(y_list,tmp_output):
    if i==1:
        ex.append(j)
    else:
        no.append(j)
ex=np.array(ex)[:,0,:,:].mean(axis=0)
no=np.array(no)[:,0,:,:].mean(axis=0)
np.save(exfile,ex)
np.save(nofile,no)
plt.figure(figsize=(5,50))
for i in range(1,25):
    plt.subplot(24,1,i)
    ex_0=ex[:,i-1]
    no_0=no[:,i-1]
    plt.scatter(np.arange(989),ex_0,c='red',s=0.5)
    plt.scatter(np.arange(989),no_0,c='blue',s=0.5)
    plt.title('kernal%d'%(i-1))
    plt.legend(['expressed','unexpressed'])
plt.tight_layout()
plt.savefig('/home/ls/deepfiber_new/testelong1.png')
plt.clf()
