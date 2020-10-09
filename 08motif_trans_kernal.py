#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:25:53 2020

@author: ls
"""

import tensorflow as tf
import numpy as np
import sys
model_file=sys.argv[1]#'/home/ls/deepfiber_new/elongation_result/model0.h5'
motif_file=sys.argv[2]#'/home/ls/deepfiber_new/kernal_motif_model0.txt'
list_kernal=sys.argv[3:]#int for index of kernal
model=tf.keras.models.load_model(model_file)
tmp=model.get_layer(index=0).get_weights()
bias=tmp[1]
tmp0=tmp[0][:,:,0,:]

def trans_kernal_motif(standard_kernal):   
    total,temp=[],[]
    for col in range(standard_kernal.shape[1]):
        for row in range(standard_kernal.shape[0]-1):
            if standard_kernal[row,col] >0:    
                temp.append(standard_kernal[row,col])
            else:
                temp.append(0)
        if np.array(temp).sum()==0:
            temp=[0.25,0.25,0.25,0.25]
        else:
            temp=[round(i/(np.array(temp).sum()),2) for i in temp]
        total.append(temp)
        temp=[]
    total=np.array(total)   
    total_trans=np.array([total[:,0],total[:,2],total[:,3],total[:,1]]).T
    return total_trans
with open(motif_file,'w') as file:  
    for i in [int(i) for i in list_kernal]:
        kernal=tmp0[:,:,i]
        standard_kernal=kernal-np.array([kernal[4,:],kernal[4,:],kernal[4,:],kernal[4,:],kernal[4,:]])
        total_trans=trans_kernal_motif(standard_kernal)
        file.write('>'+'kernal%d'%i+'\n')
        for row in total_trans:
            file.write('\t'.join([str(round(j,2)) for j in row])+'\n')
        file.write('\n')
