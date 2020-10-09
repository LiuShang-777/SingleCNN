# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:59:22 2020

@author: liushang
"""

#this script receive fasta file and output model file and genes correctly predicted
import numpy as np
import sklearn as sk
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.utils import shuffle
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#get the original file
input_file=sys.argv[1]
output_dir=sys.argv[2]
#input_file='/home/ls/deepfiber/sample_init_utr5.fa'
#get sequence data
with open(input_file,'r') as file:
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
#divide two datasets
y_result,list_result,list_result_idx=shuffle(y_result,list_result,list_result_idx)
list_result=list_result[:,:,:,np.newaxis]
def split_array(x,y,idx,total,part):
    split_x=np.array_split(x,total,axis=0)
    split_y=np.array_split(y,total,axis=0)
    split_idx=np.array_split(idx,total,axis=0)
    test_x,test_y,test_idx=split_x[part],split_y[part],split_idx[part]
    split_xm=[]
    split_ym=[]
    split_idxm=[]
    for i in range(total):
        if i !=part:
            split_xm.append(split_x[i])
            split_ym.append(split_y[i])
            split_idxm.append(split_idx[i])
    conx,cony,conidx=split_xm[0],split_ym[0],split_idxm[0]
    for i,j,z in zip(split_xm[1:],split_ym[1:],split_idxm[1:]):
        conx=np.vstack((conx,i))
        cony=np.vstack((cony,j))
        conidx=np.hstack((conidx,z))
    conx,cony,conidx=shuffle(conx,cony,conidx)
    val_x=conx[:4000,:,:,:]
    val_y=cony[:4000]
    val_idx=conidx[:4000]
    train_x=conx[4000:,:,:,:]
    train_y=cony[4000:]
    train_idx=conidx[4000:]
    return (test_x,test_y,test_idx,train_x,train_y,train_idx,val_x,val_y,val_idx)     
#transfer the array into available form

#network
def cnn_build():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=24,kernel_size=(5,12),padding='valid',activation='relu',input_shape=(5,1000,1)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2),padding='same'))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(24,activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    #model.add(tf.keras.layers.Dense(64,activation='relu'))
    #model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    model.summary()
    #adam=tf.keras.optimizers.Adam(lr=0.005)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    return model
#decode y_one_hotshot
def de_one_hot_y(array):
    result=np.zeros((array.shape[0],))
    for i in range(array.shape[0]):
        if array[i,0]==1:
            continue
        else:
            result[i]=1
    return result
#get 
def get_array_tp_tn(test_y,pred_x,list_idx):
    test_y=de_one_hot_y(test_y)
    true_list=[]
    for i in range(test_y.shape[0]):
        if test_y[i]==pred_x[i]:
            true_list.append(list_idx[i])
        else:
            continue
    return true_list

logdir='./%s'%output_dir
if not os.path.exists(logdir):
    os.mkdir(logdir)
acc_list=[]
True_list=[]
for i in range(5):
    test_x,test_y,test_idx,train_x,train_y,train_idx,val_x,val_y,val_idx=split_array(list_result,y_result,list_result_idx,5,i)    
    output_model=os.path.join(logdir,'model%d.h5'%i)
    #callbacks=[tf.keras.callbacks.TensorBoard(logdir),tf.keras.callbacks.ModelCheckpoint(output_model,save_best_only=True),
    #tf.keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)]
    callbacks=[tf.keras.callbacks.TensorBoard(logdir),tf.keras.callbacks.ModelCheckpoint(output_model,save_best_only=True)]
    model=cnn_build()
    history=model.fit(train_x,train_y,validation_data=(val_x,val_y),epochs=6,callbacks=callbacks)
    test_loss,test_acc=model.evaluate(test_x,test_y)
    pred=model.predict(test_x)
    np.save(logdir+'/testy%i.npy'%i,test_y)
    np.save(logdir+'/predict%i.npy'%i,pred)
    pred_x=np.argmax(pred,axis=1)
    true_list=get_array_tp_tn(test_y,pred_x,test_idx)
    True_list.append(true_list)
    acc_list.append(test_acc)
    print(test_acc,acc_list)
    plt.figure(figsize=(10,8))
    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.savefig('%s%d.png'%(input_file,i))
    plt.clf()
print(acc_list)
with open('%s.txt'%input_file,'w') as file:
    for i in acc_list:
        file.write(str(i)+'\n')
for i in range(len(True_list)):
    with open('%strue%d.txt'%(input_file,i),'w') as file:
        for j in True_list[i]:
            file.write(list_name[int(j)]+'\n')
