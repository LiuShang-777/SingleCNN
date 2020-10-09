# -*- coding: utf-8 -*-
# receive model file, input file and shuffle file, then output expressed and unexpressed files
import sys
import deeplift
import numpy as np
from deeplift.conversion import kerasapi_conversion as kc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
model_file=sys.argv[1]
input_file=sys.argv[2]
shuffle_file=sys.argv[3]
output_ex_file=sys.argv[4]
output_no_file=sys.argv[5]
'''
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
deeplift_model=kc.convert_model_from_saved_files(model_file,nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
deeplift_contribs_func=deeplift_model.get_target_contribs_func(find_scores_layer_idx=0,target_layer_idx=-2)
#get sequence data
def class_expressioin(input_file):
    stat,list_ex,list_no=0,[],[]
    with open(input_file,'r') as file:    
        for line in file:
            line=line.strip()
            if (line[0]=='>')&('aw' in line):
                stat=0            
            elif (line[0]=='>')&('ae' in line):
                stat=1            
            else:
                if stat==0:
                    list_no.append(line)
                else:
                    list_ex.append(line)
    return (list_ex,list_no)

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

list_ex,list_no=class_expressioin(input_file)
list_ex_sh,list_no_sh=class_expressioin(shuffle_file)
list_ex_array,list_no_array=tansfer_str_to_array(list_ex),tansfer_str_to_array(list_no)
list_ex_sh_array,list_no_sh_array=tansfer_str_to_array(list_ex_sh),tansfer_str_to_array(list_no_sh)                
list_ex_array_result,list_no_array_result,list_ex_sh_array_result,list_no_sh_array_result=[],[],[],[]
for i in range(list_ex_sh_array.shape[0]):
    list_ex_array_result.append(one_hotshot(list_ex_array[i,:],5))
    list_ex_sh_array_result.append(one_hotshot(list_ex_sh_array[i,:],5))
for i in range(list_no_sh_array.shape[0]):
    list_no_array_result.append(one_hotshot(list_no_array[i,:],5))
    list_no_sh_array_result.append(one_hotshot(list_no_sh_array[i,:],5))

list_ex_array_result,list_ex_sh_array_result,list_no_array_result,list_no_sh_array_result=np.array(list_ex_array_result),np.array(list_ex_sh_array_result),np.array(list_no_array_result),np.array(list_no_sh_array_result)
list_ex_array_result,list_ex_sh_array_result,list_no_array_result,list_no_sh_array_result=list_ex_array_result[:,:,:,np.newaxis],list_ex_sh_array_result[:,:,:,np.newaxis],list_no_array_result[:,:,:,np.newaxis],list_no_sh_array_result[:,:,:,np.newaxis]

#list_ex_array_result[:,:,500:504,:]=0
#list_no_array_result[:,:,500:504,:]=0
list_ex_scores=np.array(deeplift_contribs_func(task_idx=1,input_data_list=[list_ex_array_result],input_references_list=[list_ex_sh_array_result],batch_size=10,progress_update=1000))
list_no_scores=np.array(deeplift_contribs_func(task_idx=0,input_data_list=[list_no_array_result],input_references_list=[list_no_sh_array_result],batch_size=10,progress_update=1000))
np.save(output_ex_file,list_ex_scores)
np.save(output_no_file,list_no_scores)


