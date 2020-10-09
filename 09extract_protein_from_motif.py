#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:25:27 2020

@author: ls
"""

import pandas as pd
kernal_result={}
file_list=['init_align0','init_align1','init_align2','init_align3','init_align4']
for i in file_list:   
    dataframe=pd.read_csv('/home/ls/deepfiber_new/03activated_kernals/initiation/%s/tomtom.tsv'%i,sep='\t')
    sig=dataframe.loc[dataframe['q-value']<=0.05]
    kernal_result[i]=sig
dic_duplicate,list_total,temp={},[],[]
for i in kernal_result.keys():
    for j in list(kernal_result[i]['Target_ID']):
        temp.append(j)
    temp=list(set(temp))
    for motif in temp:
        if motif not in list_total:
            dic_duplicate[motif]=1
            list_total.append(motif)
        else:
            dic_duplicate[motif]+=1
    temp=[]
dic_more_than_three={}
for i,j in zip(dic_duplicate.keys(),dic_duplicate.values()):
    if j >=3:
        dic_more_than_three[i]=j
    else:
        continue
target_list=list(dic_more_than_three.keys())
with open('/home/ls/deepfiber_new/motif_databases/JASPAR/JASPAR2018_CORE_plants_non_redundant.meme','r') as file:
    motif_list=[]
    for line in file:
        line=line.strip()
        if 'MOTIF' in line:
            motif_list.append(line.split(' '))
with open('/home/ls/deepfiber_new/03activated_kernals/extract_motif_init.txt','w') as file:
    for i in motif_list:
        if i[1] in target_list:
            file.write(i[2]+'\n')
            
