# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:16:46 2020

@author: liushang
"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Times New Roman'

dir_='F:/deep_network/deepfiber/01cnnresult/scw/'#
testp=np.load(dir_+'predict0.npy')
testy=np.load(dir_+'testy0.npy')
for i in range(1,5):
    testp=np.r_[testp,np.load(dir_+'predict%d.npy'%i)]
    testy=np.r_[testy,np.load(dir_+'testy%d.npy'%i)]
def cal_accuracy(testp,testy):
    test_maxp=np.argmax(testp,axis=1)
    test_maxy=np.argmax(testy,axis=1)
    num=0
    for i,j in zip(test_maxp,test_maxy):
        if i==j:
            num+=1
        else:
            continue
    accuracy=round(num/testp.shape[0],2)*100 
    return accuracy
#modify corresponding dir
accuracy_initlist=np.array([cal_accuracy(np.load(dir_+'predict%d.npy'%i),np.load(dir_+'testy%d.npy'%i)) for i in range(5)])
accuracy_elonglist=np.array([cal_accuracy(np.load(dir_+'predict%d.npy'%i),np.load(dir_+'testy%d.npy'%i)) for i in range(5)])
accuracy_scwlist=np.array([cal_accuracy(np.load(dir_+'predict%d.npy'%i),np.load(dir_+'testy%d.npy'%i)) for i in range(5)])

#accuracy curve
def plot_sig(yend):    
        x=np.ones((2))
        y=np.arange(yend,yend+2,1)
        plt.plot(x,y,label='$y$',color='black',linewidth=1)
        
        x=np.arange(1,3,1)
        y=[yend+1,yend+1]
        plt.plot(x,y,label='$y$',color='black',linewidth=1)
        x0=1.5
        y0=yend+1.5
        plt.annotate(r'$***$',xy=(x0,y0),xytext=(-15,+1),xycoords='data',textcoords='offset points',fontsize=16,color='red')
        
        x=2*np.ones((2))
        y=np.arange(yend,yend+2,1)
        plt.plot(x,y,label='$y$',color='black',linewidth=1)
def plot_sig2(yend):    
        x=np.zeros((2))
        y=np.arange(yend+2,yend+4,1)
        plt.plot(x,y,label='$y$',color='black',linewidth=1)
        
        x=np.arange(0,3,2)
        y=[yend+3,yend+3]
        plt.plot(x,y,label='$y$',color='black',linewidth=1)
        x0=1
        y0=yend+3.5
        plt.annotate(r'$**$',xy=(x0,y0),xytext=(-15,+1),xycoords='data',textcoords='offset points',fontsize=16,color='red')
        
        x=2*np.ones((2))
        y=np.arange(yend+2,yend+4,1)
        plt.plot(x,y,label='$y$',color='black',linewidth=1)

x=np.arange(3)
stdinit=accuracy_initlist.std()
stdelong=accuracy_elonglist.std()
stdscw=accuracy_scwlist.std()
error_attri={'elinewidth':2,'ecolor':'black','capsize':6}
bar_width=0.4
tick_label=['initiation','elongation','scw']
mean_list=[accuracy_initlist.mean(),accuracy_elonglist.mean(),accuracy_scwlist.mean()]
std_list=[stdinit,stdelong,stdscw]
plt.figure(figsize=(15,4),dpi=600)
plt.bar(x,mean_list,bar_width,yerr=std_list,error_kw=error_attri,alpha=0.8,)
plt.xticks(x,tick_label,fontsize=25)
plt.ylabel('accuracy',fontsize=25)
plt.grid(axis='y')
plt.ylim(0,100)
plt.annotate(r'$77.8\%$',xy=(0,81.5),xytext=(-15,+1),xycoords='data',textcoords='offset points',fontsize=16,color='red')
plt.annotate(r'$77.4\%$',xy=(1,81.5),xytext=(-15,+1),xycoords='data',textcoords='offset points',fontsize=16,color='red')
plt.annotate(r'$75.6\%$',xy=(2,81.5),xytext=(-15,+1),xycoords='data',textcoords='offset points',fontsize=16,color='red')
plt.savefig('F:/deep_network/deepfiber/01cnnresult/accuracy.png')
plt.clf()

#roc curve
def cal_tpr_fpr(threshold,testp,testy):    
    list_maxp=[]
    for i in testp:
        if i[0]>=threshold:
            list_maxp.append(0)
        else:
            list_maxp.append(1)
    maxp=np.array(list_maxp)
    maxy=np.argmax(testy,axis=1)
    real_p=len([i for i in maxy if i==1])
    real_n=len([i for i in maxy if i==0])
    tp,fp=0,0
    for i,j in zip(maxp,maxy):
        if (i==1)&(i==j):
            tp+=1
        elif (i==1)&(i!=j):
            fp+=1
        else:
            continue
    tpr=tp/real_p
    fpr=fp/real_n  
    return(tpr,fpr)
tpr,fpr=[],[]
for i in np.arange(0,1.1,0.1):
    tpr.append(cal_tpr_fpr(i,testp,testy)[0])
    fpr.append(cal_tpr_fpr(i,testp,testy)[1])
def cal_auroc(tpr,fpr):
    auroc=0
    for i in range(len(tpr[:-1])):
        height=fpr[i+1]-fpr[i]
        sum_bottom=tpr[i]+tpr[i+1]
        auroc+=sum_bottom*height*0.5
    return auroc
auroc=round(cal_auroc(tpr,fpr),2)
plt.figure(figsize=(5,4),dpi=600)
plt.plot(fpr,tpr,c='r',marker='o')
plt.plot(np.arange(0,2),np.arange(0,2),linestyle='--')
plt.xlabel('FPR',fontsize=20)
plt.ylabel('TPR',fontsize=20)
plt.title('scw',fontsize=20,pad=20)#
plt.text(0.17,0.5,'AUROC: %.2f'%auroc,fontsize=15)
plt.savefig('F:/deep_network/deepfiber/01cnnresult/scw_roc.png')#
plt.clf()