# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:18:46 2020

@author: liushang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Times New Roman'
prefix='F:/deep_network/deepfiber/02deepliftresult/scw/'
file=['ex0.npy','ex1.npy','ex2.npy','ex3.npy','ex4.npy']

a,t,c,g,n=[],[],[],[],[]
for i in file:
    test=np.load(prefix+i)
    test=test.mean(axis=0)
    a.append(test[0,:,0])
    t.append(test[1,:,0])
    c.append(test[2,:,0])
    g.append(test[3,:,0])
    n.append(test[4,:,0])
a,t,c,g,n=np.array(a),np.array(t),np.array(c),np.array(g),np.array(n)
#get plot
a_mean,t_mean,c_mean,g_mean,n_mean=a.mean(axis=0),t.mean(axis=0),c.mean(axis=0),g.mean(axis=0),n.mean(axis=0)
a_std,t_std,c_std,g_std,n_std=a.std(axis=0),t.std(axis=0),c.std(axis=0),g.std(axis=0),n.std(axis=0)
#get scatter
asc,tsc,csc,gsc,nsc=a_mean[491:510],t_mean[491:510],c_mean[491:510],g_mean[491:510],n_mean[491:510]
asc_std,tsc_std,csc_std,gsc_std,nsc_std=a_std[491:510],t_std[491:510],c_std[491:510],g_std[491:510],n_std[491:510]
#mask
a_mean[491:510],t_mean[491:510],c_mean[491:510],g_mean[491:510],n_mean[491:510]=[0 for i in range(19)],[0 for i in range(19)],[0 for i in range(19)],[0 for i in range(19)],[0 for i in range(19)]
a_std[491:510],t_std[491:510],c_std[491:510],g_std[491:510],n_std[491:510]=[0 for i in range(19)],[0 for i in range(19)],[0 for i in range(19)],[0 for i in range(19)],[0 for i in range(19)]




#plot
plt.figure(figsize=(15,5),dpi=600)

x=np.arange(-500,500)
plt.plot(x,a_mean,color='#ff0000',linewidth=0.8)
plt.fill_between(x,a_mean-a_std,a_mean+a_std,color='#ff0000',alpha=0.2)
plt.plot(x,t_mean,color='#00ff00',linewidth=0.8)
plt.fill_between(x,t_mean-t_std,t_mean+t_std,color='#00ff00',alpha=0.2)
plt.plot(x,c_mean,color='#0000ff',linewidth=0.8)
plt.fill_between(x,c_mean-c_std,c_mean+c_std,color='#0000ff',alpha=0.2)
plt.plot(x,g_mean,color='#000000',linewidth=0.8)
plt.fill_between(x,g_mean-g_std,g_mean+g_std,color='#000000',alpha=0.2)
plt.plot(x,n_mean,color='grey',linewidth=0.8)
plt.fill_between(x,n_mean-n_std,n_mean+n_std,color='grey')
plt.xticks((-500,-400,-300,-200,-100,0,99,199,299,399,499),['-500','-400','-300','-200','-100','TSS','100','200','300','400','500'],fontsize=20)
plt.legend(['A','T','C','G','N'],loc='center',fontsize=15,bbox_to_anchor=(1.05,0.3),borderaxespad=0)
plt.xlabel('Loc',fontsize=20)
plt.ylabel('Score',fontsize=20)
plt.title('scw',fontsize=25,pad=20)
plt.savefig('F:/deep_network/deepfiber/02deepliftresult/scw_plot.png')
plt.clf()


#scatter
a_mean,t_mean,c_mean,g_mean,n_mean=a.mean(axis=0),t.mean(axis=0),c.mean(axis=0),g.mean(axis=0),n.mean(axis=0)
a_std,t_std,c_std,g_std,n_std=a.std(axis=0),t.std(axis=0),c.std(axis=0),g.std(axis=0),n.std(axis=0)
asc,tsc,csc,gsc,nsc=a_mean[491:510],t_mean[491:510],c_mean[491:510],g_mean[491:510],n_mean[491:510]
asc_std,tsc_std,csc_std,gsc_std,nsc_std=a_std[491:510],t_std[491:510],c_std[491:510],g_std[491:510],n_std[491:510]
plt.figure(figsize=(15,5),dpi=600)
x=np.arange(-9,10)
plt.errorbar(x,asc,yerr=asc_std,fmt='o',color='#ff0000',ecolor='#ff0000',elinewidth=1,capsize=2)
plt.errorbar(x,tsc,yerr=tsc_std,fmt='o',color='#00ff00',ecolor='#00ff00',elinewidth=1,capsize=2)
plt.errorbar(x,csc,yerr=csc_std,fmt='o',color='#0000ff',ecolor='#0000ff',elinewidth=1,capsize=2)
plt.errorbar(x,gsc,yerr=gsc_std,fmt='o',color='#000000',ecolor='#000000',elinewidth=1,capsize=2)
plt.errorbar(x,nsc,yerr=nsc_std,fmt='o',color='grey',ecolor='grey',elinewidth=1,capsize=2)
plt.xticks(np.arange(-9,10,9),['-10','TSS','10'],fontsize=20)
plt.legend(['A','T','C','G','N'],loc='lower right',fontsize=20)
plt.xlabel('loc',fontsize=20)
plt.ylabel('score',fontsize=20)
plt.text(0,-0.002,'A',fontsize=40,color='#ff0000')
plt.text(1,-0.002,'T',fontsize=40,color='#00ff00')
plt.text(2,-0.004,'G',fontsize=40,color='#000000')
plt.savefig('F:/deep_network/deepfiber/02deepliftresult/scw_scatter.png')
plt.clf()
