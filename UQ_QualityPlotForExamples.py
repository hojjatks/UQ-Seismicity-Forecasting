#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:43:07 2023

@author: hkaveh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import Functions
from scipy.stats import chi2
import matplotlib as mpl


#%%

custom_font='serif'
FontSize=8
CorrectPercent_yearly=np.loadtxt("Data/CorrectPercent_yearly.csv", delimiter=",")
Conflevels_yearly=np.loadtxt("Data/Conflevels_yearly.csv", delimiter=",")
CorrectPercent_monthly=np.loadtxt("Data/CorrectPercent_monthly.csv", delimiter=",")
Conflevels_monthly=np.loadtxt("Data/Conflevels_monthly.csv", delimiter=",")
fig = plt.figure(figsize=(3.7,2.8))

plt.rc('font',family='Serif')
plt.rcParams.update({'font.family':'Serif'})

ax = fig.add_subplot(1, 1, 1)

x=np.linspace(40,100,10)
y=x
ax.fill_between(x,y,0,color='black',alpha=.1)
ax.plot(CorrectPercent_monthly,Conflevels_monthly,'kv',label='Monthly analysis')
ax.plot(CorrectPercent_yearly,Conflevels_yearly,'g+',label='Yearly analysis')

ax.set_xlabel(r"Correct Percentage of Events",fontname=custom_font,size=FontSize)
ax.set_ylabel(r"Confidence interval",fontname=custom_font,size=FontSize)
ax.set_xlim(left=45,right=101)
ax.set_ylim(bottom=0,top=100)
ax.legend(frameon=False,fontsize=FontSize)
plt.tight_layout()
# ax.text(32,102,'(b)',ha="center", va="center", fontsize=FontSize,fontname=custom_font)
# plt.tight_layout()
fig.savefig("Figs/DataEffectUQ_quality.png",dpi=700)
