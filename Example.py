#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:49:47 2023

@author: hkaveh
"""

#%% Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

#%%


# Generate data for the Gaussian distribution
mu = 0  # mean
sigma = 1  # standard deviation
mu2=1/2
mu3=-1/2
custom_font='serif'
FontSize=8

fig=plt.figure(figsize=(3.7, 4),frameon=False)

plt.rc('font',family='Serif')
plt.rcParams.update({'font.family':'Serif'})

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
y = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
y2 = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma)**2)
y3=(1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu3)/sigma)**2)
# Create the plot
plt.figure(figsize=(8, 4))
sns.set(style="whitegrid")
plt.plot(x, y, color='blue')
plt.plot(x, y2, color='red')
plt.plot(x, y3, color='red')




sns.set(style="whitegrid")


plt.axvline(mu , color='blue', linestyle='dotted', label='-2σ')
plt.axvline(1/2 , color='red', linestyle='dotted', label='-2σ')
plt.axvline(-1/2 , color='red', linestyle='dotted', label='-2σ')

plt.axvline(mu - 1*sigma, color='blue', linestyle='dashed', label='-2σ')

plt.axvline(mu +1/2+  1.5*sigma, color='red', linestyle='dashed', label='-2σ')
plt.axvline(mu -1/2-  1.5*sigma, color='red', linestyle='dashed', label='-2σ')

plt.axvline(mu + 1*sigma, color='blue', linestyle='dashed', label='+2σ')
plt.fill_between(x, y, where=(x >= mu - 1*sigma) & (x <= mu + 1*sigma), color='blue', alpha=0.2)
plt.fill_between(x, y2, where=(x >= 0) & (x <= mu +1/2++ 1.5*sigma), color='red', alpha=0.2)
plt.fill_between(x, y3, where=(x <= 0) & (x >= mu -1/2- 1.5*sigma), color='red', alpha=0.2)



plt.ylim(bottom=0,top=.75)
# Customize the plot
plt.xlim(mu - 2.5*sigma, mu + 2.5*sigma)
plt.xlabel('')
plt.ylabel('')  # Empty string to remove y-axis label
plt.yticks([])  # Remove y-axis tick marks and labels
plt.xticks([])  # Remove y-axis tick marks and labels
# plt.xticks([mu,], [ r'$m$'],fontname=custom_font,fontsize=FontSize+4)  # Set custom tick positions and labels
plt.text(mu-1/2-1/10, -1/17, r'$m^{-}$',fontname=custom_font,fontsize=FontSize+4)
plt.text(mu+1/2-1/10, -1/16, r'$m^+$',fontname=custom_font,fontsize=FontSize+4)
plt.text(mu-1/15, -1/17, r'$m$',fontname=custom_font,fontsize=FontSize+4)

plt.annotate('', xy=[-sigma,1/2],  xytext=(sigma,1/2),arrowprops=dict(arrowstyle='|-|', color='black', lw=2, shrinkA=0, shrinkB=0))
plt.annotate(r'$J$', xy=[0,1/2+1/30],  xytext=(-.2,1/2+1/30),fontname=custom_font,fontsize=FontSize+4)

plt.annotate('', xy=[mu -1/2-  1.5*sigma,1/2+1/8],  xytext=(mu +1/2+  1.5*sigma,1/2+1/8),arrowprops=dict(arrowstyle='|-|', color='black', lw=2, shrinkA=0, shrinkB=0))
plt.annotate(r'$I$', xy=[0,1/2++1/8+1/20],  xytext=(-.2,1/2+1/8+1/20),fontname=custom_font,fontsize=FontSize+4)
# Show the plot
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)
plt.savefig('Schematic.png',dpi=800)
plt.show()

#%%
