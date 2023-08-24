#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:58:15 2023

@author: hkaveh
"""

import numpy as np
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
# Gaussian Smoothing
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
print ('Librairies imported')

#%%
# Define the repertories
data_folder = '../Reservoir_data/'
results_folder = '../Simulation_results/'
RES_folder = 'RESDictionary_data/'

GF_folder = results_folder+'Greens_functions/'
#%%

# Import the data
print ('Loading reservoir data...')
# Create the reservoir Geometry
X = np.load(data_folder + RES_folder + 'X.p',allow_pickle=True) / 1000
Y = np.load(data_folder + RES_folder + 'Y.p',allow_pickle=True) / 1000
RES = np.load(data_folder + RES_folder + 'RES.p',allow_pickle=True)
# Load the reservoir outline
Outline = pd.DataFrame(np.load(data_folder + '/ReservoirOutline.npy')) / 1000 #dim change
# Load the Stresses
Stresses_center = np.load(results_folder+'max_coulomb_stresses.npy',allow_pickle=True).item()
Stresses_edge = np.load(results_folder+'max_coulomb_stresses_EdgeCuboid.npy',allow_pickle=True).item()
depths = [key[:-1] for key in Stresses_center.keys()]
kernel = Gaussian2DKernel(6)
Dates = np.linspace(1956.0,1956.0+Stresses_center['-1m'].shape[-1]/12,Stresses_center['-1m'].shape[-1])
print ("Reservoir data loaded")
#%%
#Plot 1: plot the simulation results at different times 
iterations = [500,650,790]

fig,ax = plt.subplots(2*len(depths),len(iterations),figsize=(4.4*len(iterations),4*(2*len(depths))+1))

for ii,depth in enumerate(depths):

    ### Min and max values for coulomb_stress, kernel definition
    min_cs=0
    max_cs=0.5
#     min_cs = np.nanmin(max_coulomb_stress)
#     max_cs = np.nanmax(convolve(max_coulomb_stress[:,:,iterations[-1]],kernel,nan_treatment='fill'))
#     ### Adaptation to km 
    x_outline = Outline[2]
    y_outline = Outline[3]
    x_res = X
    y_res = Y
    
    # Plot the figure
    for jj in range(len(iterations)):
        smoothed_coulomb_stress_C = convolve(Stresses_center[f'{depth}m'][:,:,iterations[jj]],kernel,nan_treatment='fill')
        smoothed_coulomb_stress_E = convolve(Stresses_edge[f'{depth}m'][:,:,iterations[jj]],kernel,nan_treatment='fill')
        quad_C = ax[ii,jj].scatter(x_res.flatten(), y_res.flatten(), c=smoothed_coulomb_stress_C, cmap='magma',vmin=min_cs,vmax=max_cs)
        quad_E = ax[ii+len(depths),jj].scatter(x_res.flatten(), y_res.flatten(), c=smoothed_coulomb_stress_C, cmap='magma',vmin=min_cs,vmax=max_cs)
        for kk in [ii,ii+len(depths)]:
            ax[kk,jj].plot(x_outline,y_outline,'k--',zorder=3)
            ax[kk,jj].text(270, 611, '{}'.format(np.round(Dates[iterations[jj]]),2),horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='w', edgecolor='k'))
            ax[kk,jj].set_xlim([230.000,270.000])
            ax[kk,jj].set_ylim([565.000,615.000])
            ax[kk,jj].set_xlabel('x (km)')
            ax[kk,jj].set_ylabel('y (km)')
            ax[kk,jj].set_aspect('equal')
        plt.colorbar(quad_C)
        plt.colorbar(quad_E)
plt.tight_layout()

# fig.subplots_adjust(left=0.12)
# cbar_ax_press = fig.add_axes([0.05, 0.72, 0.02,0.3-0.04])
# cbar_ax_disp = fig.add_axes([0.05, 0.4, 0.02,0.3-0.04])
# cbar_ax_stress = fig.add_axes([0.05, 0.05,0.02,(0.3-0.04)])
# cb_press = fig.colorbar(quad1, cax=cbar_ax_press,orientation='vertical',label='Pressure (MPa)')
# cb_press.ax.yaxis.set_ticks_position('left')
# cb_press.ax.yaxis.set_label_position('left')
# cb_disp = fig.colorbar(quad2, cax=cbar_ax_disp,orientation='vertical',label='Z displacement (mm)')
# cb_disp.ax.yaxis.set_ticks_position('left')
# cb_disp.ax.yaxis.set_label_position('left')
# cb_stress = fig.colorbar(quad3, cax=cbar_ax_stress,orientation='vertical',label='Coulomb stress change (MPa)'.format(depth))
# cb_stress.ax.yaxis.set_ticks_position('left')
# cb_stress.ax.yaxis.set_label_position('left')
# plt.show()


