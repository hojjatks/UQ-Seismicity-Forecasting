#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:26:17 2023

@author: hkaveh
"""


## To show how the method works, we solve a simple example. In a very simple model we assume that seimicity increases linearly with time: 
## R(t)=R0+a*(t) wherly with time: 
## R(t)=R0+a*(t) where R0, a are constants but unknown. Becaue R0, a are constants but unknown. Because they are unknown, they would lead to epistemic uncertainity. Also, since the process is a Poisson process we have aleatoric uncertainity on top
## Now there are two parameters to invert for, R0 and a (assuming model family with the form R0+a*(t))

#%% Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import Functions
from scipy.stats import chi2
import matplotlib as mpl

matplotlib.rcParams['font.family'] = 'serif'
    #plt.style.use('default')
cmap = plt.cm.RdYlGn
FontSize=8
custom_font='serif'
np.random.seed(150)
#%% Generating the True background rate and one catalog

Tmax=2020
Tmin=1990
N_train=15*12 # Number of years to be considered in the training set

Timevector=np.linspace(Tmin,Tmax,(Tmax-Tmin)*12)
Time_year=np.linspace(Tmin,Tmax,(Tmax-Tmin))
TimeTrain=Timevector[0:N_train]

True_values=np.array([-1442.5/12,.75/12]) # True Values of R0 and a [R0,a]
True_rate=True_values[0]+True_values[1]*Timevector
Realization=np.random.poisson(lam=True_rate,size=np.size(True_rate))
# For monthly data, to be able to see better, we aggerate data each year
True_rate_to_plot=True_rate.reshape(-1, 12).sum(axis=1)

Realization_to_plot=Realization.reshape(-1, 12).sum(axis=1)

fig = plt.figure(figsize=(3.7, 2.8))# Plotting the True rate and the Catalog (one realization of the Poisson Process)
plt.rc('font',family='Serif')
plt.rcParams.update({'font.family':'Serif'})
ax = fig.add_subplot(1, 1, 1)
ax.step(Timevector,True_rate,label='True Rate')
ax.step(Timevector,Realization,label='Synthetic Catalog',alpha=.5)
ax.set_xlabel('Time')
ax.set_ylabel('Number/Rate of events')
ax.legend(frameon=False)

#%% In real-world problem one need to invert for the model parameters using some inversion algorithm
# Here, since we have a very simple model of seimicity (only for Pedagogical goals to understand Algorithm 1) we can simply grid some prior range and find the likelihood of each point inside the grid:
# Specifying the range of parameters
R0_min=(True_values[0]-20/12)
R0_max=(True_values[0]+20/12)
a_min=0/12
a_max=1/12
N=1000 # Number of initial condition for each parameter
# So the total number will be N^2

R0_init=np.linspace(R0_min,R0_max,N)
a_init=np.linspace(a_min,a_max,N)
LogLikelihood=np.empty((N,N))
# Now for each parameter we want to find the likelihood
for i in range(N):
    for j in range(N):
        u=np.array([R0_init[i],a_init[j]])
        LogLikelihood[i,j]=(Functions.FindLogLikelihood(u,Realization[0:N_train],Timevector[0:N_train]))
    
# Now we have Likelihood of a lot of points!
# In the next section I first show you to find uncertainty bound only for one confidence level, then in the section after that one, I will use same methods to plot intervals for multiple bounds.
# Note in this tuterial we select (1-\xi) and (1-gamma) equal, in practice to find the minimum bound for that interval, one needs to explore different combination of those two while keeping (1-\xi)* (1-gamma)=Conflevels
# Notation change: instead of var \xi that is used in the paper I use zeta

#%% Quantifying the Uncertainity for one Confidence level: 
fig = plt.figure(figsize=(8, 2.8))

plt.rc('font',family='Serif')
plt.rcParams.update({'font.family':'Serif'})
ax = fig.add_subplot(1, 1, 1)
# Step 1 (from Algorithm 1): Likelihoods have been calculated and stored in LogLikelihood
q=2 # Number of parameters of the model (this is only for the simple model(only for Pedagogical goals to understand Algorithm 1)) for the model of Eq 1 in the paper we have q=4
conf= 90 # Desired level of confidence
zeta=  1-0.99 

gamma=1-.9/(1-zeta) # One can play with gamma and zeta, for monthly study, since the epistemic uncertainty is small we increase the confidence level in it

# Defining the chi2 distribution using scipy
ch=chi2(df=q)
# Step 2 (finding the loglikelihood): 
MaxLog=np.nanmax(LogLikelihood)
MLE_Index=np.unravel_index(np.nanargmax(LogLikelihood), LogLikelihood.shape)
MLE_number=R0_init[MLE_Index[0]]+a_init[MLE_Index[1]]*Timevector

# Step 3:
alpha=np.exp(-1/2*ch.ppf(1-zeta)) # ( Eq 12)

# Step 4:
Theta=np.empty((0,q))
R_concat=np.empty((0,Timevector.size)) # This will be the rate of events for all the elements inside the set theta
# R_concat=[]
for i in range(N):
    for j in range(N):
        if LogLikelihood[i,j] >= np.log(alpha)+MaxLog:# Eq 9 in the paper (here we are working with log of likelihood instead of the likelihood)
           u=np.array([R0_init[i],a_init[j]])
           Theta=np.vstack((Theta, u))
           R_pred=u[0]+u[1]*Timevector
                      
           R_concat=np.vstack((R_concat,R_pred))
           # R_concat.append(R_pred)
           # R_concat=np.append(R_concat,R_pred.reshape(Timevector.size,1),axis=1)
           ax.step(Timevector,R_pred,alpha=.005,color='red')
           
           
ax.step(Timevector,R_pred,label=r'$\mathbf {h}^{99\%}$',alpha=.005,color='red')
R_concat=R_concat.T
# Step 5: 
MinR=np.min(R_concat[:,0:],axis=1)            
MaxR=np.max(R_concat[:,0:],axis=1)

Confmin=np.zeros(np.shape(MinR))
Confmax=np.zeros(np.shape(MaxR))
# Step 6:      
    
for i in range(MinR.size):
        
    Confmin[i] =   +chi2.ppf(gamma/2,2*MinR[i])/2           # Eq 14
    Confmax[i]=   +  chi2.ppf(1-gamma/2,2*(MaxR[i]+1))/2    # Eq 14



ax.step(Timevector,Confmin,color='black',label=r"$\geq$"+'{:.1f}% Confidence Bound'.format(conf))
ax.step(Timevector,Realization,label='Synthetic Catalog',color='blue')  

ax.step(Timevector,MLE_number,label=r'$\mathbf {h}^{MLE}$',color='cyan')
ax.step(Timevector,Confmax,color='black')
ax.set_xlabel(r"Time (year)",fontname=custom_font,size=FontSize)
ax.set_ylabel(r"Number/rate of events",fontname=custom_font,size=FontSize)
plt.axvspan(Timevector[N_train],Tmax, color='black', alpha=0.5, lw=0)
ax.set_xlim(left=Tmin,right=Tmax)
ax.set_ylim(top=20)
leg=ax.legend(frameon=False,fontsize=FontSize,loc='upper left')
for lh in leg.legendHandles: 
    lh.set_alpha(1)

plt.grid(False)

fig.savefig('Figs/Timeseriesmonthly.png',bbox_inches = 'tight',dpi=700)
#%% Quantifying the Uncertainity

# Lets find confidence interval for different confidence levels!
minConf=20 # No one cares about 20 percent confidence level, but let's find it anyways!
maxConf=90
Nconf=10 # Griding the range(minConf,maxConf) into Nconf levels
Conflevels=np.linspace(minConf,maxConf,Nconf)



fig = plt.figure(figsize=(7.4,2.5))

plt.rc('font',family='Serif')
plt.rcParams.update({'font.family':'Serif'})

ax = fig.add_subplot(1, 2, 1)
ax2=fig.add_subplot(1,2,2)
ax.grid(False)
ax2.grid(False)

ax.set_xlabel(r"Time (year)",fontname=custom_font,size=FontSize)
ax.set_ylabel(r"Number/rate of events",fontname=custom_font,size=FontSize)
HighBoundMatrix=np.zeros((Nconf,Timevector.size)) # We are going to record the upper confidence intervals here, for different levels
LowBoundMatrix=np.zeros((Nconf,Timevector.size))  # We are going to record the lower confidence intervals here, for different levels
IncorrectCount=np.zeros((Nconf,1))                # Counts of data that are outside certain confidence interval


for index, conf in enumerate(Conflevels):
    zeta=  1-0.99 
    gamma=1-conf/100/(1-zeta) 

    # Now we find alpha using Eq 12 in the paper:
    alpha=np.exp(-1/2*ch.ppf(1-zeta)) # ( Eq 12)
    # Now we find set theta
    Theta=np.empty((0,2))
    R_concat=np.empty((0,Timevector.size))
    for i in range(N):
        for j in range(N):
            if LogLikelihood[i,j] >= np.log(alpha)+MaxLog:# Eq 9 in the paper (here we are working with log of likelihood instead of the likelihood)
                u=np.array([R0_init[i],a_init[j]])
                Theta=np.vstack((Theta, u))
                R_pred=u[0]+u[1]*Timevector
                R_concat=np.vstack((R_concat,R_pred))


    # Finding the minimum and the maximum for each time bin (step 5 in Algorithm)
    R_concat=R_concat.T
    MinR=np.min(R_concat[:,0:],axis=1)            
    MaxR=np.max(R_concat[:,0:],axis=1)

    Confmin=np.zeros(np.shape(MinR))
    Confmax=np.zeros(np.shape(MaxR))
        
    for i in range(MinR.size):
        
        Confmin[i] =   +chi2.ppf(gamma/2,2*MinR[i])/2           # Eq 14
        Confmax[i]=   +  chi2.ppf(1-gamma/2,2*(MaxR[i]+1))/2    # Eq 14
  

    HighBoundMatrix[index,:]=Confmax
    LowBoundMatrix[index,:]=Confmin
    c=cmap(Conflevels[index]/100)
    ax.step(Timevector,Confmin,color=c)
    ax.step(Timevector,Confmax,color=c)
    counter=0 # Counting number of data points that are outside the range
    for i in range(Timevector.size):
            if Confmax[i]<Realization[i] or Confmin[i]>Realization[i] :
                counter+=1
    IncorrectCount[index]=counter
Incorrectpercent=IncorrectCount/Timevector.size
CorrectPercent=(1-Incorrectpercent)*100
x=np.linspace(30,100,10)
y=x
ax2.fill_between(x,y,0,color='black',alpha=.1)
ax2.plot(CorrectPercent,Conflevels,'g+')
ax2.set_xlabel(r"Correct Percentage of Events",fontname=custom_font,size=FontSize)
ax2.set_ylabel(r"Confidence interval",fontname=custom_font,size=FontSize)
ax2.set_xlim(left=30,right=100)
ax2.set_ylim(bottom=0,top=100)
ax2.text(32,102,'(b)',ha="center", va="center", fontsize=FontSize,fontname=custom_font)
plt.tight_layout()
    # fig.savefig('Qualitytogether.png', bbox_inches = 'tight',dpi=600)
  

ax.text(1991,15,'(a)',ha="center", va="center", fontsize=FontSize,fontname=custom_font)
ax.set_xlim(left=Tmin,right=Tmax)
ax.axvspan(Timevector[N_train],Tmax, color='black', alpha=0.5, lw=0)

ax.step(Timevector,MLE_number,label=r'$\mathbf {h}^{MLE}$',color='cyan')
ax.step(Timevector,Realization,label='Synthetic Catalog',color='blue')
norm = mpl.colors.Normalize(vmin=minConf, vmax=maxConf) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb=fig.colorbar(sm,ax=ax)
cb.set_label(label=r'Percent Confidence interval bound',size=FontSize,fontname="serif")
leg=ax.legend(frameon=False,fontsize=FontSize)
for lh in leg.legendHandles: 
    lh.set_alpha(1)

fig.savefig('Figs/QualityMonthly.png',bbox_inches = 'tight',dpi=700)

#%%

np.savetxt("Data/CorrectPercent_monthly.csv", CorrectPercent, delimiter=",")
np.savetxt("Data/Conflevels_monthly.csv", Conflevels, delimiter=",")
