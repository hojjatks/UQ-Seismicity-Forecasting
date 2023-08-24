


import numpy as np
def FindR(u,time):
    R=u[0]+u[1]*time  # (A simple Pedagogical  example) Linear model
    return R
def FindLogLikelihood (u,R0,time,LIKELIHOOD='Poisson',gamma=1):

    if LIKELIHOOD=='Poisson':
        R = FindR(u,time) # This is a vector predicted by the model, we need log of this vector
        log_R_model=np.log(R)
        
        dum1=np.dot(R0,log_R_model)
        dum2=np.sum(R)
        w=(dum1-dum2)

    if LIKELIHOOD=='Gaussian':
        R=FindR(u,time)
        No_zero_R0=R0[R0 != 0]
        Nonzeros=np.size(No_zero_R0)
        gamma=np.sqrt(np.mean(No_zero_R0))
#        w=-0.5*(np.linalg.norm(R-R0,ord=1))**2/(gamma**2)
        w=-0.5*(np.linalg.norm(R-R0))**2/(Nonzeros*gamma**2)
        
    return w
