# In[ ]:
import numpy as np
from scipy.stats import qmc


##################################################
# Generate_LOP_data_LHS_Uniform generates FLoP data of uniformly distributed demand zones
# I is the number of samples
# J is the number of options
# kk is the range for uniform distribution, i.e., U[0, kk]. Set as 20 in our paper
# seed is the random seed for LHS

def Generate_LOP_data_LHS_Uniform(I,J,seed, kk):       
    points = (qmc.LatinHypercube(d=2,seed=seed,optimization = 'random-cd')).random(n=I)
    customer_x_axis = points[:,0] * kk
    customer_y_axis = points[:,1] * kk

    np.random.seed(123456789)
    facility_x_axis = np.random.uniform(0, kk, J)
    np.random.seed(987654321)
    facility_y_axis = np.random.uniform(0, kk, J)
 
    Distance = np.zeros((I,J))
    for j in range(J):
        Distance[:,j] = np.sqrt((customer_x_axis - facility_x_axis[j])**2 
                              + (customer_y_axis - facility_y_axis[j])**2)

    s = np.arange(L) + 1              ### Service charge
    R = np.zeros(L*J+1)               ### R values
    for k in range(J):
        R[k*L+1:k*L+L+1] = s

    G = np.zeros((I,J,L))             ### Customer accessing cost + charge
    for l in range(L):
        G[:,:,l] = Distance  + s[l]

    U = np.zeros((I,L*J+1))           ### Utility
    for i in range(I):
        for k in range(J):
            U[i,k*L+1:k*L+L+1] = -G[i,k]

    U[:,0] = - 10   ### Utility of the outside option / customer's budget

    return U, R


if __name__ == '__main__':
    ##################################################
    # below is example usage 
    I = 1000         # sample size
    
    p = 10           # numbers of facilites to open
     
    JJ = 50          # candicate facilities
    L  = 10          # candicate pricing levelså
    seed = 5
    S = 1
    mean_of_normal = 10
    std_of_normal = np.sqrt(100 / 3)
    
    kk = 20
    U, r = Generate_LOP_data_LHS_Uniform(I,JJ,seed, kk)
    
    I,J = U.shape
    
    r = np.ones((I, 1)) @ r.reshape((1, J))
    
    
    US = np.zeros((I, J, S))
    US[:,:,0] = U
    
    for i in range(I):
        for j in range(J):
            if U[i][j] < U[i][0]:
                r[i][j] = 0
                
    delta = np.zeros((I,J,J))
    for i in range(I):
        for j in range(J):
            delta[i,j] = U[i,j] < U[i]
    ##################################################
