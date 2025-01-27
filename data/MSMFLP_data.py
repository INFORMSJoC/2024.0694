# In[ ]:
import numpy as np
import scipy.stats as stats
from scipy.stats import qmc

####################################################
# Generate_MCP_data_LHS_Normal generates 
# I is the number of samples
# J is the number of options
# O is the utility of outside option
# seed is the random seed for LHS
# mean_of_normal is the mean of normal distribution
# std_of_nornal is the standard deviation of normal distribution

def Generate_MCP_data_LHS_Normal(I,J,O,seed):
    points = (qmc.LatinHypercube(d=2,seed=seed,optimization = 'random-cd')).random(n=I)
    customer_x_axis = stats.norm.ppf(points[:,0],loc=mean_of_normal, scale=std_of_normal)
    customer_y_axis = stats.norm.ppf(points[:,1],loc=mean_of_normal, scale=std_of_normal)

    np.random.seed(123456789)
    facility_x_axis = np.random.uniform(0, 20, J)
    np.random.seed(987654321)
    facility_y_axis = np.random.uniform(0, 20, J)
 
    Distance = np.zeros((I,J))
    for j in range(J):
        Distance[:,j] = np.sqrt((customer_x_axis - facility_x_axis[j])**2 
                              + (customer_y_axis - facility_y_axis[j])**2)

    np.random.seed(13579)
    attraction = np.random.uniform(1,20,J)
    ### compute U
    U = np.ones((I,1))@attraction.reshape((1,J)) / Distance ** 2
    ### compute R
    R = U/(U+O)
    return U, R
####################################################


if __name__ == '__main__':
    ###############################################
    # below is example usage
    I = 1000   # sample size 
    
    p = 20      # numbers of facilites to open
    
    JJ = 400    # candicate facilities
    
    O = 3   # uitlity of outside option
    
    S = 1
    
    mean_of_normal = 10
    std_of_normal = np.sqrt(100 / 3)
    seed = 1
    
    U, r =  Generate_MCP_data_LHS_Normal(I,JJ,O,0)
    
    
    I, J = U.shape
    
    
    delta = np.zeros((I,J,J))
    for i in range(I):
        for j in range(J):
            delta[i,j] = U[i,j] < U[i]
    ###############################################
