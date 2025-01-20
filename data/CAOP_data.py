import numpy as np
import scipy.stats as stats
from scipy.stats import qmc

###########################################################
# Ali_data defines the function to generate CAOP data under exponomial choice model, used in Aouad et al. (2023)
# II is the number of samples
# JJ is the number of options 
# sigma is the varinace of lognormal distribution
# seed is the random seed for LHS
# seed0 is the random seed to generate reward and mean utility
# u_var is the variance of mean utility
# no_purchase is the index of outside option

def Ali_data(II, JJ, sigma, seed, seed0, u_var):
    ### generate reward and mean utility
    np.random.seed(seed0 + 1)
    R = np.random.lognormal(0, sigma, JJ)
    np.random.seed(seed0 + 2)
    u = -np.sort(-np.random.normal(1, u_var, JJ)) + 1  
    no_purchase = np.where(u >= 0)[0][-1]
    R[no_purchase] = 0
    u = u - u[no_purchase]
    r = np.ones((II, 1)) @ R.reshape((1, JJ))
    
    ### generate samples
    points = (qmc.LatinHypercube(d = JJ,seed = seed, optimization = 'random-cd')).random(n = II)
    U_random = stats.expon.ppf(points, scale = 1)
    U = np.ones((II, 1)) @ u.reshape((1, JJ)) - U_random
    return U, r, int(no_purchase)

# Ali_data_no_sample defines the function to generate reward and mean utility only. Readers can define their own uncertainity distribution.
def Ali_data_no_sample(II, JJ, sigma, seed0, u_var):
    np.random.seed(seed0 + 1)
    R = np.random.lognormal(0, sigma, JJ)
    np.random.seed(seed0 + 2)
    u = -np.sort(-np.random.normal(1, u_var, JJ)) + 1  
    no_purchase = np.where(u >= 0)[0][-1]
    R[no_purchase] = 0
    u = u - u[no_purchase]
    r = np.ones((II, 1)) @ R.reshape((1, JJ))
    return u, r

#################################################################
# Generate_AO_MNP_data_LHS is the function to generate CAOP data under multinomial probit model
# I is the number of samples
# J is the number of options
# seed is the random seed for LHS
def Generate_AO_MNP_data_LHS(I,J,mean_of_normal,std_of_normal,seed):
    ### generate reward and mean utility 
    np.random.seed(13579)
    u_determistic = np.random.randint(0,100,J)                                #determistic utility
    u_determistic = np.insert(u_determistic,0,50)    
    np.random.seed(24680)
    R = np.random.randint(0,100,J)                                            #determistic reward
    R = np.insert(R,0,0)
    r = np.ones((I,1))@(R.reshape((1,J+1)))

    ### generate samples
    points = (qmc.LatinHypercube(d=J+1,seed=seed)).random(n=I)
    U_random = stats.norm.ppf(points,loc=mean_of_normal, scale=std_of_normal)
    U = np.ones((I,1))@(u_determistic.reshape((1,J+1))) + U_random
    return(U, r)

#  Generate_AO_MNP_data_no_sample defines the function to generate reward and mean utility only. Readers can define their own uncertainity distribution.
def Generate_AO_MNP_data_no_sample(I, J, seed):
    np.random.seed(13579)
    u_determistic = np.random.randint(0,100,J)                                #determistic utility
    u_determistic = np.insert(u_determistic,0,50)    
    np.random.seed(24680)
    R = np.random.randint(0,100,J)                                            #determistic reward
    R = np.insert(R,0,0)
    r = np.ones((I,1))@(R.reshape((1,J+1)))
    return u_determistic, r
###################################################################

# In[ ]:
# below is example usage
if __name__ == '__main__':
    I = 100            # number of samples
    J = 200            # number of assortment
     
    # Example for Ali_data
    p = int(0.1 * J)
    sigma = 0.2        # price_var
    u_var = 0.5
    
    seed = seed0 = 1
    
    U, r, no_purchase = Ali_data(I, J, sigma, seed, seed0, u_var)
    
    delta = np.zeros((I,J,J))
    for i in range(I):
        for j in range(J):
            delta[i,j] = U[i,j] < U[i]
    ###################################################
    
    
    # Example for multinomial probit model
    seed = 1
    mean_of_normal = 0
    std_of_normal = 100
    U_MNP,r_MNP = Generate_AO_MNP_data_LHS(I,J,mean_of_normal,std_of_normal,seed)
