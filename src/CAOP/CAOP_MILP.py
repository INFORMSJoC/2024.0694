import numpy as np
import gurobipy as gb
from gurobipy import GRB
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
#############################################################



#################################################################
# Generate_AO_data_LHS is the function to generate CAOP data under multinomial probit model
# I is the number of samples
# J is the number of options
# seed is the random seed for LHS
def Generate_AO_data_LHS(I,J,mean_of_normal, std_of_normal, seed):
    ### generate reward and mean utility 
    np.random.seed(13579)
    u_determistic = np.random.randint(0,100,J)                                #determistic utility
    u_determistic = np.insert(u_determistic,0,50)    
    np.random.seed(24680)
    R = np.random.randint(0,100,J)                                          #determistic reward
    R = np.insert(R,0,0)
    r = np.ones((I,1))@(R.reshape((1,J+1)))

    ### generate samples
    points = (qmc.LatinHypercube(d=J+1,seed=seed)).random(n=I)
    U_random = stats.norm.ppf(points,loc=mean_of_normal, scale=std_of_normal)
    U = np.ones((I,1))@(u_determistic.reshape((1,J+1))) + U_random
    return(U, r)

#  Generate_AO_data_no_sample defines the function to generate reward and mean utility only. Readers can define their own uncertainity distribution.
def Generate_AO_data_no_sample(J, seed):
    np.random.seed(13579)
    u_determistic = np.random.randint(0,100,J)                                #determistic utility
    u_determistic = np.insert(u_determistic,0,50)    
    np.random.seed(24680)
    R = np.random.randint(0,100,J)                                          #determistic reward
    R = np.insert(R,0,0)
    r = np.ones((I,1))@(R.reshape((1,J+1)))
    return u_determistic, r
###################################################################



def gen_set(i, j):
    ss = []
    for k in range(J):
        if U[i][k] < U[i][j]:
            ss.append(k)
    return ss

def solve_milp():
    mlp = gb.Model('mlp')

    x = mlp.addVars(J, vtype = GRB.BINARY, name = 'x')
    y = mlp.addVars(I, J, ub = 1, name = 'y')
    obj = gb.quicksum(r[i][j] * y[i,j] for i in range(I) for j in range(J)) / I 
    mlp.setObjective(obj, GRB.MAXIMIZE)

    mlp.addConstr(x[no_purchase] == 1)
    mlp.addConstr(x.sum() <= p)
    mlp.addConstr(x.sum() >= 2)
    mlp.addConstrs(gb.quicksum(y[i,j] for j in range(J)) == 1 for i in range(I))
    mlp.addConstrs(y[i,j] <= x[j] for i in range(I) for j in range(J))
    mlp.addConstrs(gb.quicksum(y[i,k] for k in gen_set(i, j)) + x[j] <= 1 for i in range(I) for j in range(J))
    
    mlp._x = x
    mlp._y = y
    mlp._c = 0
    mlp.params.TimeLimit = 3600
    mlp.optimize()
    X = np.zeros(J)
    Y = np.zeros((I, J))
    for j in range(J):
        X[j] = x[j].x
        for i in range(I):
            Y[i][j] = y[i,j].x

    return round(mlp.Runtime, 1), mlp.NodeCount, mlp._c, mlp.objVal, round(mlp.MIPGap * 100, 2)

solve_milp()


if __name__ == '__main__':
    I = 100       # number of samples
    J = 100           # number of assortment


    p = int(0.1 * J)
    sigma = 0.2 #price_var
    u_var = 0.5

    seed = seed0 = 1

    U, r, no_purchase = Ali_data(I, J, sigma, seed, seed0, u_var)
    

    delta = np.zeros((I,J,J))
    for i in range(I):
        for j in range(J):
            delta[i,j] = U[i,j] < U[i]
    
    time, node, cut, obj, gap = solve_milp()



