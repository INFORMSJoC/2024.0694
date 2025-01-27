# In[]:
import numpy as np
from scipy.stats import qmc
import scipy.stats as stats
import gurobipy as grb
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
def Generate_AO_data_no_sample(I, J, seed):
    np.random.seed(13579)
    u_determistic = np.random.randint(0,100,J)                                #determistic utility
    u_determistic = np.insert(u_determistic,0,50)    
    np.random.seed(24680)
    R = np.random.randint(0,100,J)                                          #determistic reward
    R = np.insert(R,0,0)
    r = np.ones((I,1))@(R.reshape((1,J+1)))
    return u_determistic, r
###################################################################


def BIBC():
    model = grb.Model()
    model.setParam('OutputFlag', 1)
    model.setParam('lazyConstraints',1)

    x = model.addVars(J,vtype=grb.GRB.BINARY)
    y = model.addVars(I,J)

    model.setObjective(grb.quicksum(r[i,j]*y[i,j] for i in range(I) for j in range(J))/I, grb.GRB.MAXIMIZE)

    model.addConstr(x[no_purchase] == 1)
    model.addConstr(x.sum() <= p)
    model.addConstr(x.sum() >= 2)


    for i in range(I):
        model.addConstr(grb.quicksum(y[i,j] for j in range(J)) == 1)    
        for j in range(J):
            model.addConstr(y[i,j] <= x[j])

    '''This is the preprocessing constraints'''
    ### y_{ij} can be 1 only if U[i,j] is better than the outside option
    for i in range(I):
        for j in range(J):
            if U[i,j] < U[i][no_purchase]: 
                model.addConstr(y[i,j] == 0)

    def lazy_cut(model, where):
        if where == grb.GRB.Callback.MIPSOL:
              x_vals = model.cbGetSolution(model._x) 
              y_vals = model.cbGetSolution(model._y)
              set_open = [j for j in range(J) if x_vals[j] > 0.5]
              x_sol = np.zeros(J)    ### x solution from the tree
              x_sol[set_open] = 1
              y_sol = np.zeros((I,J))  ### y solution from the tree
              for i in range(I):
                  for j in range(J):
                      y_sol[i,j] = y_vals[i,j]
              ### solve for optimal y given current x
              y_last = np.zeros((I,J))
              for i in range(I):
                  index = set_open[np.argmax(U[i][set_open])]
                  y_last[i,index] = 1
              ## add cuts                            
              Uy = U*y_last         
              for i in range(I): 
                   if np.sum(U[i]*y_sol[i]) * (1+1e-5) < np.sum(Uy[i]): ### check cuts violation
                      model.cbLazy(grb.quicksum(U[i,j]*y[i,j] for j in range(J))
                                >= grb.quicksum(Uy[i,j]*x[j] for j in range(J)))
                      model._number_of_cuts +=1

    model.Params.Cuts = 3
    model.Params.TimeLimit = 3600
    model._x = x
    model._y = y
    model._number_of_cuts = 0
    model.optimize(lazy_cut) 
    
    return round(model.Runtime, 1), model.NodeCount, model._number_of_cuts, model.objVal, round(model.MIPgap * 100, 2)

if __name__ == '__main__':
    ###############################################
    # below is example usage
    I = 100       # number of samples
    J = 100            # number of assortment


    p = int(0.1 * J)
    sigma = 0.2 #price_var
    u_var = 0.5

    seed = seed0 = 1

    U, r, no_purchase = Ali_data(I, J, sigma, seed, seed0, u_var)


    delta = np.zeros((I,J,J))
    for i in range(I):
        for j in range(J):
            delta[i,j] = U[i,j] < U[i]

    n = J
    ''' tranform uitlity values into rankings'''
    for i in range(I):
        index = np.argsort(U[i])
        U[i,index] = np.arange(J) + 1
        
    Time, node, cuts, Obj, gap = BIBC()




