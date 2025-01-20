# In[]:
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import time 
import numexpr as ne
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



##################### integer separation ###############################
def int_dual_np(x):
    J_open = [j for j in range(J) if x[j] > 0.5]   ### j such that x[j] = 1
    lamb, mu, nu = np.zeros(I), np.zeros((I,J)), np.zeros((I,J))
    for i in range(I):
        ## index of customer's choice
        m_i = J_open[np.argmax(U[i][J_open])]
        ## get lamb
        lamb[i] = r[i,m_i]
        ## get mu
        J2 = [j for j in J_open if j != m_i]
        
        mu[i,m_i] = max(np.max(r[i][J2] - lamb[i]), 0)
        
        ## get nu
        nu[i] = (1-x) * np.maximum(r[i] - lamb[i] - delta[i,:,m_i]*mu[i,m_i],0)
    ## calculate dual_obj
    dual_obj_arr = lamb
    return lamb, nu, mu, dual_obj_arr
################### fractional separation with ne #########################
def frac_dual_nps(x):
    #xd = np.expand_dims(np.ones((J, 1)) @ x.reshape((1, J)), 0).repeat(I, 0)
    xd = np.broadcast_to(x, (I, J, J))
    ## get beta
    xdd = ne.evaluate('1 - delta * xd', optimization = 'aggressive')
    xk = np.min(xdd, axis = 2)
    xm = np.ones((I, 1)) @ x.reshape((1, J))
    beta = np.minimum(xm, xk)
    ## solve knapsack problem to get lamb
    index = np.argsort(-r, axis = 1)
    lamb = np.zeros(I)
    for i in range(I):
        beta_sort_i = beta[i][index[i]]
        count = 1
        while np.sum(beta_sort_i[:count]) < 1:
            count += 1
        lamb[i] = r[i,index[i,count - 1]]
    ## get eta
    lambm = lamb.reshape((I,1)) @ np.ones((1,J))
    eta = np.maximum(ne.evaluate('r - lambm', optimization = 'aggressive'),0)
    ## get nu
    nu = ne.evaluate('eta * (beta == xm)', optimization = 'aggressive')
    ## get mu
    mu  = np.zeros((I,J,J))
    k_ind = np.argmin(xdd, axis = 2)   ## time-consumingss
    eta_ind = ne.evaluate('eta  - nu', optimization = 'aggressive')
    for i in range(I):
        for j in range(J):
            mu[i][j][k_ind[i,j]] = eta_ind[i,j]
    ## compute dual obj
    dual_obj_arr = lamb + np.sum(ne.evaluate('eta * beta', optimization = 'aggressive'), axis = 1)
    return lamb, nu, mu, dual_obj_arr



def lazyBD(m2, where):
    ################# integer separation #######################
    if where == GRB.Callback.MIPSOL:
        xt = m2.cbGetSolution(m2._x)
        thetat = m2.cbGetSolution(m2._t)
        xp = np.zeros(J)
        for j in range(J):
            xp[j] = round(xt[j])
        
        lamb, nu, mu, dual_obj = int_dual_np(xp) ###### solve dual subproblem
        mu_sum = mu.sum(axis = 1)
        ############# cut for each i ####################
        for i in range(I):
            if thetat[i] > dual_obj[i]:
                m2._lazy += 1
                rhs = gb.LinExpr(nu[i] - mu[i], [m2._x[j] for j in range(J)])
                m2.cbLazy(m2._t[i] <= lamb[i] + rhs + mu_sum[i])
    
    ############### heuristic integer separation #######################
    elif where == GRB.Callback.MIPNODE and m2._heuristic == 1:
        status = m2.cbGet(GRB.Callback.MIPNODE_STATUS)
        node_cnt = m2.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if status == GRB.OPTIMAL:
            if node_cnt % m2._E2 == 0:
                x_rel = m2.cbGetNodeRel(m2._x)
                theta_rel = m2.cbGetNodeRel(m2._t)
                xp = np.zeros(J)
                for j in range(J):
                    xp[j] = x_rel[j]
                
                index = np.argsort(-xp)[:p]
                
                xpp = np.zeros(J)
                xpp[index] = 1
                    
                
                lamb, nu, mu, dual_obj = int_dual_np(xpp) ###### solve dual subproblem
                mu_sum = mu.sum(axis = 1)
                dual_obj_check = lamb + np.sum((nu - mu) * xpp, axis = 1) + mu_sum
        ############# cut for each i ####################
                for i in range(I):
                    if theta_rel[i] > dual_obj_check[i]:
                        m2._user += 1
                        rhs = gb.LinExpr(nu[i] - mu[i], [m2._x[j] for j in range(J)])
                        m2.cbCut(m2._t[i] <= lamb[i] + rhs + mu_sum[i])
    
    
    ################ fractional separation ############################
    elif where == GRB.Callback.MIPNODE and m2._frac_key == 1:
        status = m2.cbGet(GRB.Callback.MIPNODE_STATUS)
        node_cnt = m2.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if status == GRB.OPTIMAL:                
            if node_cnt % m2._E1 == 0 and node_cnt != 0:
                    print('add user cut at every E node')
                    x_rel = m2.cbGetNodeRel(m2._x)
                    theta_rel = m2.cbGetNodeRel(m2._t)
                    xp = np.zeros(J)
                    for j in range(J):
                        xp[j] = x_rel[j]
                   
                    lamb, nu, mu, dual_obj = frac_dual_nps(xp)
                    nmu = np.sum(ne.evaluate('delta * mu', optimization = 'aggressive'), axis = 1)
                    mu_sum = mu.sum(axis = (1, 2))
                    ################# cut for each i #######################
                    for i in range(I):
                        if theta_rel[i] > dual_obj[i]:
                            m2._user += 1
                            rhs = gb.LinExpr(nu[i] - nmu[i], [m2._x[j] for j in range(J)])
                            m2.cbCut(m2._t[i] <= lamb[i] + rhs + mu_sum[i])
                            

def SBBD(stage_1):

    s_time = time.time()
    cut_time = 0

    ############# m1 is stage 1 model ####################
    m1 = gb.Model('stage1')
    x = m1.addVars(J, ub = 1, name = 'x')
    theta = m1.addVars(I, ub = 1e5, name = 'theta')
    obj = theta.sum() / I
    m1.setObjective(obj, GRB.MAXIMIZE)
    m1.addConstr(x[no_purchase] == 1)
    m1.addConstr(x.sum() <= p)
    m1.addConstr(x.sum() >= 2)

    x_stablizer = np.zeros(J)
    x_stablizer[no_purchase] = 1
    x_stablizer[1:] = (p - 1) / (J - 1)

    step_size = 0.5
    m1.params.OutputFlag = 0
    user_cut = 0

    ############## m2 is stage 2 model #####################
    m2 = gb.Model('stage2')
    x2 = m2.addVars(J, vtype = GRB.BINARY, name = 'x2')
    theta2 = m2.addVars(I, ub = 1e5, name = 'theta2')
    obj2 = theta2.sum() / I
    m2.setObjective(obj2, GRB.MAXIMIZE)
    m2.addConstr(x2[no_purchase] == 1)
    m2.addConstr(x2.sum() <= p)
    m2.addConstr(x2.sum() >= 2)

    m2._x = x2
    m2._t = theta2
    m2._E1 = 500 #frac
    m2._E2 = 200 #heuristic
    m2._lazy = 0
    m2._user = 0
    m2.Params.LazyConstraints = 1
    m2.Params.Presolve = 0
    #m2.Params.Cuts = 3
    m2._frac_key = 0
    m2._heuristic = 1
    m2.Params.OutputFlag = 1

    LB = 0
    UB = 1e6
    UB_list = []
    LB_list = []
    X_list = []
    count = 1

    ####################### stage 1, solve lp ###########################
    #print('stage 1 starts')
    while (UB - LB) / UB > 1e-2:
        count += 1
        m1.optimize()
        UB = min(UB, m1.objVal)
        UB_list.append(UB)
        X = np.zeros(J)
        for j in range(J):
            X[j] = x[j].x
        X_list.append(X)

        X = step_size * X + (1 - step_size) * x_stablizer

        x_stablizer = (X + x_stablizer) / 2
        
        step_size = min(step_size + 0.05, 1)


        sep_time = time.time()

        if stage_1:
            ######################## accelerated fractional cut ##########################
            lamb, nu, mu, dual_obj = frac_dual_nps(X)
            nmu = np.sum(ne.evaluate('delta * mu', optimization = 'aggressive'), axis = 1)
            mu_sum = np.sum(mu, axis = (1,2))
            for i in range(I):
                if theta[i].x > dual_obj[i]:
                    user_cut += 1
                    ########### add to m1 #####################
                    rhs = gb.LinExpr(nu[i] - nmu[i], [x[j] for j in range(J)])
                    m1.addConstr(theta[i] <= lamb[i] + rhs + mu_sum[i])
                    ########### add to m2 #####################
                    nrhs = gb.LinExpr(nu[i] - nmu[i], [x2[j] for j in range(J)])
                    m2.addConstr(theta2[i] <= lamb[i] + nrhs + mu_sum[i])
            print("fractional separation time is", round(time.time() - sep_time,1))
            cut_time += time.time() - sep_time

            LB = max(LB, dual_obj.sum() / I)
            LB_list.append(LB)

        if count > 50:
            break
        print(count, " gap:", (UB - LB) / UB)
    print('relaxtion obj is', LB)
    print("\ntotal fractional cut separation time is", time.time() -s_time)
    print("\nStage 1 is", time.time() - s_time)

    m2.Params.TimeLimit = 3600 - (time.time() - s_time)

    ############################ stage 2, integer separation only #################################
    print('stage 2 starts')
    m2.optimize(lazyBD)
    print('#################################################')
    print('total time is', time.time() - s_time + m2.Runtime)
    print('number of fractional cut is', user_cut)
    print('number of integer cut is', m2._lazy)
    print('number of heuristic integer cut is', m2._user)
    print('total lazy cut is', m2._lazy + m2._user)
    print('Objective',m2.objVal)

if __name__ == '__main__':

    I = 100            # number of samples
    J = 200            # number of assortment

    p = int(0.1 * J)
    sigma = 0.2 #price_var
    u_var = 0.5

    seed = seed0 = 1

    U, r, no_purchase = Ali_data(I, J, sigma, seed, seed0, u_var)
    

    delta = np.zeros((I,J,J))
    for i in range(I):
        for j in range(J):
            delta[i,j] = U[i,j] < U[i]

    stage_1 = 1
    SBBD(stage_1)
