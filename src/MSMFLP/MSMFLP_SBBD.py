import numpy as np
import gurobipy as gb
from gurobipy import GRB
import time 
import numexpr as ne
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


# In[ ]:


def lazyBD(m2, where):
    ################# integer separation #######################
    if where == GRB.Callback.MIPSOL:
        xt = m2.cbGetSolution(m2._x)
        thetat = m2.cbGetSolution(m2._t)
        xp = np.zeros(J)
        for j in range(J):
            xp[j] = round(xt[j])
        
        ##### new int sep #################
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
            ############ if root node
            if node_cnt == 0:
                #print('add user cut at root node')
                x_rel = m2.cbGetNodeRel(m2._x)
                theta_rel = m2.cbGetNodeRel(m2._t)
                xp = np.zeros(J)
                for j in range(J):
                    xp[j] = x_rel[j]
                
                #xp = ms2._step_size * xp + (1 - ms2._step_size) * ms2._xstab
        
                #ms2._xstab = (xp + ms2._xstab) / 2
                
                lamb, nu, mu, dual_obj = frac_dual_nps(xp)
                nmu = np.sum(delta * mu, axis = 1)
                ################# cut for each i #######################
                for i in range(I):
                    if theta_rel[i] > dual_obj[i]:
                        m2._user += 1
                        rhs1 = gb.quicksum(nu[i,j] * m2._x[j] for j in range(J))
                        rhs2 = gb.quicksum(nmu[i,j] * m2._x[j] for j in range(J))
                        m2.cbCut(m2._t[i] <= lamb[i] + rhs1 - rhs2)
           
            if node_cnt % m2._E1 == 0 and node_cnt != 0:
                    #print('add user cut at every E node')
                    x_rel = m2.cbGetNodeRel(m2._x)
                    theta_rel = m2.cbGetNodeRel(m2._t)
                    xp = np.zeros(J)
                    for j in range(J):
                        xp[j] = x_rel[j]

                    lamb, nu, mu, dual_obj = frac_dual_nps(xp) ###### solve dual subproblem
                    nmu = np.sum(delta * mu, axis = 1)
                    ################# cut for each i #######################
                    for i in range(I):
                        if theta_rel[i] > dual_obj[i]:
                            m2._user += 1
                            rhs1 = gb.quicksum(nu[i,j] * m2._x[j] for j in range(J))
                            rhs2 = gb.quicksum(nmu[i,j] * m2._x[j] for j in range(J))
                            m2.cbCut(m2._t[i] <= lamb[i] + rhs1 - rhs2)
                            


def SBBD():

    s_time = time.time()
    cut_time = 0
    
    ############# m1 is stage 1 model ####################
    m1 = gb.Model('stage1')
    x = m1.addVars(J, ub = 1, name = 'x')
    theta = m1.addVars(I, ub = 1e5, name = 'theta')
    obj = gb.quicksum(theta) / I
    m1.setObjective(obj, GRB.MAXIMIZE)
    m1.addConstr(gb.quicksum(x) == p)
    
    frac_cut = 0
    
    m1.params.OutputFlag = 0
    
    ############## m2 is stage 2 model #####################
    m2 = gb.Model('stage2')
    x2 = m2.addVars(J, vtype = GRB.BINARY, name = 'x2')
    theta2 = m2.addVars(I, ub = 1e5, name = 'theta2')
    obj2 = gb.quicksum(theta2) / I
    m2.setObjective(obj2, GRB.MAXIMIZE)
    m2.addConstr(gb.quicksum(x2) == p)
    
    m2._x = x2
    m2._t = theta2
    m2._E1 = 300
    m2._E2 = 100 #heuristic
    m2._lazy = 0
    m2._user = 0
    m2.Params.LazyConstraints = 1
    m2.Params.Presolve = 0
    m2._frac_key = 0
    m2._heuristic = 1
    
    
    LB = 0
    UB = 1e6
    UB_list = []
    LB_list = []
    X_list = []
    count = 0
    
    x_list = x.select('*')
    x2_list = x2.select('*')
    
    ####################### stage 1, solve lp ###########################
    print('stage 1 starts')
    while (UB - LB) / UB > 1e-4:
        count += 1
        m1.optimize()
        UB = min(UB, m1.objVal)
        UB_list.append(UB)
        X = np.zeros(J)
        for j in range(J):
            X[j] = x[j].x
        X_list.append(X)
        
        sep_time = time.time()
        
        ######################## accelerated fractional cut ##########################
        lamb, nu, mu, dual_obj = frac_dual_nps(X)
        nmu = np.sum(ne.evaluate('delta * mu'), axis = 1)
        mu_sum = np.einsum('ijk->i', mu)
        nu_diff = nu - nmu
    
        for i in range(I):
            if theta[i].x > dual_obj[i]:
                frac_cut += 1
                ########### add to m1 #####################
                rhs = gb.LinExpr(nu_diff[i], x_list)
                m1.addConstr(theta[i] <= lamb[i] + rhs + mu_sum[i])
                ########### add to m2 #####################
                nrhs = gb.LinExpr(nu_diff[i], x2_list)
                m2.addConstr(theta2[i] <= lamb[i] + nrhs + mu_sum[i])
        print("fractional separation time is", round(time.time() - sep_time,1))
        cut_time += time.time() - sep_time
        
    
        
        
        
        LB = max(LB, dual_obj.sum() / I)
        LB_list.append(LB)
        
        if count > 10:
            break
        print(count, " gap:", (UB - LB) / UB)
    print('relaxtion obj is', LB)
    print("\ntotal fractional cut separation time is", time.time() -s_time)
    print("\nStage 1 is", time.time() - s_time)
    
    m2.Params.TimeLimit = 7200 - (time.time() - s_time)
    
    ############################ stage 2, integer separation only #################################
    print('stage 2 starts')
    m2.optimize(lazyBD)
    print('number of fractional cut is', frac_cut)
    print('number of integer cut is', m2._lazy)
    print('number of heuristic integer cut is', m2._user)
    print(m2.objVal)

if __name__ == '__main__':
    I = 100   # sample size 

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
            
    SBBD()




