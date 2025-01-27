import numpy as np
import gurobipy as gb
from gurobipy import GRB
import time 
import numexpr as ne
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
                
                xpp3 = xp[1:].reshape(JJ, L)

                max_xpp3_index = np.argmax(xpp3, axis=1)
                sort_xpp3_index = np.argsort(-xpp3.max(axis=1))
                xpp4 = np.zeros((JJ, L))

                for k in range(p):
                    xpp4[sort_xpp3_index[k], max_xpp3_index[k]] = 1

                xpp5 = np.zeros(JJ * L + 1)
                xpp5[0] = 1
                xpp5[1:] = xpp4.ravel()

                
                lamb, nu, mu, dual_obj = int_dual_np(xpp5) ###### solve dual subproblem
                mu_sum = mu.sum(axis = 1)
                dual_obj_check = lamb + np.sum((nu - mu) * xpp5, axis = 1) + mu_sum
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
    m1.addConstr(x[0] == 1)
    m1.addConstr(x.sum() == p + 1)
    m1.addConstrs(gb.quicksum(x[j] for j in range(k * L + 1, k * L + L + 1)) <= 1 for k in range(JJ))
    
    frac_cut = 0
    m1.params.OutputFlag = 0
    
    
    ############## m2 is stage 2 model #####################
    m2 = gb.Model('stage2')
    x2 = m2.addVars(J, vtype = GRB.BINARY, name = 'x2')
    theta2 = m2.addVars(I, ub = 1e5, name = 'theta2')
    obj2 = gb.quicksum(theta2) / I 
    m2.setObjective(obj2, GRB.MAXIMIZE)
    m2.addConstr(x2[0] == 1)
    m2.addConstr(x2.sum() == p + 1)
    m2.addConstrs(gb.quicksum(x2[j] for j in range(k * L + 1, k * L + L + 1)) <= 1 for k in range(JJ))
    
    m2._x = x2
    m2._t = theta2
    m2._E1 = 300
    m2._E2 = 200 #heuristic
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
    
    ####################### stage 1, solve lp ###########################
    print('stage 1 starts')
    while (UB - LB) / (UB) > 1e-4:
        count += 1
        m1.optimize()
        UB = min(UB, m1.objVal)
        UB_list.append(UB)
        X = np.zeros(J)
        for j in range(J):
            X[j] = x[j].x
        
        #X = step_size * X + (1 - step_size) * x_stablizer
            
        #x_stablizer = (X + x_stablizer) / 2
            
        #step_size = min(step_size + 0.05, 1)
        
        
        X_list.append(X)
        
        sep_time = time.time()
        cut_time += time.time() - sep_time
        
        
        ######################## accelerated fractional cut ##########################
        lamb, nu, mu, dual_obj = frac_dual_nps(X)
        nmu = np.sum(ne.evaluate('delta * mu', optimization = 'aggressive'), axis = 1)
        mu_sum = np.sum(mu, axis = (1,2))
        for i in range(I):
            if theta[i].x > dual_obj[i]:
                frac_cut += 1
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
        
        if count > 15:
            break
        print(count, " gap:", (UB - LB) / (UB) * 100)
    print('relaxtion obj is', LB)
    print("\ntotal fractional cut separation time is", time.time() -s_time)
    print("\nStage 1 is", time.time() - s_time)
    t1 = time.time() - s_time
    
    m2.Params.TimeLimit = 3 * 3600 - (time.time() - s_time)
    
    ############################ stage 2, integer separation only #################################
    print('stage 2 starts')
    m2.optimize(lazyBD)
    print('number of fractional cut is',  frac_cut)
    print('number of integer cut is', m2._lazy)
    print('number of fractional cut is', m2._user)
    print(m2.objVal)
    print('total time is', m2.Runtime + t1)
    print('totoal cut is', frac_cut + m2._lazy + m2._user)


if __name__ == '__main__':
    ##################################################
    # below is example usage 
    I = 100   # sample size

    p = 10          # numbers of facilites to openå
     
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
    
    SBBD()





