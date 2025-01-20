import numpy as np
import scipy.stats as stats
from scipy.stats import qmc
import gurobipy as grb


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
    ###############################################
    
    
    d = 1/I * np.ones(I)
    u = U
    u0 = O
    ### compute pi
    pi = u/u0        ### u of facilities,   u0 outside option
    ##################################################
    # define master problem    
    m = grb.Model()
    m.setParam('TimeLimit', 3600)
    m.setParam('IntFeasTol', 1e-9) # default 1e-5
    m.setParam('FeasibilityTol', 1e-9) # default 1e-5
    m.setParam('OptimalityTol', 1e-9) # default 1e-5
    m.setParam('Cuts', 3)
    
    x = m.addVars(J,vtype=grb.GRB.BINARY,name='X')
    w = m.addVars(I)
    m.update()
    m.setObjective(grb.quicksum(w[i] for i in range(I)),grb.GRB.MINIMIZE)
    m.addConstr(x.sum() == p)
    
    def lazy_cut(model, where):
        if where == grb.GRB.Callback.MIPSOL:     
             x_vals = np.array([model.cbGetSolution(m._x[i]) for i in range(J)])
             x_last = np.round(x_vals)  # Round to nearest integer solutions
             set_of_close = [j for j, x in enumerate(x_last) if x == 0]  # The set of closed facilities
             ######################################################################      
             
             # Sorting algorithm for the best y
             y_last = np.zeros((I, J))
             pi_x = pi * (np.ones((I, 1)) @ x_last.reshape((1, J)))
             max_indices = pi_x.argmax(axis = 1)
             y_last[np.arange(I), max_indices] = 1  # This is the best solution
    
             piy = np.sum(pi * y_last, axis = 1) + 1
             Phi = d * (1 / piy)
            
            
            
            
             ######################################################################
             # compute lamda
             lamda = np.zeros((I,J))       # intinally, set lamda = 0
                
             x_last_m = np.ones((I, 1)) @ x_last.reshape((1, J))
             xy1 = (y_last < x_last_m).astype(int) # matrix of y < x
             #xy2 = 1 - xy1 # matrix of y = x
             dm = d.reshape((I, 1)) @ np.ones((1, J))
             mpiy = piy.reshape((I, 1)) @ np.ones((1, J))
             dpy = dm * pi / mpiy ** 2
             b_pi = dpy * xy1
             mu = np.max(b_pi, axis = 1).reshape((I, 1)) @ np.ones((1, J))
             lamda = np.maximum(dpy - mu, 0)
             
             '''for i in range(I):
                     set_2, b_pi = [],[0]  # set_2 is the set of j such that y = x
                     for j in range(J):
                         if y_last[i,j] < x_last[j]:
                             b_pi.append(d[i]*pi[i,j]/piy[i]**2)
                         else: 
                             set_2.append(j)
    
                     mu = max(b_pi)  # give the mu value for current i                 
    
                     for j in set_2:
                         lamda[i,j] = max(d[i]*pi[i,j]/piy[i]**2 - mu,0)'''
             
              
            
            
            
            
            ###################################################################### 
             w_vals = model.cbGetSolution(m._w)                           
             # benders cut                    
             for i in range(I):
                 if w_vals[i] < Phi[i]:
                     m.cbLazy(w[i] >= Phi[i] - grb.quicksum(lamda[i,j]*(x[j]-x_last[j]) for j in range(J)))
                     m._number_of_benders_cuts += 1
             # optimality cut seems unnecessary for small problem, but useful for large-scale                            
             m.cbLazy(grb.quicksum(w[i] for i in range(I)) >= np.sum(Phi)*(1- grb.quicksum(x[j] for j in set_of_close)))
             m._number_of_opt_cuts += 1
           
            
            
    m._x = x
    m._w = w
    m._number_of_benders_cuts = 0
    m._number_of_opt_cuts = 0
    
    m.Params.lazyConstraints = 1
    m.optimize(lazy_cut)
    x_last = np.zeros(J)
    for j in range(J):
        x_last[j] = round(x[j].x)
    print("##############################################")
    print("Loss", 1 - m.ObjVal)
    print("Proft", sum(d)-m.ObjVal)
    print("Number of facilities",int(np.sum(x_last)))
    print("Total solve_time = ", round(m.Runtime,1))
    print("Number of branch and cut nodes is ", round(m.NodeCount))
    print("Number of Benders_cuts", m._number_of_benders_cuts)
    print("Number of opt_cuts", m._number_of_opt_cuts)
    print("##############################################")






