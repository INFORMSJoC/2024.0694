import numpy as np
import gurobipy as gb
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


# In[]:
import gurobipy as grb

def BIBC():
    model = grb.Model()
    model.setParam('OutputFlag', 1)
    model.setParam('lazyConstraints',1)

    x = model.addVars(J,vtype=grb.GRB.BINARY)
    y = model.addVars(I,J)

    model.setObjective(grb.quicksum(r[i,j]*y[i,j] for i in range(I) for j in range(J))/I, grb.GRB.MAXIMIZE)

    model.addConstr(x[0] == 1)
    model.addConstr(x.sum() == p + 1)
    model.addConstrs(gb.quicksum(x[j] for j in range(k * L + 1, k * L + L + 1)) <= 1 for k in range(JJ))


    for i in range(I):
        model.addConstr(grb.quicksum(y[i,j] for j in range(J)) == 1)    
        for j in range(J):
            model.addConstr(y[i,j] <= x[j])

    '''This is the preprocessing constraints'''
    ### y_{ij} can be 1 only if U[i,j] is better than the outside option
    for i in range(I):
        for j in range(J):
            if U[i,j] < U[i,0]: 
                model.addConstr(y[i,j] == 0)

    def lazy_cut(model, where):
        if where == grb.GRB.Callback.MIPSOL:
              print("----------- sep -----------")
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


    n = J
    ''' tranform uitlity values into rankings'''
    for i in range(I):
        index = np.argsort(U[i])
        U[i,index] = np.arange(J) + 1



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


    n = J
    ''' tranform uitlity values into rankings'''
    for i in range(I):
        index = np.argsort(U[i])
        U[i,index] = np.arange(J) + 1
        
    time, node, cut, obj, gap = BIBC()



