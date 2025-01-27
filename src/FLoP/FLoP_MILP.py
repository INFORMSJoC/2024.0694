import numpy as np
import gurobipy as gb
from gurobipy import GRB
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

def gen_set(i, j):
    ss = []
    for k in range(J):
        if U[i][k] < U[i][j]:
            ss.append(k)
    return ss

def milp():
    mlp = gb.Model('mlp')

    x = mlp.addVars(J, vtype = GRB.BINARY, name = 'x')
    y = mlp.addVars(I, J, ub = 1, name = 'y')
    obj = gb.quicksum(r[i][j] * y[i,j] for i in range(I) for j in range(J)) / I 
    mlp.setObjective(obj, GRB.MAXIMIZE)
    
    mlp.addConstr(x[0] == 1)
    mlp.addConstr(x.sum() == p + 1)
    mlp.addConstrs(gb.quicksum(x[j] for j in range(k * L + 1, k * L + L + 1)) <= 1 for k in range(JJ))
    mlp.addConstrs(gb.quicksum(y[i,j] for j in range(J)) == 1 for i in range(I))
    mlp.addConstrs(y[i,j] <= x[j] for i in range(I) for j in range(J))
    mlp.addConstrs(gb.quicksum(y[i,k] for k in gen_set(i, j)) + x[j] <= 1 for i in range(I) for j in range(J))
    
    mlp.params.TimeLimit = 3600
    mlp.params.OutputFlag = 1
    
    mlp.optimize()
    
    return mlp.Runtime, mlp.NodeCount, mlp.objVal, mlp.MIPgap

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


    time, node, gap = milp()




