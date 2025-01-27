# In[ ]:
from math import *
import numpy as np
import gurobipy as grb
import scipy.stats as stats
from scipy.stats import qmc

######################## application 1 data ###############################
II = 100      # number of samples
JJ = 100           # number of assortment

p = 10

M = 10

timelimit = 7200
####################################################
mean_of_normal = 0
std_of_normal = 100

#####################################################
# Generate_AO_data_MC is a function that uses MC to sample utilities
# This function serves as benchmark with large samples, i.e., 10^6 
# That is why we use MC instead of LHS to create benchmarks; otherwise sampling process is intractable
def Generate_AO_data_MC(I,J,seed):
    np.random.seed(13579)
    u_determistic = np.random.uniform(0,100,J)                                #determistic utility
    u_determistic = np.insert(u_determistic,0,50)    
    np.random.seed(24680)
    R = np.random.uniform(0,100,J)                                          #determistic reward
    R = np.insert(R,0,0)
    r = np.ones((I,1))@(R.reshape((1,J+1)))
    ### generate sampling ramdom
    np.random.seed(seed)
    U = np.ones((I,1))@(u_determistic.reshape((1,J+1))) + np.random.normal(size = (I,J+1), loc = mean_of_normal, scale = std_of_normal)
    return(U, r)


def Generate_AO_data_LHS(I,J,seed):
    np.random.seed(13579)
    u_determistic = np.random.randint(1,100,J)                                #determistic utility
    u_determistic = np.insert(u_determistic,0,50)    
    np.random.seed(24680)
    R = np.random.randint(1,100,J)                                          #determistic reward
    R = np.insert(R,0,0)
    r = np.ones((I,1))@(R.reshape((1,J+1)))
    ### generate samplings
    points = (qmc.LatinHypercube(d=J+1,seed=seed)).random(n=I)
    U_random = stats.norm.ppf(points,loc=mean_of_normal, scale=std_of_normal)
    U = np.ones((I,1))@(u_determistic.reshape((1,J+1))) + U_random
    return(U, r)



LS = 1000000
U_large, r_large = Generate_AO_data_MC(LS,JJ,0)
U_large += np.abs(U_large).max() + 1

####################### SAA Loop #############################################
'''SAA Loop. Here, we consider M replications'''
#### Loop for M seeds
Seed_list = [(i+1) for i in range(M)]


Eval_Obj_list = []
SAA_Obj_list = []
Large_Sample_Variance = []
CPU_list = []
x_sol, x_open  = np.zeros(JJ + 1), []


for seed in Seed_list:
    print("  ")
    print("  ")
    print("Current seed:", seed)
    
    J = JJ
    I = II

    U, r = Generate_AO_data_LHS(I,J,seed)

    I, J = U.shape
    ############### preprocessing ####################
    ''' tranform uitlity values into rankings'''
    for i in range(I):
        index = np.argsort(U[i])
        U[i,index] = np.arange(J) + 1    
    for i in range(I):
        for j in range(J):
            if U[i][j] < U[i][0]:
                r[i][j] = 0
    delta = np.zeros((I,J,J))
    for i in range(I):
        for j in range(J):
            delta[i,j] = U[i,j] < U[i]


    model = grb.Model()
    model.setParam('OutputFlag', 1)
    model.setParam('lazyConstraints',1)
    model.Params.Cuts = 3
    
    x = model.addVars(J,vtype=grb.GRB.BINARY)
    y = model.addVars(I,J)
    
    model.setObjective(grb.quicksum(r[i,j]*y[i,j] for i in range(I) for j in range(J))/I, grb.GRB.MAXIMIZE)
    model.addConstr(x[0] == 1)
    model.addConstr(x.sum() ==  p+1)
    
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
    ''' Warm-start the SAA problem (from iteration 2 onwards)'''
    if seed >= 2:
        y_last = np.zeros((I,J))
        for i in range(I):
            index = x_open[np.argmax(U[i][x_open])]
            y_last[i,index] = 1            
        for j in range(J):
            x[j].start = x_sol[j]
            for i in range(I):
                y[i,j].start = y_last[i,j]
    ''' End of Warm-start'''

    
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
    
    model._x = x
    model._y = y
    model._number_of_cuts = 0
    model.optimize(lazy_cut) 


    print("################################################")
    print('total time is',model.Runtime)
    print("Obj is", model.ObjVal)
    ### Get solution
    for j in range(J):
        x_sol[j] = round(x[j].x)  
        if x_sol[j] > 0.5:
           x_open.append(j)

    ########### evalaute the solution using large-sample: U_large, r_large
    Ux = np.ones((LS,1))@x_sol.reshape((1,J))*U_large
    index = np.argmax(Ux,axis=1)
    i_list = np.arange(0,LS)
    obj_eval = r_large[i_list,index[i_list]].sum()/LS
    print('Large Sample Eval:',obj_eval)
    print("difference",round((model.objVal - obj_eval)/obj_eval*100,2),"%")

    ### compute sample variance of large-sample
    Phi = r_large[index[i_list]]
    MEAN = np.mean(Phi)

    Large_Sample_Variance.append(np.sum((Phi - MEAN)**2)/(1000000-1)/1000000)
    SAA_Obj_list.append(model.objVal)
    Eval_Obj_list.append(obj_eval)
    CPU_list.append(model.Runtime)


### record and save the data
final_result = {}
final_result["Seed"] = Seed_list
final_result["SAA Obj"] = SAA_Obj_list
final_result["Eval Obj"] = Eval_Obj_list
final_result["LS Variance"] = Large_Sample_Variance
final_result["CPU"] = CPU_list



###############################################################################
max_index = np.argmax(Eval_Obj_list[:M])
Best_Objective = Eval_Obj_list[:M][max_index]

Average_ObjN =  np.mean(Eval_Obj_list[:M])
Vairance_ObjN = np.sum((np.array(Eval_Obj_list[:M]) - Average_ObjN)**2)/(M-1)/M

Average_MN = np.mean(SAA_Obj_list[:M])
SAA_Vairance = np.sum((np.array(SAA_Obj_list[:M]) - Average_MN)**2)/(M-1)/M + Large_Sample_Variance[max_index]
SAA_Relative_Gap = (Average_MN - Best_Objective)/Best_Objective * 100

print("##################################################################")
print("Best Obj N':", Best_Objective)
print("Worst Obj N':", np.min(Eval_Obj_list[:M]))
print("Average Obj N':", Average_ObjN)
print("Variance Obj N':", Vairance_ObjN)
print("Average MN:", Average_MN)
print("SAA Difference:", round(Average_MN - Best_Objective,4))
print("SAA Relative Gap:", round(SAA_Relative_Gap,2), "%")
print("SAA Variance:", SAA_Vairance)
print("Average CPU:", np.average(CPU_list[:M]))
