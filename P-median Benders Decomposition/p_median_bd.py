import sys
import os
import cplex
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from itertools import product
import sys

from docplex.mp.relax_linear import LinearRelaxer

from cplex.callbacks import LazyConstraintCallback
from cplex.callbacks import UserCutCallback

from docplex.mp.callbacks.cb_mixin import *
from docplex.mp.model import Model
from docplex.mp.constants import ComparisonType

from timeit import default_timer as timer


np.random.seed(0)
GRIDSIZE = 1000
class Data():
    def __init__(self,ni,nj,p):
        self.ni = ni
        self.nj = nj
        self.p = p
        self.N = set(range(ni)) #Clientes
        self.M = set(range(nj)) #Facilidades
        self.icoord = np.random.randint(1, high=GRIDSIZE, size=(ni, 2))
        self.jcoord = np.random.randint(1, high=GRIDSIZE, size=(nj, 2))
        d = self.icoord[:, np.newaxis, :] - self.jcoord[np.newaxis, :, :]
        self.d = np.round(np.linalg.norm(d, axis=-1)).astype(int)
        nk={}
        K={}
        for i in self.N:
            # nk is the number of different distances from i to any site
            Ki = list()
            Ki.append(len(set(self.d[i])))
            nk[i]=Ki 
        self.nk=nk
        for i in self.N:
            for e in self.nk[i]:
                temp = e
            K[i] = list(set(range(temp)))    
        self.K = K
        D = [ [(self.d[i][j],j) for j in self.M] for i in self.N ]
        for i in self.N:
            D[i] = sorted(D[i])
        self.D = D
      
        
def build_model(dat : Data, **kwargs):
    nk = dat.nk
    N = dat.N
    D = dat.D
    M = dat.M #Facilidades
    K = dat.K
    
    pmp = Model(name='p master problem benders based formulation\n', **kwargs)
    theta = pmp.continuous_var_list(N, lb=[D[i][0][0] for i in N], name='theta')
    y = pmp.binary_var_list(M, name='y')
    pmp.y = y
    pmp.theta = theta

    for chave, lista_valores in nk.items():
        nk[chave] = int(lista_valores[0])

    pmp.minimize(pmp.sum(theta[i] for i in N))
    
    pmp.add_constraint(pmp.sum(y[j] for j in M) == dat.p)
    
    #pmp.export_as_lp('pmp.lp')
    return pmp

def solve_subproblem(dat : Data, m: Model,_y,_theta):
    '''
            let{k in K[i]} _z[i,k] := max(0,1-sum{j in J: c[i,j] <= D[i,k]} _y[j]);
        # finding _k_i 
        let _k := 0;
        let lr := 1;
        let val := 1 - _y[S[i,lr]];
        repeat while (val > 1e-6 and lr < nj)
        {
            if c[i,S[i,lr+1]] > c[i,S[i,lr]] then
               let _k := _k + 1;
               
            let lr := lr + 1;
            let val := val - _y[S[i,lr]];   
        }
        
        if _k == 0 then
        {
            let{j in J} a[nbc,i,j] := 0;
            let infi := D[i,1];
        }
        else
        {
            for{j in J}
            {
                if c[i,j] <= D[i,_k] then
                   let  a[nbc,i,j] := D[i,_k+1] - c[i,j];
                else
                   let  a[nbc,i,j] := 0;
            }
            let infi := D[i,_k+1] - sum{j in J: c[i,j] <= D[i,_k]} (D[i,_k+1] - c[i,j]) * _y[j];            
        }
        
        let b[nbc,i] := D[i,_k + 1];

    '''
    D,ni,nj,d,M,N,K = dat.D,dat.ni,dat.nj,dat.d,dat.M,dat.N,dat.K
    sup = 0.0    
    
    for i in N:
       _k = finding_k(dat,_y,i)
       a = [0.0] * nj
       if _k == -1:
          a = [0.0] * nj
          infi = D[i][0][0] 
       else:
          for j in M:
             if d[i][j] <= D[i][_k][0]:
                a[j] =  (D[i][_k+1][0] - d[i][j])
             else:
                a[j] = 0.0   
          infi = D[i][_k+1][0] - sum( (D[i][_k+1][0] - d[i][j]) * _y[j] for j in M if d[i][j] <= D[i][_k][0]) ;
       sup += infi
       b = D[i][_k + 1][0]
       if _theta[i] < infi:
          m.add_constraint( m.theta[i] >= b - m.sum( a[j] * m.y[j] for j in M ) )
    return sup,_k
    
def finding_k(dat,_y,i):
    D,ni,nj,d,M,N,K = dat.D,dat.ni,dat.nj,dat.d,dat.M,dat.N,dat.K
    tempsum = sum(_y[j] for j in M if d[i][j] == D[i][0][0])
    if tempsum >= 1:
       k = -1
    else:
       listOfKs = [ k for k in K[i] if sum(_y[j] for j in M if d[i][j] <= D[i][k][0]) < 1 ]
       k = max(listOfKs)
    
    return k    
def solve_bd(dat: Data, m : Model):
    nk = dat.nk
    N = dat.N
    D = dat.D
    M = dat.M #Facilidades
    K = dat.K
    ub = cplex.infinity
    lb = 0.0
    gap = 100.0
    stop = False
    it = 0
    while(stop == False):
         it += 1
         # solving the master problem
         #s = m.solve(log_output=True)
         s = m.solve(log_output=False)
         sd = m.solve_details
         
         lb = s.objective_value   
         _y = s.get_value_list(m.y)
         _theta = s.get_value_list(m.theta)
         
         # solving the subproblem
         sup,k = solve_subproblem(dat,m,_y,_theta)
         
         ub = min(sup,ub)
         gap = 100.0 * (ub - lb)/ub
         nbb=sd.nb_nodes_processed
         if gap < 1e-6:
            stop = True
         print('{:d} '.format(it),end=' ')
         print('{:10.3f} '.format(lb),end=' ')
         print('{:10.3f} '.format(ub),end=' ')   
         print('{:10.3f} '.format(gap),end=' ')   
         print('{:10.3f} '.format(sup),end=' ') 
         
         
         print()  

    return nbb,lb,sup,ub,gap,it
 
            
if __name__ == "__main__":
    # parse args
    #Instâncias de :
    #100-10-2
    #100-50-10
    #500-50-10
    #500-100-10
    #1000-100-20
    #1000-500-100
    #2000-1000-500
    #2000-1500-500
    time_bd = [] 
    instances = [
    [100, 10, 5],
    [100, 50, 10],
    [500, 50, 10],
    [500, 100, 10],
    [1000, 100, 50],
    [1000, 500, 50],
    [1000, 500, 100],
    [2000, 1000, 500],
    [2000, 1500, 500]
    ]
    
    nb_nodes_processed_list = []
    gap_list = [] 
    time_list = []
    of_value = []
    bd_iterations = []  

    for ins in instances:
        ni,nj,p = ins[0],ins[1],ins[2] 
        dat = Data(ni,nj,p)     
        #original_model(dat)   
        m = build_model(dat)
        print(ins)
        #m.print_information()
        start = timer()
        nbb,lb,sup,ub,gap,it  = solve_bd(dat,m)
        end = timer()    
        print(gap)
        time = end-start
        
        nb_nodes_processed_list.append(nbb)
        gap_list.append(gap)
        time_list.append(time)
        of_value.append(lb)
        bd_iterations.append(it)
        data = {
                #'N° de Instâncias': number_of_instances,
                'Nós de Branch and Bound': nb_nodes_processed_list,
                'GAP': gap_list,
                'Time': time_list,
                'Objetive Function': of_value,
                'Benders iterations' : bd_iterations
            }
        df = pd.DataFrame(data)
        df.to_csv("P-median/resultados_p-median_bd.csv",index = False)  
     