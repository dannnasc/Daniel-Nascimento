import sys
import os
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

np.random.seed(0)
GRIDSIZE = 1000
class Data():
    def __init__(self,ni,nj,p):
        self.ni = ni
        self.nj = nj
        self.p = p
        self.icoord = np.random.randint(1, high=GRIDSIZE, size=(ni,2))
        self.jcoord = np.random.randint(1, high=GRIDSIZE, size=(nj,2))
        d = self.icoord[:,np.newaxis,:] - self.jcoord[np.newaxis,:,:]
        self.d = np.round(np.linalg.norm(d, axis=-1)).astype(int)
        I = range(self.ni) #Clientes
        J = range(self.nj) #Facilidades
        self.I = I
        self.J = J

def build_model(dat : Data, **kwargs):
    m = Model(name='p-median', **kwargs)
    I = dat.I
    J = dat.J

    
    y = m.binary_var_list(J, name='y')
    x = m.continuous_var_matrix(I, J, lb=0, ub=1, name='x')
    m.x = x
    m.y = y
    
    total_transportation_cost = m.sum(x[i, j] * dat.d[i][j] for (i,j) in product(I,J))
    m.add_kpi(total_transportation_cost, 'Total transportation cost')
    m.minimize(total_transportation_cost)
    
    m.add_constraint(m.sum(y[j] for j in J) == dat.p)
    
    m.add_constraints(m.sum(x[i, j] for j in J) == 1 for i in I)
    
    m.add_constraints(x[i, j] <= y[j] for (i,j) in product(I,J))
    
    return m

   
if __name__ == "__main__":
    # parse args
    #Instâncias de :
    #100-10-2
    #100-50-10
    #1000-100-20
    #1000-500-100
    #2000-1000-500
    nb_nodes_processed_list = []
    gap_list = [] 
    time_list = []
    linear_gap_list = [] 
    of_value = []     
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
    
    for ins in instances:
        
        ni,nj,p = ins[0],ins[1],ins[2] 
        dat = Data(ni,nj,p)
        m = build_model(dat)
        m.print_information()
    
        s = m.solve(log_output=False)
        assert s
        sd = m.solve_details
        
                #Desligar cortes
        
        m.parameters.preprocessing.presolve = 0
        params = m.parameters
        params.threads = 8
        params.mip.strategy.heuristicfreq = -1
        params.mip.limits.aggforcut = 0
        params.mip.limits.cutsfactor = 0
        params.mip.cuts.mircut = -1
        params.mip.cuts.implied = -1
        params.mip.cuts.gomory = -1
        params.mip.cuts.flowcovers = -1
        params.mip.cuts.pathcut = -1
        params.mip.cuts.liftproj = -1
        params.mip.cuts.zerohalfcut = -1
        params.mip.cuts.cliques = -1
        params.mip.cuts.covers = -1
         

        nb_nodes_processed_list.append(sd.nb_nodes_processed)
        gap_list.append(sd.gap)
        time_list.append(sd.time)
        
        lp = LinearRelaxer.make_relaxed_model(m)
        m.set_time_limit(600)
        s_lp = lp.solve(log_output=False)
        assert s_lp
        lb = lp.objective_value
        
        ub = m.objective_value
        
        of_value.append(m.objective_value)
        data = {
                #'N° de Instâncias': number_of_instances,
                'Nós de Branch and Bound': nb_nodes_processed_list,
                'GAP': gap_list,
                'Time': time_list,
                'Objetive Function': of_value
            }
        df = pd.DataFrame(data)
        df.to_csv("P-median/resultados_p-median_f1.csv",index = False)   