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
        #Parametros
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
        D = []
        for i in self.N:
                Di = sorted(set(self.d[i]))
                D.append(Di)
        self.D = D


def build_model(dat : Data, **kwargs):
    m = Model(name='P-MEDIAN', **kwargs)
    
    nk = dat.nk
    N = dat.N #Clientes
    M = dat.M #Facilidades
    K = dat.K
    

    D = dat.D

    y = m.binary_var_list(M, name='y')
    z = m.binary_var_matrix(N, K, name='z')

    m.y = y
    m.z = z
    for chave, lista_valores in nk.items():
        nk[chave] = int(lista_valores[0])

    sum_of_alocation_distances = m.sum(D[i][0] + m.sum((D[i][k] - D[i][k-1]) * z[i,k] for k in range(1,len(K[i])) if k < nk[i]) for i in N)
    m.add_kpi(sum_of_alocation_distances, 'Sum of alocation distances')
    m.minimize(sum_of_alocation_distances)
    
    m.add_constraint(m.sum(y[j] for j in M) == dat.p)
    
    m.add_constraints(z[i,0] + m.sum(y[j] for j in M if dat.d[i][j] == D[i][0])  >= 1 for i in N)
    
    m.add_constraints(z[i,k] + m.sum(y[j] for j in M if dat.d[i][j] == D[i][k-1]) >= z[i,k-1] for i in N for k in K[i] if k>0)
    
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
        df.to_csv("P-median/resultados_p-median_f3.csv",index = False)      