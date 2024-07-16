
import pandas as pd
import numpy as np
from scipy import optimize


"""
Contains the definition of abstract class ScenarioGen
"""
from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Contains abstract methods for scenario generation
    """

    @abstractmethod
    def get_optimal_allocations(self,*args,**kwargs):
        """
        Get the Optimal weights 
        """





class CvarMretOpt(Strategy):


    def __init__(self,ratio=0.5,risk_level=0.3):


        self.ratio = ratio
        self.risk_level = risk_level
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):
        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount 
        self.results = self.optimize(self.ratio,self.risk_level)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]
    
    def get_cvar_value(self):
        return self.results.x[-1]

    def optimize(self,ratio,risk_level):

        """Solve the problem of minimizing the function 
                -(1-c) E[Z(x)] + c AVaR[Z(x)]
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))

        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets+1))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) # Rk  
            lhs_ineq[i,-1] = 1    # n

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets+1))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets+1):
            bnd.append((0,float('inf')))

        bnd[-1] = (float('-inf'),float('inf'))

        obj = np.ones((1,self.num_senarios+self.num_assets+1))*(1/risk_level)*(1/self.num_senarios)*(ratio)
        obj[0,-1] = -1*(ratio)
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*(1-ratio)*np.array(np.transpose(mean))
        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")

        return opt

        

class MeanSemidevOpt(Strategy):

    def __init__(self,ratio):
       
        self.ratio = ratio 
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount
        self.results = self.optimize(self.ratio)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]

    def optimize(self,ratio):

        
        """
            ρ =  Hybrid Risk Measure
            ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

            where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))
        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) + np.transpose(mean)  # Rk

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets):
            bnd.append((0,float('inf')))

        obj = np.ones((1,self.num_senarios+self.num_assets))*(1/self.num_senarios)*ratio
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*np.array(np.transpose(mean))

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt
        


class EqualyWeighted(Strategy):

    def __init__(self):
    
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        return (np.ones((1,self.num_assets))/self.num_assets)*investment_amount
    
    
    