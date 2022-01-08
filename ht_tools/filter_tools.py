# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 23:46:19 2021

@author: Hogan
"""

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import tkinter.filedialog
import datetime 
import math 
import time
import calendar
import matplotlib.mlab as mlab
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from datetime import datetime,timedelta
from scipy.optimize import minimize, rosen, rosen_der
from statsmodels.tsa.ar_model import AR
from scipy.stats import kurtosis
from scipy.stats import skew
from tqdm import tqdm

#%% Kalman Filter

def initial_estimate(data: pd.Series):
    # x_k = A + B*x_{k-1} +C*e_k
    # y_k = x_k + D*w_k
    state_equation = AR(data).fit(1)
    A,B = state_equation.params
    C = state_equation.resid.std()
    D = (data - A/(1-B)).std()
    return A, B, C, D

def single_dimension_kalman_filter(y: pd.Series, A: float, B: float, C: float, D: float, smoother=None):
    # Declare the key outputs
    x_priori = np.array([y.iloc[0]])
    x_posteriori = np.array([y.iloc[0]])
    R_priori = np.array([D**2])
    R_posteriori = np.array([D**2])
    kalman_gain = np.array([np.nan])
    
    for i in range(1, len(y)):
        # Predict
        x_minus = float(A + B*x_posteriori[i-1])
        R_minus = float(B**2*R_posteriori[i-1] +C**2)
        
        # Update
        K = float(R_minus/(R_minus+D**2))
        x = float(x_minus + K*(y.iloc[i]-x_minus))
        R = R_minus - K*R_minus

        # Append
        x_priori = np.append(x_priori, x_minus)
        x_posteriori = np.append(x_posteriori, x)
        R_priori = np.append(R_priori, R_minus)
        R_posteriori = np.append(R_posteriori, R)
        kalman_gain = np.append(kalman_gain, K)
        
    x_priori = pd.Series(data=x_priori, index=y.index)
    x_posteriori = pd.Series(data=x_posteriori, index=y.index)
    R_priori = pd.Series(data=R_priori, index=y.index)
    R_posteriori = pd.Series(data=R_posteriori, index=y.index)
    kalman_gain = pd.Series(data=kalman_gain, index=y.index)
    
    if smoother == 'shumway stoffer':
        x_posteriori, R_priori, R_posteriori, A, B, C, D = shumway_stoffer_smoother(A, B, C, D, x_posteriori, R_priori, R_posteriori, kalman_gain, y)
    
    # Output order: x_priori, x_posteriori, R_priori, R_posteriori, kalman_gain, A, B, C, D
    return x_priori, x_posteriori, R_priori, R_posteriori, kalman_gain, A, B, C, D

def shumway_stoffer_smoother(A: float, B: float, C: float, D: float, x_posteriori: pd.Series, R_priori: pd.Series, R_posteriori: pd.Series, kalman_gain: pd.Series, y: pd.Series):
    N = len(x_posteriori)
    x_ss = pd.Series([0 for i in range(N)])
    R_ss = pd.Series([0 for i in range(N)])
    R_ss_priori = pd.Series([0 for i in range(N)])
    J = pd.Series([0 for i in range(N)])
    
    x_ss.iloc[N-1] = x_posteriori[N-1]
    R_ss.iloc[N-1] = R_posteriori[N-1]
    R_ss_priori.iloc[N-1] = R_priori[N-1]

    R_ss_priori.iloc[N-2] = B*(1-kalman_gain[N-1])*R_posteriori[N-2]
    
    # Smoothing step
    for i in range(N-2,-1,-1):
        J.iloc[i] = B*R_posteriori[i]/R_priori[i+1]
        x_ss.iloc[i] = x_posteriori[i] + J[i]*(x_ss[i+1]-(A+B*x_posteriori[i]))
        R_ss.iloc[i] = R_posteriori[i] + J[i]**2*(R_ss[i+1]-R_priori[i+1])
        
    for i in range(N-3,-1,-1):
        R_ss_priori.iloc[i] = J[i]*R_posteriori[i+1] + J[i]*J[i+1]*(R_ss_priori[i+1]-B*R_posteriori[i+1])
    
    x_ss.index = y.index
    R_ss_priori.index = y.index  
    R_ss.index = y.index
    
    state_equation = AR(x_ss).fit(1)
    A_hat, B_hat = state_equation.params
    C_hat = state_equation.resid.std()
    D_hat = (y - state_equation.fittedvalues).dropna().std()

    
    """
    # Estimation of parameters
    alpha = sum(R_ss[:-1] + x_ss[:i-1]**2)
    beta = sum(R_ss_priori[:-1].values + x_ss[:-1].values*x_ss[1:].values)
    gamma = sum(x_ss[1:])
    delta = gamma - x_ss[N-1] + x_ss[0]
    
    A_hat = (alpha*gamma-delta*beta)/(N*alpha-delta**2)
    B_hat = (N*beta-gamma*delta)/(N*alpha-delta**2)
    
    C_hat_sq = 0.0
    D_hat_sq = 0.0
    
    for i in range(1,N):
        C_hat_sq = C_hat_sq + (R_ss[i] + x_ss[i]**2 + A_hat**2 + B_hat**2*R_ss[i-1] + B_hat**2*x_ss[i-1]**2 - 2*A_hat*x_ss[i] + 2*A_hat*B_hat*x_ss[i-1] -2*B_hat*R_ss_priori[i-1] - 2*B_hat*x_ss[i]*x_ss[i-1])/(N-1)
    for i in range(N):
        D_hat_sq = D_hat_sq + (y[i]**2 - 2*y[i]*x_ss[i] + R_ss[i] + x_ss[i]**2)/N
    """    
     
    
    # Output order: x_posteriori, R_priori, R_posteriori
    return x_ss, R_ss_priori, R_ss, A_hat, B_hat, C_hat, D_hat

class KalmanFilter():
    """
    Args:
        y: observed values, same as data
        x_posteriori: posteriori estimation, i.e. Kalman filtered values
        x_priori: priori estimation
    """
    def __init__(self, data: pd.Series, y=None, adf_pvalue=None, x_priori=None, x_posteriori=None, kalman_gain=None, parameters=None):
        self.data = data
        self.adf_pvalue = ts.adfuller(data)[1]
    
    def fit(self, n_iteration: int=0, smoother='shumway stoffer'):
        A,B,C,D = initial_estimate(self.data)
        x_priori, x_posteriori, R_priori, R_posteriori, kalman_gain, A, B, C, D = single_dimension_kalman_filter(self.data, A, B, C, D, smoother)
        
        if n_iteration > 0:
            for i in range(n_iteration):
                x_priori, x_posteriori, R_priori, R_posteriori, kalman_gain, A, B, C, D = single_dimension_kalman_filter(self.data, A, B, C, D, smoother)
                
        self.x_priori = x_priori
        self.x_posteriori = x_posteriori
        self.R_priori = R_priori
        self.R_posteriori = R_posteriori
        self.kalman_gain = kalman_gain
        self.A, self.B, self.C, self.D = A, B, C, D
        return True
    
    def plot(self, show='priori', figure_size=(16,9), deviation_pct: float=None, title=None):
        plt.style.use('seaborn')
        fig=plt.figure()
        plt.rcParams['figure.figsize'] = (figure_size[0], figure_size[1])
        
        plt.plot(self.data, label='Observed Value')
        
        if show=='priori':
            estimate, label = self.x_priori, 'Priori Estimate'
        if show=='posteriori':
            estimate, label = self.x_posteriori, 'Posteriori Estimate'
        plt.plot(estimate, label=label)
        
        if deviation_pct is not None:
            deviation = (self.data - estimate).quantile(deviation_pct)
            plt.fill_between(self.data.index,estimate-deviation, estimate+deviation, alpha=0.2, label=str(int(deviation_pct*100))+'% percentile deviation')
            
        plt.legend()
        if title is not None:
            plt.title(title)

        plt.show()
        return fig

#%% DESP Filter

def single_dimension_desp_filter(y: pd.Series, alpha: float=0.5):
    # Declare the key outputs
    filter1 = np.array([y.iloc[0]])
    filter2 = np.array([y.iloc[0]])
    level = np.array([y.iloc[0]])
    slope = np.array([0])
    y_priori = np.array([y.iloc[0]])
    
    for t in range(1, len(y)):
        # Double filtering
        s1 = alpha*y[t] + (1-alpha)*filter1[t-1]
        s2 = alpha*s1 + (1-alpha)*filter2[t-1]
        
        # Intercept and slope
        b1 = alpha/(1-alpha)*(s1-s2)
        b0 = 2*s1 - s2 - t*b1
        
        # Predict
        y_minus = (2+alpha/(1-alpha))*filter1[t-1] - (1+alpha/(1-alpha))*filter2[t-1]
        
        # Append
        filter1 = np.append(filter1, s1)
        filter2 = np.append(filter2, s2)
        level = np.append(level, b0)
        slope = np.append(slope, b0)
        y_priori = np.append(y_priori, y_minus)
        
    # Output order: filter1, filter2, y_priori, level, slope
    return pd.Series(data=filter1, index=y.index), pd.Series(data=filter2, index=y.index), pd.Series(data=y_priori, index=y.index)#, pd.Series(data=level, index=y.index), pd.Series(data=slope, index=y.index)


class DESPFilter():
    
    def __init__(self, data: pd.Series, y_priori=None, filter1=None, filter2=None, alpha=None):
        self.data = data

    def fit(self, alpha=0.025):
        if alpha<0 or alpha>1:
            raise TypeError('alpha must be in (0,1)')

        self.filter1, self.filter2, self.y_priori = single_dimension_desp_filter(self.data, alpha)
        self.alpha = alpha

        return
    
    def plot(self, show='priori', figure_size=(16,9), deviation_pct: float=None, title=None):
        plt.style.use('seaborn')
        fig=plt.figure()
        plt.rcParams['figure.figsize'] = (figure_size[0], figure_size[1])
        
        plt.plot(self.data, label='Observed Value')
        
        if show=='priori':
            estimate, label = self.y_priori, 'Priori Estimate'
        if show=='filter1':
            estimate, label = self.filter1, 'First Filter'
        if show=='filter2':
            estimate, label = self.filter2, 'Second Filter'
        plt.plot(estimate, label=label)
        
        if deviation_pct is not None:
            deviation = (self.data - estimate).quantile(deviation_pct)
            plt.fill_between(self.data.index,estimate-deviation, estimate+deviation, alpha=0.2, label=str(int(deviation_pct*100))+'% percentile deviation')
            
        plt.legend()
        if title is not None:
            plt.title(title)

        plt.show()
        return fig
    
#%% Median Filter

def median_filter(series: pd.Series, window: int):
    result = series.rolling(window).median()
    return result

class MedianFilter():
    def __init__(self, data: pd.Series, window: int):
        self.data = data
        self.window = window
        
