# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 00:32:09 2021

@author: Hogan
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List
from abc import ABC, abstractmethod
import datetime
import scipy.stats
import math
from statsmodels.tsa.ar_model import AutoReg
from scipy.special import gamma
import warnings
import numpy as np
from statsmodels.tsa import ar_model
from scipy import stats
import statsmodels.tsa.stattools as ts
import scipy.optimize as opt
from scipy.stats import ks_2samp
import random
import statsmodels.api as sm
warnings.filterwarnings('ignore')

#%% Parkinson Ratio and Parkinson Score

def parkinson_ratio(series: pd.Series, interval: int=25, detrend=False):
    Parkinson_volatility, traditional_volatility = 0, 0
    l, D = np.array([]), np.array([])
    n = int(len(series)/interval)
    if detrend==True:
        data = detrend_series(series)
    else:
        data = series
    for i in range(n):
        period = data[interval*i:interval*(i+1)]
        price_highest, price_lowest, price_open, price_close = period.max(), period.min(), period.iloc[0], period.iloc[-1]
        l = np.append(l, price_highest - price_lowest) # Make it clearer
        D = np.append(D, price_close - price_open)

    return math.sqrt((l**2).mean()/(D**2).mean()/(4*np.log(2)))

def get_rolling_parkinson_ratio_for_series(series: pd.Series, lookbackperiod: int=500, interval: int=20):
    parkinsons = series.rolling(lookbackperiod).apply(parkinson_ratio, args=(interval,False)).dropna()
    parkinsons = pd.DataFrame(data=parkinsons, index=series.index[-len(parkinsons):], columns=['Parkinson Ratio'])

    return parkinsons


#%% Hurst Exponent


def hurst_exponent(series: pd.Series, max_lag):
    lags = range(2, max_lag)
    tau =[np.std(np.subtract(np.array(series.iloc[lag:]), np.array(series.iloc[:-lag]))) for lag in lags]
    reg = sm.OLS(np.log(tau), sm.add_constant(np.log(lags))).fit()
    hurst = reg.params[1]
    #t = (reg.params[1]-0.5)/reg.bse[1]
    #z_score = scipy.stats.norm(0, 1).cdf(t)

    return hurst#, z_score
    
def get_rolling_hurst_exponent_for_series(series: pd.Series, max_lag: int=20, lookbackperiod: int=250):
    
    hursts = np.array([])
    z_scores = np.array([])
    
    for t in range(lookbackperiod, len(series)+1):
        data = series.iloc[t-lookbackperiod:t]
        hurst = hurst_exponent(data, max_lag)
        hursts = np.append(hursts, hurst)
    res = pd.DataFrame(data=np.nan, index=series.index[-len(hursts):], columns=['Hurst Exponent','Z Score'])
    res.iloc[:,0]=hursts
    res.iloc[:,1]=z_scores
    
    return res


#%% Find linear combination of series that form mean reverting spread
"""
This function uses series as inputs and generates a vector w, the linear combination of the series, which gives us the best mean reverting spread.
We optimize the Parkinson Ratio to get w.
"""
def form_linear_combination(data: pd.DataFrame, w: list)->pd.Series:
    # The reason we take dataframe instead of series as input is to automatically get the number of series and align index
    spread = pd.Series(data.dot(np.array(w)))
    spread.index=data.index
    return spread

def generate_mean_reverting_linear_combination(data: pd.DataFrame, interval: int=25, detrend=False, display=False):
    # Require at least two series input, all series must have the same index

    if len(data.dropna().index)<10*interval:
        if display == True:
            print("Data length is too short comparing to interval")
        return None
        
    n = len(data.columns)
    w = [random.uniform(-1, 1) for i in range(n)]
    
    train=data.copy().dropna()
    train['Zero']=0.0
    model = sm.OLS(train['Zero'],sm.add_constant(train[data.columns]))
    result = model.fit()
    w = result.params[1:]
    
    def temp(weight):
        if detrend==False:
            spread = form_linear_combination(data.dropna(), weight)
        else:
            spread = detrend_series(form_linear_combination(data.dropna(), weight))
        #res = parkinson_ratio(series=spread, interval=interval)
        #lookbackperiod = len(spread.index)-5
        #parkinsons = spread.rolling(lookbackperiod).apply(parkinson_ratio, args=(interval,False)).dropna()
        return ts.adfuller(spread,regression='c')[1]
    w_opt, ratio = None, None
    w_opt = opt.fmin(func=temp, x0=w, disp=False)
    spread = form_linear_combination(data.dropna(), w_opt)
    ratio = parkinson_ratio(series=spread, interval=interval)
    adf_pvalue = ts.adfuller(spread, regression='c')[1]
    return w_opt, ratio, adf_pvalue



def generate_trending_linear_combination(data: pd.DataFrame, interval: int=25, detrend=False, display=False):
    # Require at least two series input, all series must have the same index
    if len(data.index)<10*interval:
        if display == True:
            print("Data length is too short comparing to interval")
        return None
        
    n = len(data.columns)
    w = [1 for i in range(n)]
    
    def temp(weight):
        if detrend==False:
            spread = form_linear_combination(data, weight)
        else:
            spread = detrend_series(form_linear_combination(data, weight))
        res = parkinson_ratio(series=spread, interval=interval)
        return res #+ ts.adfuller(spread)[1]**2
    w_opt, ratio = None, None
    w_opt = opt.fmin(func=temp, x0=w, disp=False)
    spread = form_linear_combination(data, w_opt)
    ratio = parkinson_ratio(series=spread, interval=interval)
    adf_pvalue = ts.adfuller(spread)[1]
    return w_opt, ratio, adf_pvalue


#%% Some random stuff for experimental use

def generate_arithmetic_brownian_series(miu, sigma, length, start=0):
    trending_series = np.array([])
    U=start
    for t in range(1,length+1):
        trending_series = np.append(trending_series, U)
        dU = miu + sigma*np.random.normal(0,1)
        U += dU
    return pd.Series(trending_series)

def generate_O_U_series(theta, alpha, mu, sigma, length, start=0):
    mean_reverting_series = np.array([])
    U = 100
    for i in range(1,length+1):
        mean_reverting_series = np.append(mean_reverting_series, U)
        deltaU = theta*(alpha+mu*i-U)+sigma*np.random.normal(0,1)
        U += deltaU
    return pd.Series(mean_reverting_series)

#%% Temporal regression and detrend
def temporal_regression(series: pd.Series):
    
    data = pd.DataFrame(data=np.nan, index=series.index, columns=['X','y'])
    data.iloc[:,0] = [i+1 for i in range(len(series))]
    data.iloc[:,1] = series
    X = data.dropna().iloc[:,0]
    y = data.dropna().iloc[:,1]
    
    regression = sm.OLS(y,sm.add_constant(X)).fit()
    const = regression.params[0]
    slope = regression.params[1]
    sigma = regression.bse[1]
    pvalue = regression.pvalues[1]
    t = (slope)/regression.bse[1]
    z_score = scipy.stats.norm(0, 1).cdf(t)
    return slope, sigma, const


def detrend_series(series: pd.Series, slope=None, const=None, remove_const=True):
    if slope is None and const is None:
        slope, z_score, const = temporal_regression(series)
    trend = np.array([slope for i in range(len(series))])
    if remove_const==False:
        const=0
    return series - trend.cumsum() - const
    