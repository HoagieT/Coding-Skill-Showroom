# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 23:31:46 2021

@author: Hogan
"""
import math

def cox_stuart_test(series: pd.Series):
    c = int(len(series)/2)
    count = 0
    for i in range(c):
        if series.iloc[i+c]>series.iloc[i]:
            count += 1
        elif series.iloc[i+c]<series.iloc[i]:
            count -= 1
    return count/c#, negative_count/c

def buishand_range_test(series: pd.Series):
    mean = series.mean()
    sigma = series.std()
    s = (series-mean).cumsum()
    s_max, s_min = s.max(), s.min()
    return (s_max-s_min)/sigma/math.sqrt(len(series))

def pettitt_test(series: pd.Series):
    
    return

def sens_slope(series: pd.Series):
    slopes = np.array([])
    for i, j in combinations([s for s in range(len(series))], 2):
        slope = (series.iloc[j] - series.iloc[i])/(j-i)
        slopes = np.append(slopes, slope)
    return pd.Series(slopes).median()

def macd(series, long_period, short_period, signal_period):
    macd = series.rolling(short_period).mean()-series.rolling(long_period).mean()
    macd = macd - macd.rolling(signal_period).mean()
    return macd

def ppo(series, long_period, short_period, signal_period):
    macd = (series.rolling(short_period).mean()-series.rolling(long_period).mean())/series.rolling(long_period).mean()*100
    #macd = macd - macd.rolling(signal_period).mean()
    return macd


#%% The TUBE Algorithm

class TubeAlgorithm():
    