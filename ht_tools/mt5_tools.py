# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 00:40:59 2021

@author: Hogan
"""

from typing import List
import MetaTrader5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pytz
from datetime import datetime, timezone, timedelta, date
import ht_tools.mean_reversion_tools as mrt
import statsmodels.tsa.stattools as ts
import time
from scipy import stats
import csv
import glob
import random
import os.path
import math

frequency={
    'M1': MetaTrader5.TIMEFRAME_M1,
    'M2': MetaTrader5.TIMEFRAME_M2,
    'M3': MetaTrader5.TIMEFRAME_M3,
    'M5': MetaTrader5.TIMEFRAME_M5,
    'M10': MetaTrader5.TIMEFRAME_M10,
    'M15': MetaTrader5.TIMEFRAME_M15,
    'M20': MetaTrader5.TIMEFRAME_M20,
    'M30': MetaTrader5.TIMEFRAME_M30,
    'H1': MetaTrader5.TIMEFRAME_H1,
    'H2': MetaTrader5.TIMEFRAME_H2,
    'H4': MetaTrader5.TIMEFRAME_H4,
    'H6': MetaTrader5.TIMEFRAME_H6,
    'H8': MetaTrader5.TIMEFRAME_H8,
    'H12': MetaTrader5.TIMEFRAME_H12,
    'D1': MetaTrader5.TIMEFRAME_D1,
    }

order_type={
    -1: MetaTrader5.ORDER_TYPE_SELL,
    1: MetaTrader5.ORDER_TYPE_BUY,
    }

filling_type={
    1: MetaTrader5.ORDER_FILLING_FOK,
    2: MetaTrader5.ORDER_FILLING_IOC,
    }

pytz.timezone("UTC")

leverage = 10
timezone_adjustment=2


#%% MT5 account info
balance=None
equity=None
free_margin=None
margin=None


#%% Trade log
trade_log = pd.DataFrame(columns=['Trade ID','Name','Open Time','Close Time','Open Price','Close Price','Lot','P/L','Comment'])
error_log=open("error_log.txt", "a")

def retrieve_trade_log(file='log.csv'):
    global trade_log
    if os.path.isfile(file):
        trade_log = pd.read_csv(file)
    else:
        trade_log = pd.DataFrame(columns=['Trade ID','Name','Open Time','Close Time','Open Price','Close Price','Lot','P/L','Comment'])
    return

def save_trade_log(file='log.csv'):
    trade_log.to_csv(file, index=False)
    return

def update_account_history(file='account_history.csv'):
    global balance, equity
    get_account_info()
    if os.path.isfile(file):
        account_history = pd.read_csv(file)
    else:
        account_history = pd.DataFrame(columns=['Date','Balance','Equity'])
    checkpoint = datetime.strftime(date(datetime.now().year, datetime.now().month,datetime.now().day), '%Y-%m-%d')
    if len(account_history[account_history['Date']==checkpoint])!=0:
        account_history[account_history['Date']==checkpoint]=[checkpoint, balance,equity]
    else:
        account_history.loc[len(account_history.index)]=[checkpoint, balance, equity]
    account_history.to_csv(file, index=False)
    return

#%% Log in and shut down
def log_in(account: int, password: str):
    MetaTrader5.initialize()
    authorized=MetaTrader5.login(account, password=password)
    if authorized:
        print('Connection established')
        get_account_info()
    else:
        print("failed to connect at account #{}, error code: {}".format(account, MetaTrader5.last_error()))
    return authorized

def shut_down():
    return MetaTrader5.shutdown()

def get_account_info():
    info = MetaTrader5.account_info()
    global balance, equity, free_margin, margin
    balance = info.balance
    equity = info.equity
    free_margin = info.margin_free
    margin = info.margin
    return



#%% Get price and other information of the symbol/ticker

def get_price_history_for_ticker(ticker: str, n_obs: int, freq: str, end_time: pd.Timestamp = datetime.now()) -> pd.Series:
    timeframe=frequency.get(freq)
    symbol_info_dict = MetaTrader5.symbol_info(ticker)._asdict()
    contract_size=symbol_info_dict.get('trade_contract_size')
    unit_value = symbol_info_dict.get('trade_tick_value')
    rates = MetaTrader5.copy_rates_from(ticker, timeframe, end_time, n_obs)
    close_price = pd.DataFrame(rates).iloc[:,4]*contract_size
    close_price.index = pd.to_datetime(pd.DataFrame(rates)['time'], unit='s')
    close_price.name = ticker
    
    # bid and ask prices
    return close_price


def get_price_history_for_ticker_list(ticker_list: list, n_obs: int, freq: str, end_time: pd.Timestamp = datetime.now()) -> pd.DataFrame:
    timeframe=frequency.get(freq)
    series = [0 for i in range(len(ticker_list))]
    for i in range(len(ticker_list)):
        series[i] = get_price_history_for_ticker(ticker_list[i], n_obs, freq, end_time)
    close = pd.concat(series, axis=1, ignore_index=True)
    close.columns=ticker_list
    
    return close


def get_latest_tick_for_ticker(ticker: str):
    lasttick = MetaTrader5.symbol_info_tick(ticker)
    return lasttick.bid, lasttick.ask

def get_request_result_for_trade(ticker: str, lot: float, deviation=100):
    price = get_latest_tick_for_ticker(ticker)[lot>0]
    request = {
        "action": MetaTrader5.TRADE_ACTION_DEAL,
        "symbol": ticker,
        "volume": np.round(float(lot),2),
        "price": price,
        "deviation": deviation,
        "type": order_type.get(np.sign(lot)),
        "type_filling": filling_type.get(MetaTrader5.symbols_get(ticker)[0].filling_mode),
        }
    request_result = MetaTrader5.order_check(request)
    return request_result

def is_ticker_active(ticker: str, threshold_sec=60):
    last_quote_time = pd.to_datetime(MetaTrader5.symbol_info_tick(ticker).time_msc, unit='ms')-timedelta(hours=timezone_adjustment)
    time_check = datetime.now()-last_quote_time
    return time_check<timedelta(seconds=threshold_sec)

def open_position_at_market_price(ticker: str, lot: float, deviation=100):
    price = get_latest_tick_for_ticker(ticker)[lot>0]
    request = {
        "action": MetaTrader5.TRADE_ACTION_DEAL,
        "symbol": ticker,
        "volume": np.round(float(lot),2),
        "price": price,
        "deviation": deviation,
        "type": order_type.get(np.sign(lot)),
        "type_filling": filling_type.get(MetaTrader5.symbols_get(ticker)[0].filling_mode),
        }
    request_result_code = MetaTrader5.order_check(request).retcode
    print(MetaTrader5.order_check(request))
    if request_result_code != 0: # If the request is granted, the retcode won't be zero
        return False
    
    execution_result = MetaTrader5.order_send(request)
           
    return execution_result

def get_volume_min_for_ticker(ticker: str):
    
    return MetaTrader5.symbols_get(ticker)[0].volume_min
    
def get_ticker_list_from_path(path: str, keyword=None, currency='USD'):
    if keyword is not None:
        temp = MetaTrader5.symbols_get(keyword)
    else:
        temp = MetaTrader5.symbols_get()
    ticker_list = []
    for symbol in temp:
        if (path in symbol.path) and (symbol.currency_base==currency):
            ticker_list += [symbol.name]
    return ticker_list

#%% Whatever the trading platform, the structure of TickerLinearCombination must stay the same

class TickerLinearCombination():
    """
    Args:
        status: 1 - Long, 0 - None, -1 - Short, 'error' - open request incomplete, 'suspended' - Not available for trading
        lot: the volume of the combination that is being traded, can be positive or negative
        counter: no specific purpose, just an integar object that can be used as a counter for anything, e.g. stop loss cool down
        open_price: Record the dealing price of the current position. =None if not currently holding a position.
        contract_size: how many unit is in 1 contract
        additional_info: Store informaiton such as frequency, timeframe, cool down period, etc.
    """
    def __init__(self, 
                 ticker_list, weights, name=None,
                 max_volume=100, bid=None, ask=None, trade_tick_value=None, price_history=None, ticker_price_history=None, margin_requirement=None, additional_info1=None, additional_info2=None, additional_info3=None):
        
        # Attributes that need to be stored if program is closed
        self.ticker_list = ticker_list
        self.weights = weights
        self.name = name
        self.status = 0
        self.lot = 0 # The lot of the onging trade, + if long, - if short
        self.open_price, self.close_price = None, None
        self.open_time, self.close_time, self.trade_id = None, None, None
        self.trade_tickets = [None for i in range(len(ticker_list))]
        self.trade_volumes = [0 for i in range(len(ticker_list))]
        self.margin_occupied = [0 for i in range(len(ticker_list))]
        
        
        # Attributes that are obtained with functions
        self.n_tickers = len(ticker_list)
        self.contract_size = [MetaTrader5.symbol_info(ticker_list[i])._asdict().get('trade_contract_size') for i in range(len(ticker_list))]
        self.pnl = 0
        # MT5 has three filling types: Fill or Kill, Immediate or Cancel, and Return
        self.filling_type = [filling_type.get(MetaTrader5.symbols_get(ticker)[0].filling_mode) for ticker in ticker_list]
        self.max_volume = min([math.floor(MetaTrader5.symbols_get(ticker_list[i])[0].volume_max/abs(self.weights[i])) for i in range(self.n_tickers)])
        
        self.chart = None
        self.bid, self.ask, self.margin_requirement = bid, ask, margin_requirement
        self.price_history = price_history

        
        # Metrics
        self.adf_pvalue = np.nan
        self.parkinson_ratio = np.nan
        self.parkinson_score = np.nan
        self.cool_down = 0
        
    def is_active(self, threshold_sec=60):
        for ticker in self.ticker_list:
            if is_ticker_active(ticker, threshold_sec) == False:
                return False
        return True
    
    def get_pnl(self):
        pnl=0
        for ticket in self.trade_tickets:
            if ticket is not None:
                pnl += MetaTrader5.positions_get(ticket=ticket)[0].profit
        self.pnl = pnl
        
        return pnl
    
    def get_margin_requirement(self, lot=1.0):
        account_margin = MetaTrader5.account_info().margin
        margins = [0 for i in range(self.n_tickers)]
        for i in range(len(margins)):
            request_result = get_request_result_for_trade(self.ticker_list[i],abs(self.weights[i]*lot))
            margins[i] = abs(request_result.margin - account_margin)
        
        self.margin_requirement = sum(margins)
        return sum(margins)
    
    def get_metrics(self, lookbackperiod=500, interval=20):
        if self.price_history is None:
            raise TypeError("Must obtain price data from MetaTrader5 before calling this function")
        self.adf_pvalue = ts.adfuller(self.price_history.dropna())[1]
        self.parkinson_ratio = mrt.get_rolling_parkinson_ratio_for_series(self.price_history.dropna(), lookbackperiod=lookbackperiod)
        self.parkinson_score = stats.percentileofscore(self.parkinson_ratio, 1,'rank')
        
        return self.adf_pvalue, self.parkinson_score, self.parkinson_ratio
    
    def get_price_history(self, time = datetime.now(), n_obs=500, freq='H1', timezone=timezone_adjustment):
        ticker_price_history = get_price_history_for_ticker_list(ticker_list=self.ticker_list, end_time=time+timedelta(hours=timezone), n_obs=n_obs*4, freq=freq).dropna().iloc[-n_obs:]
        self.ticker_price_history = ticker_price_history
        self.price_history = ticker_price_history.dot(self.weights)

        return self.price_history
    
    def get_latest_tick(self):
        bid_index = {-1: 1, 1: 0}
        ask_index = {-1: 0, 1: 1}
        # Bid price
        bid_price, ask_price = 0, 0
        for i in range(self.n_tickers):
            bid_price += self.weights[i] * get_latest_tick_for_ticker(self.ticker_list[i])[bid_index.get(np.sign(self.weights[i]))] * self.contract_size[i]
            ask_price += self.weights[i] * get_latest_tick_for_ticker(self.ticker_list[i])[ask_index.get(np.sign(self.weights[i]))] * self.contract_size[i]
            
        self.bid, self.ask = bid_price, ask_price
        return bid_price, ask_price
    
    
    def open_position(self, lot: float, deviation=100):
        global error_log
        if self.status == 'suspended':
            print(self.name, 'trading suspended')
            return False
        trade_price = [None for i in range(self.n_tickers)]
        execution_result = [None for i in range(self.n_tickers)]
        requests = [None for i in range(self.n_tickers)]
        margin_occupied = [0 for i in range(self.n_tickers)]
        trade_result = True
        
        trade_tickets = [None for i in range(self.n_tickers)] # Initiate trade tickets for each ticker in the combination
        trade_volumes = [0 for i in range(self.n_tickers)]

        account_margin = MetaTrader5.account_info().margin
        # Construct the requests and check if all the requests can be executed
        for i in range(self.n_tickers):
            price = get_latest_tick_for_ticker(self.ticker_list[i])[np.sign(lot*self.weights[i])>0]
            requests[i] = {
                "action": MetaTrader5.TRADE_ACTION_DEAL,
                "symbol": self.ticker_list[i],
                "volume": np.round(float(abs(lot*self.weights[i])),2),
                "price": price,
                "deviation": deviation,
                "type": order_type.get(np.sign(lot*self.weights[i])),
                "type_filling":self.filling_type[i],
                }
            request_result = MetaTrader5.order_check(requests[i])
            if request_result is None:
                return False
            margin_occupied[i] = abs(request_result.margin - account_margin)
            if request_result.retcode != 0: # If the request is rejected, the retcode won't be zero
                print(self.name, 'open request failed due to', self.ticker_list[i],' error code: ', MetaTrader5.order_check(requests[i]).retcode, file=error_log)
                return False
        
        # At this point, the trade requests are all good, now we check if they can be successfully executed
        self.lot = lot # Whether succeed or not, a trade will happen, thus we need the lot and trade id
        self.trade_id = random.randint(100000,999999) # Assign a trade id to the combination
        
        # Execute the orders
        for i in range(self.n_tickers):
            execution_result[i] = MetaTrader5.order_send(requests[i])
            execution_result_code = execution_result[i].retcode
            
            if execution_result_code == MetaTrader5.TRADE_RETCODE_DONE:
                trade_tickets[i] = execution_result[i].order
                trade_price[i] = execution_result[i].price*self.contract_size[i]
                trade_volumes[i] = execution_result[i].volume
                self.margin_occupied[i] = margin_occupied[i]

        self.trade_tickets = trade_tickets
        self.trade_volumes = trade_volumes
        
        # Check if all the orders have been successfully executed. If not, at least one of the trade tickets will be None.
        if None in trade_tickets:
            self.status = 'error'
            return False
        
        # If the execution is successful, obtain the relevant information
        self.open_price = np.array(trade_price).dot(self.weights)
        self.open_time = datetime.now()
        self.status = np.sign(lot)
        
        # Trade log, only record if the trade is successfully executed
        global trade_log
        row = len(trade_log.index)
        trade_log.loc[row] = [self.trade_id, self.name, self.open_time, None, self.open_price, None, self.lot, None, None]

        return True
    
    
    def close_position(self, time_out_sec=30, deviation=100, comment=None):
        global error_log
        if self.status==0 or self.status=='terminated' or self.status=='suspended': # Already closed
            return True
        real_trade_close = all(v is not None for v in self.trade_tickets)
        trade_price = [0 for i in range(self.n_tickers)]
        close_execution_result = [None for i in range(self.n_tickers)]
        requests = [None for i in range(self.n_tickers)]
        check_requests = True
        trade_result = False
            
        if self.status != 'error':
            pnl =self.get_pnl() / sum(self.margin_occupied)
    
        # Try closing the position in each ticker repeatedly until all of them are successfully closed
        start = time.time()
        while not trade_result:
            for i in range(self.n_tickers):
                if self.trade_tickets[i] is not None:
                    price = get_latest_tick_for_ticker(self.ticker_list[i])[np.sign(self.lot*self.weights[i])<0]
                    request = {
                        "action": MetaTrader5.TRADE_ACTION_DEAL,
                        "symbol": self.ticker_list[i],
                        "volume": float(self.trade_volumes[i]),
                        "price": price,
                        "deviation": deviation,
                        "position": self.trade_tickets[i],
                        "type": order_type.get(np.sign(-self.lot*self.weights[i])),
                        "type_filling": self.filling_type[i],
                        }
                    close_execution_result[i] = MetaTrader5.order_send(request)
                    if close_execution_result[i] is None:
                        continue
                    if close_execution_result[i].retcode == MetaTrader5.TRADE_RETCODE_DONE:
                        self.trade_tickets[i]=None # Successful close will set the mt5 trade ticket to None
                        trade_price[i] = close_execution_result[i].price*self.contract_size[i]
            
            # If all positions are closed, then all slots of the trade tickets should be 0
            trade_result = all(v is None for v in self.trade_tickets)
            
            # Time out if the position cannot be closed
            end = time.time()
            if end - start >= time_out_sec:
                print(datetime.strftime(datetime.now(), '%d/%m/%Y, %H:%M:%S'), self.name, 'close positoin time out', file=error_log)
                self.status='error'
                return False
                
        self.close_price = np.array(trade_price).dot(self.weights)
        
        # Trade log
        if real_trade_close:
            global trade_log
            self.close_time=datetime.now()
            content = [self.trade_id, self.name, self.open_time, self.close_time, self.open_price, self.close_price, self.lot, pnl, comment]
            if trade_log[trade_log['Trade ID']==self.trade_id] is not None:
                trade_log[trade_log['Trade ID']==self.trade_id]=content
            else:
                trade_log.loc[len(trade_log.index)]=content
        self.lot=0
        self.status=0
        return True


#%% Close all combinations

def close_all_combinations(combination_dict):
    n_succeed = 0
    n_fail = 0
    for keys, item in combination_dict.items():
        if item.close_position(comment='forced close'):
            n_succeed +=1
        else:
            n_fail+=1
    print('Succeeded:',n_succeed, 'Failed:',n_fail)
    return n_fail==0
    