import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime as dt
from pandas.tseries.offsets import BDay

def returns_cov(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    return cov_matrix

start_date = '2023-07-24'
end_date = '2024-07-20'

stocks = ['NVDA','BTCC-B.TO', 'ENB.TO', 'SIA.TO', 'XSP.TO']

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))*252)


def risk_contribution(weights, cov_matrix):
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    marginal_contribution = np.dot(cov_matrix, weights)
    risk_contribution = weights * marginal_contribution / portfolio_vol
    return risk_contribution

def objective_function(weights, cov_matrix):
    rc = risk_contribution(weights, cov_matrix)
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    rc_target_percent = np.ones(5)/5
    risk_target = np.multiply(portfolio_vol, rc_target_percent)
    obj_value = sum(np.square(rc-risk_target.T))
    print(f"Objective Value: {obj_value:.6f}, Weights: {weights}")
    return obj_value

def optimize_erc(cov_matrix):
    num_assets = len(cov_matrix)
    initial_weights = np.ones(num_assets) / num_assets  
    bounds = [(0.01, 0.99) for _ in range(num_assets)]  
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
    result = minimize(objective_function, initial_weights, args=(cov_matrix,), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    print("Optimization Result:", result)
    return result.x


start_date = '2023-07-24'
end_date = '2024-07-20'
stocks = ['NVDA','BTCC-B.TO', 'ENB.TO', 'SIA.TO', 'XSP.TO']

weights = np.ones(5)/5

cov_matrix = returns_cov(stocks, start_date, end_date)

print(optimize_erc(cov_matrix))
print(portfolio_volatility(weights, cov_matrix))



