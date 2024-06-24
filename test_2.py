
# %%

from src.datasource.yahoodata import YahooDataSource
import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# %%
tickers = ['MSFT','MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO']

column_name = 'Close'
interval = '1mo'

# %%

start_date = datetime.datetime(2010,1,1)
end_date = datetime.datetime(2020,1,1)

# %%
 
main_data = YahooDataSource(start_date,end_date,tickers,columns=[column_name],interval=interval).get_data_by_column_tickers(columns=[column_name],tickers=tickers)
main_data

# %%

# Calculate the returns
returns = main_data.pct_change().dropna()
returns = pd.DataFrame(returns)
returns


# %%

# def calculate_mean(returns) -> pd.Series:
#     mean_returns = returns.mean()
#     return mean_returns

# def calculate_covariance_matrix(returns) -> pd.DataFrame:
#     covariance_matrix = returns.cov()
#     return covariance_matrix


# mean_returns = calculate_mean(returns)
# covariance_matrix = calculate_covariance_matrix(returns)

# print("Mean Returns:")
# print(mean_returns)

# print("\nCovariance Matrix:")
# print(covariance_matrix)

# # %%
# def portfolio_variance(weights):
#      return np.dot(weights.T, np.dot(covariance_matrix, weights))

# def return_min_variance_portfolio(mean_returns: pd.Series, covariance_matrix: pd.DataFrame, constraint=None, allow_shorting=False):
    
#     num_assets = len(mean_returns)
        
#     initial_weights = np.ones(num_assets) / num_assets
    
#     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
#     if allow_shorting:
#         bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
#     else:
#         bounds = tuple((0, float('inf')) for _ in range(num_assets))  
    
#     if constraint:
#         constraints.append(constraint)
      
    
#     result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
#     if result.success:
#         return result.x
#     else:
#         raise ValueError("Optimization failed")


# min_variance_weights = return_min_variance_portfolio(mean_returns, covariance_matrix, allow_shorting=False)

# print("Minimum Variance Portfolio Weights:")
# print(min_variance_weights)

# # %%
# def get_max_return(mean_returns: pd.Series):
#     return mean_returns.max()

# def minimize_func(weights, cov_matrix):
#     return np.matmul(np.matmul(np.transpose(weights), cov_matrix), weights)

# def get_optimal_weights(mean_returns: pd.Series, cov_matrix: pd.DataFrame, target_return=None, constraint=None, allow_shorting=False):
    
#     num_assets = len(mean_returns)
    
#     # Constraints: weights sum to 1
#     constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
#     # If a target return is specified, add it as a constraint
#     if target_return is not None:
#         constraints.append({'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return})
    
#     # If an additional constraint is provided, add it
#     if constraint:
#         constraints.append(constraint)
    
#     # Set bounds based on whether short selling is allowed
#     if allow_shorting:
#         bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
#     else:
#         bounds = tuple((0, float('inf')) for _ in range(num_assets))  
    
#     # Initial equal distribution
#     initial_weights = np.ones(num_assets) / num_assets
    
#     # Optimization
#     result = minimize(minimize_func, initial_weights, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    
#     # Check if the optimization was successful
#     if result.success:
#         return result.x
#     else:
#         raise ValueError("Optimization failed")

# # max achievable return
# max_target_return = get_max_return(mean_returns)
# print("Maximum Achievable Target Return:")
# print(max_target_return)

# # W/o target return and w/o short selling
# optimal_weights_no_target_no_short = get_optimal_weights(mean_returns, covariance_matrix, allow_shorting=False)
# print("Optimal Portfolio Weights (No Target Return, No Short Selling):")
# print(optimal_weights_no_target_no_short)

# # With target return and w/o short selling
# target_return = 0.02  # Example target return
# optimal_weights_target_no_short = get_optimal_weights(mean_returns, covariance_matrix, target_return=target_return, allow_shorting=False)
# print("\nOptimal Portfolio Weights (Target Return, No Short Selling):")
# print(optimal_weights_target_no_short)

# # With target return and with short selling
# optimal_weights_target_short = get_optimal_weights(mean_returns, covariance_matrix, target_return=target_return, allow_shorting=True)
# print("\nOptimal Portfolio Weights (Target Return, Short Selling Allowed):")
# print(optimal_weights_target_short)
# %%

# multiple functions -> Individual duty

# -> calculate_mean(data)

# -> calculate_covariance(data)

# -> retrun_min_variance_portfolio(mean_return,covariance_matrix,constraint=None) -> return min_return

    # if "non-negative":
    
# -> get_optimal_weights(mean_return,covariance_matrix,traget_return=None,constraint=None)

    # if "non-negative":
# %%


# %%
class MeanVariance:
    
    def __init__(self, returns):
        self.returns = returns
        self.mean_returns = self.calculate_mean()
        self.cov_matrix = self.calculate_covariance_matrix()

    def calculate_mean(self) -> pd.Series:
        mean_returns = self.returns.mean()
        return mean_returns

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        covariance_matrix = self.returns.cov()
        return covariance_matrix

    def portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def return_min_variance_portfolio(self, constraint=None, allow_shorting=False):
        num_assets = len(self.mean_returns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  

        if constraint:
            constraints.append(constraint)

        result = minimize(self.portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)

    def get_max_return(self) -> float:
        return self.mean_returns.max()

    def minimize_func(self, weights):
        return np.matmul(np.matmul(np.transpose(weights), self.cov_matrix), weights)
    
    def get_optimal_weights(self, target_return=None, constraint=None, allow_shorting=False):
        num_assets = len(self.mean_returns)

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

        # If a target return is specified, add it as a constraint
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda weights: np.dot(weights, self.mean_returns) - target_return})

        # If an additional constraint is provided, add it
        if constraint:
            constraints.append(constraint)

        # Set bounds based on whether short selling is allowed
        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  

        # Initial equal distribution
        initial_weights = np.ones(num_assets) / num_assets

        # Optimization
        result = minimize(self.minimize_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Check if the optimization was successful
        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)
        
# %%

opti = MeanVariance(returns)

# Calculate the maximum achievable return
max_target_return = opti.get_max_return()
print("Maximum Achievable Target Return:")
print(max_target_return)

#Calculate Min Var portfolio weights
min_var_weights = opti.return_min_variance_portfolio()
print("\nMinimum-Variance Portfolio default")
print(min_var_weights)

# Without target return and without short selling
optimal_weights_no_target_no_short = opti.get_optimal_weights(allow_shorting=False)
print("\nOptimal Portfolio Weights (No Target Return, No Short Selling):")
print(optimal_weights_no_target_no_short)

# With a valid target return 
target_return = 0.019  
optimal_weights_target_no_short = opti.get_optimal_weights(target_return=target_return, allow_shorting=False)
print("\nOptimal Portfolio Weights (Target Return, No Short Selling):")
print(optimal_weights_target_no_short)

# With target return and with short selling
target_return = 0.030
optimal_weights_target_short = opti.get_optimal_weights(target_return=target_return, allow_shorting=True)
print("\nOptimal Portfolio Weights (Target Return, Short Selling Allowed):")
print(optimal_weights_target_short)

    
# %%
#dict(zip(test_keys, test_values))
for i in range(0,120,5):

    sub_data = main_data.iloc[i:i+5]

    mean = opti.calculate_mean(sub_data)

    covariance = opti.calculate_covariance_matrix(sub_data)

    optimal_weights = opti.get_optimal_weights(mean,covariance,allow_shorting=True)