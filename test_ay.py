import datetime
import pandas as pd
# from backtest.backtest import BackTest
from src.algorithms.strategy import CvarMretOpt,MeanSemidevOpt,EqualyWeighted
from src.datasource.yahoodata import YahooDataSource
from pprint import pprint

tickers = ['MSFT','MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO']

column_name = 'Close'
interval = '1d'

start_date = datetime.datetime(2020,1,1)
end_date = datetime.datetime(2024,1,1)

main_data = YahooDataSource(tickers,start_date,end_date,columns=[column_name],interval=interval)

data = main_data.get_data()
# print(data)
print(main_data.get_data_by_frequency(start_date,end_date,'1MS'))
# from src.algorithms.strategy import MeanSemidevOpt,CvarMretOpt
# mean_semideviation = CvarMretOpt(1.2)
# weights = pd.DataFrame(mean_semideviation.run_strategy(main_data)).T
# weights.index = pd.to_datetime(weights.index)
# # weights.set_index(pd.to_datetime(weights.index),inplace=True)
# print(weights)

# get all keys and values
# keys = [key.strftime('%Y-%m-%d') for key in weights.keys()]
# values = weights.values()
# pprint(keys)