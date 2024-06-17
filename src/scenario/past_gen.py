
import datetime
import pandas as pd
from ..datasource.yahoodata import YahooDataSource
from .sceanrio_gen import ScenarioGen


class PastGen(ScenarioGen):
    """
    Implements azure blob file storage
    """

    def __init__(self,start_date:datetime.datetime,end_date:datetime.datetime,tickers:list[str],column_name:str):

        self.data =  YahooDataSource(start_date,end_date,tickers,columns=[column_name]).get_data_by_column_tickers(columns=[column_name],tickers=tickers)
 
    def gen_scenarios(self,starting_index:int,ending_index:int):
        """
        Return the data with the given starting_index and ending_index
        """
        return self.data.iloc[starting_index:ending_index,:]
    
    def get_summary(self,data:pd.DataFrame):

        """
        Return the summary stastics of given data starting_index and ending_index
        """
        series_mean = data.mean(axis=0)
        series_var = data.var(axis=0)
        series_std = data.std(axis=0)
        df_summary = pd.DataFrame([series_mean, series_var,series_std],index = ["mean","variance","std"])

        return df_summary
    
    def get_covariance(self,data:pd.DataFrame):
        "Return Covariance matrix of the given data"
        return data.cov()
