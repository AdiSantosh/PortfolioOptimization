"""
Contains the definition of abstract class ScenarioGen
"""
from abc import ABC, abstractmethod

class ScenarioGen(ABC):
    """
    Contains abstract methods for scenario generation
    """

    @abstractmethod
    def gen_scenarios(self,starting_index:int,ending_index:int,*args,**kwargs):
        """
        Generate Scenarios for given number of days
        """

    @abstractmethod
    def get_summary(self,data,starting_index:int=0,ending_index:int=-1,*args,**kwargs):
        """
        Compute summaries of given data
        """

    