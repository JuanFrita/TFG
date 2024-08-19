from classes.P2pPipeline import P2pPipeline
from classes.BayesianPipeline import BayesianPipeline
from classes.BaseCNNPipeline import BaseCNNPipeline

class CNNFactory():
    
    def get(model) -> BaseCNNPipeline:
        if model == "bayesian":
            return BayesianPipeline()
        elif model == "p2p":
            return P2pPipeline()
        else:
            return None