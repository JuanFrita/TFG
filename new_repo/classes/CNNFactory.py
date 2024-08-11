import P2pPipeline
import BayesianPipeline

class CNNFactory():
    
    def get(model):
        if model == "bayesian":
            return BayesianPipeline()
        elif model == "p2p":
            return P2pPipeline()
        else:
            return None