
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import MeanShift, estimate_bandwidth

warnings.filterwarnings('ignore') 

model_fname = "model.save"

MODEL_NAME = "MeanShift"


class ClusteringModel:
    
    def __init__(self, bandwidth, verbose=False) -> None:
        self.bandwidth = bandwidth
        self.verbose = verbose
        
        self.model = self.build_model()
        
        
    def build_model(self):
        model = MeanShift(bin_seeding=True, bandwidth = self.bandwidth)
        return model
    

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.model.predict()
        # same for fit(), transform(), fit_transform(), etc.
        attr = getattr(self.model, name)

        if not callable(attr): return attr

        def wrapper(*args, **kwargs):
            return getattr(self.model, attr.__name__)(*args, **kwargs)

        return wrapper    
    
    
    def evaluate(self, x_test): 
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))


    @classmethod
    def load(cls, model_path):         
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = ClusteringModel.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def get_data_based_model_params(data): 
    bandwidth = estimate_bandwidth(data, quantile=0.25)
    return { "bandwidth": bandwidth }
