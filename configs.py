# local imports
from models import *


class EvalConfig:
    
    models = [
        ("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
    ]
    split_metrics = ["mae"]
    loo_metrics = []
    full_metrics = []

    # Split parameters
    test_size = None  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = None  # -- configure the numer of recommendations (> 1) --
