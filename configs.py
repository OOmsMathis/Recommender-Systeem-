# local imports
from models import *


class EvalConfig:
    
    models = [
        ("baseline_1", ModelBaseline1, {}),
        ("baseline_2", ModelBaseline2, {}),
        ("baseline_3", ModelBaseline3, {}),
        ("svd_model", SVD, {"random_state": 1}),  
    ]
    split_metrics = ["mae",'rmse']
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 10  # -- configure the numer of recommendations (> 1) --
