# local imports
from models import ContentBased
from models import UserBased
from models import *

class EvalConfig:  
    models = [
        # Ridge regression on each individual feature
        ("content_ridge_title_length", ContentBased, {"features_methods": ["title_length"], "regressor_method": "ridge"}),
        # Baseline SVD model
        ("baseline4_svd", UserBased, {"model_type": "svd", "n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}),
    ]

    split_metrics = ["mae","rmse"]
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 20  # -- configure the numer of recommendations (> 1) --
