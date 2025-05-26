# local imports
from models import ContentBased
from models import UserBased
from models import *

class EvalConfig:  
    models = [
        (
            "User_based_cosine",
            UserBased,
            {
                "k": 5,
                "min_k": 3,
                "sim_options": {
                    "name": "cosine",
                    "user_based": True
                }
            }
        ),
        (
            "User_based_msd",
            UserBased,
            {
                "k": 5,
                "min_k": 3,
                "sim_options": {
                    "name": "msd",
                    "user_based": True
                }
            }
        ),
        (
            "User_based_pearson",
            UserBased,
            {
                "k": 5,
                "min_k": 3,
                "sim_options": {
                    "name": "pearson",
                    "user_based": True
                }
            }
        ),
        (
            "User_based_pearson_baseline",
            UserBased,
            {
                "k": 5,
                "min_k": 3,
                "sim_options": {
                    "name": "pearson_baseline",
                    "user_based": True
                }
            }
        ),
        (
            "User_based_jaccard",
            UserBased,
            {
                "k": 5,
                "min_k": 3,
                "sim_options": {
                    "name": "jaccard",
                    "user_based": True
                }
            }
        ),
    ]

    split_metrics = ["mae","rmse", "accuracy"]
    loo_metrics = ["hit_rate","precision"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 20  # -- configure the numer of recommendations (> 1) --
