# local imports
from models import ContentBased



class EvalConfig:  
    models = [
        ("content_ridge", ContentBased, {
            "features_method": ["genome_tags", "Genre_binary"],
            "regressor_method": "ridge"})
    ]

    split_metrics = ["mae","rmse"]
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
