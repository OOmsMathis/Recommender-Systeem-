# local imports
from models import ModelBaseline1, ModelBaseline2, ModelBaseline3, ModelBaseline4,ContentBased



class EvalConfig:
    
    models = [
        ("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
        ("baseline_2", ModelBaseline2, {}),
        ("baseline_3", ModelBaseline3, {}),
        ("baseline_4", ModelBaseline4, {"random_state": 1}),
         #Ajouts pour le Content-Based
        ("content_random_score", ContentBased, {
            "features_method": None,
            "regressor_method": "random_score"
        }),
        ("content_random_sample", ContentBased, {
            "features_method": None,
            "regressor_method": "random_sample"
        }),
        ("content_linear", ContentBased, {
            "features_method": "title_length",  
            "regressor_method": "linear"
        }),
    ]
    split_metrics = ["mae","rmse"]
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
