# local imports
from models import ModelBaseline1, ModelBaseline2, ModelBaseline3, ModelBaseline4,ContentBased



class EvalConfig:
    
    models = [
        ("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
        ("baseline_2", ModelBaseline2, {}),
        ("baseline_3", ModelBaseline3, {}),
        ("baseline_4", ModelBaseline4, {"random_state": 1}),
         #Ajouts pour le Content-Based
        ("content_linear", ContentBased, {
            "features_method": "title_length",  
            "regressor_method": "linear"
        }),
          ("content_lasso", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "lasso"
         }),
          ("content_random_forest", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "random_forest"
        }),
        ("content_neural_network", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "neural_network"
         }),
        ("content_decision_tree", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "decision_tree"
        }),
        ("content_ridge", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "ridge"
        }),
        ("content_gradient_boosting", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "gradient_boosting"
        }),
        ("content_knn", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "knn"
        }),
        ("content_elastic_net", ContentBased, {
        "features_method": "title_length",
        "regressor_method": "elastic_net"
        }),
]
        
   
    
    split_metrics = ["mae","rmse"]
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
