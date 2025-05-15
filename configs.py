# local imports
from models import ContentBased



class EvalConfig:  
    models = [
        ("content_lasso", ContentBased, {
            "features_method": "Year_of_release",
            "regressor_method": "lasso"
        }),
        ("content_random_forest", ContentBased, {
            "features_method": "Year_of_release",
            "regressor_method": "random_forest"
        }),
        ("content_neural_network", ContentBased, {
            "features_method": "Year_of_release",
            "regressor_method": "neural_network"
        }),
        ("content_decision_tree", ContentBased, {
            "features_method": "Year_of_release",
            "regressor_method": "decision_tree"
        }),
        ("content_ridge", ContentBased, {
            "features_method": "Year_of_release",
            "regressor_method": "ridge"
        }),
        ("content_gradient_boosting", ContentBased, {
            "features_method": "Year_of_release",
            "regressor_method": "gradient_boosting"
        }),
        ("content_knn", ContentBased, {
            "features_method": "Year_of_release",
            "regressor_method": "knn"
        }),
        ("content_elastic_net", ContentBased, {
            "features_method": "Year_of_release",
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
