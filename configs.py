# local imports
from models import ContentBased



class EvalConfig:
    
    models = [
        
        ("content_linear", ContentBased, {
            "features_methods": ["title_length", "Year_of_release"],  
            "regressor_method": "linear"
        }),
        ("content_linear", ContentBased, {
            "features_methods": ["title_length","Year_of_release"],  
            "regressor_method": "lasso"
        }),
        ("content_linear", ContentBased, {
            "features_methods": ["title_length","Year_of_release"],  
            "regressor_method": "ridge"
        }),
        
]
        
    
    split_metrics = ["mae","rmse"]
    
    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
