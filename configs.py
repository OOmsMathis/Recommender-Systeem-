# local imports
from models import ContentBased



class EvalConfig:
    
    models = [
        
        ("content_1combi", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags","tfidf_relevance"],  
            "regressor_method": "linear"
        }),
        ("content_2combi", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags","tfidf_relevance"],  
            "regressor_method": "ridge"
        }),
        ("content_3", ContentBased, {
            "features_methods": ["Year_of_release"],  
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
