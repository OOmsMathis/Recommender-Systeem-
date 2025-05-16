# local imports
from models import ContentBased



class EvalConfig:
    
    models = [
        
        ("content_linear", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "linear"
        }),
        ("content_ridge", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "ridge"
        }),
        ("content_knn", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "knn"
        }),
        ("content_lasso", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "lasso"
        }),
        ("content_elastic_net", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "elastic_net"
        }),
        ("content_decision_tree", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "decision_tree"
        }),
        
        
]
        
    
    split_metrics = ["mae","rmse"]
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]
    
    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --


"""
 ("content_2combii", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "ridge"
"""