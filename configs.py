# local imports
from models import ContentBased



class EvalConfig:
    
    models = [

        ("content_ridge", ContentBased, {
            "features_methods": [ "average_ratings","genome_tags"],  
            "regressor_method": "ridge"
        }),
        ("content_ridge1", ContentBased, {
            "features_methods": [ "genome_tags"],  
            "regressor_method": "ridge"
        }),

        ("content_ridge2", ContentBased, {
            "features_methods": [ "average_ratings"],  
            "regressor_method": "ridge"
        }),
        ("content_ridge3", ContentBased, {
            "features_methods": [ "average_ratings","title_tfidf","tmdb_cast"],  
            "regressor_method": "ridge"
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

("content_knn", ContentBased, {
            "features_methods": ["title_length", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","genome_tags"],  
            "regressor_method": "knn"
        }),
        ("content_ridge4", ContentBased, {
            "features_methods": ["tmdb_vote_average", "Year_of_release", "average_ratings",'count_ratings',"Tags","Genre_binary","Genre_tfidf","genome_tags"],  
            "regressor_method": "gradient_boosting"
        }),
"""