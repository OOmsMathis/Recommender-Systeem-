# local imports
from models import ContentBased
from models import UserBased
from models import *

class EvalConfig:  
    models = [
        # Ridge models for all features except genome_tags and tfidf_relevance
        ("model_title_length_ridge", ContentBased, {"features_methods": ["title_length"], "regressor_method": "ridge"}),
        ("model_Year_of_release_ridge", ContentBased, {"features_methods": ["Year_of_release"], "regressor_method": "ridge"}),
        ("model_average_ratings_ridge", ContentBased, {"features_methods": ["average_ratings"], "regressor_method": "ridge"}),
        ("model_count_ratings_ridge", ContentBased, {"features_methods": ["count_ratings"], "regressor_method": "ridge"}),
        ("model_Genre_binary_ridge", ContentBased, {"features_methods": ["Genre_binary"], "regressor_method": "ridge"}),
        ("model_Genre_tfidf_ridge", ContentBased, {"features_methods": ["Genre_tfidf"], "regressor_method": "ridge"}),
        ("model_Tags_ridge", ContentBased, {"features_methods": ["Tags"], "regressor_method": "ridge"}),
        ("model_tmdb_vote_average_ridge", ContentBased, {"features_methods": ["tmdb_vote_average"], "regressor_method": "ridge"}),
        ("model_title_tfidf_ridge", ContentBased, {"features_methods": ["title_tfidf"], "regressor_method": "ridge"}),
        ("model_tmdb_popularity_ridge", ContentBased, {"features_methods": ["tmdb_popularity"], "regressor_method": "ridge"}),
        ("model_tmdb_budget_ridge", ContentBased, {"features_methods": ["tmdb_budget"], "regressor_method": "ridge"}),
        ("model_tmdb_revenue_ridge", ContentBased, {"features_methods": ["tmdb_revenue"], "regressor_method": "ridge"}),
        ("model_tmdb_profit_ridge", ContentBased, {"features_methods": ["tmdb_profit"], "regressor_method": "ridge"}),
        ("model_tmdb_runtime_ridge", ContentBased, {"features_methods": ["tmdb_runtime"], "regressor_method": "ridge"}),
        ("model_tmdb_vote_count_ridge", ContentBased, {"features_methods": ["tmdb_vote_count"], "regressor_method": "ridge"}),
        ("model_tmdb_cast_ridge", ContentBased, {"features_methods": ["tmdb_cast"], "regressor_method": "ridge"}),
        ("model_tmdb_director_ridge", ContentBased, {"features_methods": ["tmdb_director"], "regressor_method": "ridge"}),
        ("model_tmdb_original_language_ridge", ContentBased, {"features_methods": ["tmdb_original_language"], "regressor_method": "ridge"}),

        # Gradient Boosting models for all features except genome_tags and tfidf_relevance
        ("model_title_length_gradient_boosting", ContentBased, {"features_methods": ["title_length"], "regressor_method": "gradient_boosting"}),
        ("model_Year_of_release_gradient_boosting", ContentBased, {"features_methods": ["Year_of_release"], "regressor_method": "gradient_boosting"}),
        ("model_average_ratings_gradient_boosting", ContentBased, {"features_methods": ["average_ratings"], "regressor_method": "gradient_boosting"}),
        ("model_count_ratings_gradient_boosting", ContentBased, {"features_methods": ["count_ratings"], "regressor_method": "gradient_boosting"}),
        ("model_Genre_binary_gradient_boosting", ContentBased, {"features_methods": ["Genre_binary"], "regressor_method": "gradient_boosting"}),
        ("model_Genre_tfidf_gradient_boosting", ContentBased, {"features_methods": ["Genre_tfidf"], "regressor_method": "gradient_boosting"}),
        ("model_Tags_gradient_boosting", ContentBased, {"features_methods": ["Tags"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_vote_average_gradient_boosting", ContentBased, {"features_methods": ["tmdb_vote_average"], "regressor_method": "gradient_boosting"}),
        ("model_title_tfidf_gradient_boosting", ContentBased, {"features_methods": ["title_tfidf"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_popularity_gradient_boosting", ContentBased, {"features_methods": ["tmdb_popularity"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_budget_gradient_boosting", ContentBased, {"features_methods": ["tmdb_budget"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_revenue_gradient_boosting", ContentBased, {"features_methods": ["tmdb_revenue"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_profit_gradient_boosting", ContentBased, {"features_methods": ["tmdb_profit"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_runtime_gradient_boosting", ContentBased, {"features_methods": ["tmdb_runtime"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_vote_count_gradient_boosting", ContentBased, {"features_methods": ["tmdb_vote_count"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_cast_gradient_boosting", ContentBased, {"features_methods": ["tmdb_cast"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_director_gradient_boosting", ContentBased, {"features_methods": ["tmdb_director"], "regressor_method": "gradient_boosting"}),
        ("model_tmdb_original_language_gradient_boosting", ContentBased, {"features_methods": ["tmdb_original_language"], "regressor_method": "gradient_boosting"}),

]
    
    split_metrics = ["mae","rmse", "accuracy"]
    loo_metrics = ["hit_rate","precision"]
    full_metrics = ["diversity", "novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
