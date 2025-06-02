# local imports
from models import ContentBased
from models import UserBased
from models import *

class EvalConfig:  
  models = [
    ("average_ratings_count_ratings_ridge", ContentBased, {
      "features_methods": ["average_ratings", "count_ratings"],
      "regressor_method": "ridge"
      , "ridge_alpha": 1.0
    })
  ]

  split_metrics = ["mae", "rmse", "accuracy"]
  loo_metrics = ["hit_rate", "precision"]
  full_metrics = ["diversity", "novelty"]

  # Split parameters
  test_size = 0.25  # -- configure the test_size (from 0 to 1) --

  # Loo parameters
  top_n_value = 40  # -- configure the numer of recommendations (> 1) --
