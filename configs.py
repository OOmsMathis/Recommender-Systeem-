# local imports
from models import ContentBased
from models import UserBased
from models import *

class EvalConfig:  
    models = [
      # UserBased models with min_k from 2 to 10
      ("user-based-hm_2", UserBased, {
        "k": 4,
        "min_k": 2,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_3", UserBased, {
        "k": 4,
        "min_k": 3,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_4", UserBased, {
        "k": 4,
        "min_k": 4,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_5", UserBased, {
        "k": 4,
        "min_k": 5,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_6", UserBased, {
        "k": 4,
        "min_k": 4,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_7", UserBased, {
        "k": 4,
        "min_k": 7,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_8", UserBased, {
        "k": 4,
        "min_k": 8,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_9", UserBased, {
        "k": 4,
        "min_k": 9,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
      ("user-based-hm_10", UserBased, {
        "k": 4,
        "min_k": 10,
        "sim_options": {
          'name': 'pearson_baseline',
          'user_based': True,
        }
      }),
    ]

    
    split_metrics = ["mae","rmse", "accuracy"]
    loo_metrics = ["hit_rate","precision"]
    full_metrics = ["diversity", "novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
