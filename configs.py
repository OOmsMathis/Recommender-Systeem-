# local imports
from models import ContentBased
from models import UserBased
from models import *

class EvalConfig:  
  models = [
    ("SVD_PLUS", ModelBaseline5, {
      "n_factors": 125,
      "n_epochs": 40,
      "lr_all": 0.005,
      "reg_all": 0.02 }),

  ]

  split_metrics = ["mae", "rmse", "accuracy"]
  loo_metrics = ["hit_rate", "precision"]
  full_metrics = ["diversity", "novelty"]

  # Split parameters
  test_size = 0.25  # -- configure the test_size (from 0 to 1) --

  # Loo parameters
  top_n_value = 40  # -- configure the numer of recommendations (> 1) --
