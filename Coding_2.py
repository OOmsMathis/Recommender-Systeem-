from loaders import load_ratings, load_items
from surprise.model_selection import train_test_split
#surprise
df_ratings = load_ratings(surprise_format=True)
df_items = load_items()


def generate_split_predictions(algo, ratings_dataset, test_size=0.25):
    # Split the dataset into training and testing sets
    trainset, testset = train_test_split(ratings_dataset, test_size=test_size)
    # Train the algorithm on the training set
    algo.fit(trainset)
    # Generate predictions on the test set
    predictions = algo.test(testset)
    return predictions

"""implement generate_loo_top_n 1
: implement a function that takes as input a surprise algorithm ( algo ), a surprise Dataset ( ratings_dataset ) and the EvalConfig ,
and that outputs the top-N recommendations made on the anti testset obtained by the
leave-one-out method. The function also needs to output the testset as it will be required during evaluation. Hints : use the LeaveOneOut class (with one split) from
the model_selection surprise package. Note that a trainset object from the surprise
package has a method .build_anti_testset() that allows to output the anti test-set
1
loo stands for "leave one out"
2
MLSMM2156 Recommender Systems 2021-2022
related to a training-test split. Use the top_n_value from EvalConfig to configure
a recommendation list of 40 movies."""
from surprise.model_selection import LeaveOneOut


def generate_loo_top_n(algo, ratings_dataset, eval_config):
    #leaveOneOut object with one split
    loo = LeaveOneOut(n_splits=1)
    # Split the dataset into training and testing sets
    trainset, testset = next(loo.split(ratings_dataset))
    # Train the algorithm on the training set
    algo.fit(trainset)
    # Generate the anti-testset
    anti_testset = trainset.build_anti_testset()
    # Generate predictions on the anti-testset
    predictions = algo.test(anti_testset)
    
    # Get top-N recommendations
    top_n = get_top_n(predictions, n=eval_config.top_n_value)
    
    return top_n, testset