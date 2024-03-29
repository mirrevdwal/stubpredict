import logging
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, GridSearchCV

RANDOM_SEED = 8392
SCORING_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"] # metrics to use
CV_SPLITS = 10 # number of cross-validation splits

def prepare_dataset(dataset, min_stubs=3):
    """
    Remove all stubs that are of an ammo type with fewer than `min_stubs` stubs.
    """

    return dataset[dataset.groupby("ammo_type").ammo_type.transform("count")>=min_stubs].copy()


def get_smote_data(X_train, y_train):
    """
    Use Synthetic Minority Oversampling Technique (SMOTE) to increase minority
    class size.
    """

    counter = Counter(y_train)
    print(f"Class sizes before SMOTE: {list(counter.items())}")

    smt = SMOTE(k_neighbors=1)
    X_train_SMOTE, y_train_SMOTE = smt.fit_resample(X_train, y_train)

    counter = Counter(y_train_SMOTE)
    print(f"Class sizes after SMOTE:  {list(counter.items())}")

    return X_train_SMOTE, y_train_SMOTE


def init_results_df():
    """
    Initialize an empty DataFrame to populate with the name of the classifier
    and the values returned from vross validation.
    """

    results_evaluation = pd.DataFrame({
        "classifier_name": [],
        "fit_time": [],
        "score_time": [],
    })

    for metric in SCORING_METRICS:
        results_evaluation["test_" + metric] = []
    
    return results_evaluation


def train_models(models, X_train, y_train):
    """
    Train the given models using the training data.
    """

    # Initialize results DataFrame
    training_results = init_results_df()

    for (name, classifier) in models.items():
        logging.info(f"Currently training classifier {name}")

        # Note: precision will throw a warning if there are no predicted samples for a certain class.
        # As we believe that a 0 precision value is best for these cases, we can simply ignore these warnings.
        warnings.filterwarnings("ignore")
        cv_scores = cross_validate(classifier, X_train, y_train, cv=CV_SPLITS, scoring=SCORING_METRICS)

        # Store results of current model in a dictionary,
        # using the average of the scores among folds
        current_results = {
            "classifier_name":[name],
        }

        # Populate the dictionary with the results of the cross-validation
        for metric_name, score_per_fold in cv_scores.items():
            current_results[metric_name] = [ cv_scores[metric_name].mean() ]

        # Generate the results to populate the pandas.DataFrame
        current_result = pd.DataFrame(current_results)

        # Append to the main dataframe with the results
        training_results = pd.concat([training_results, current_result], ignore_index=True)

    return training_results


def tune_models(X_train, y_train):
    """
    Tune the four models in order to improve performance.
    """

    gs_KNN, gs_tree, gs_forest, gs_GNB = create_gs_models()

    # Fit the training data of the dataset with all features
    gs_KNN.fit(X_train, y_train)
    gs_tree.fit(X_train, y_train)
    gs_forest.fit(X_train, y_train)
    gs_GNB.fit(X_train, y_train)

    return gs_KNN, gs_tree, gs_forest, gs_GNB


def create_gs_models():
    """
    Create GridSearchCV"s for the models used.
    """

    # K-nearest neighbors
    clf_KNN = KNeighborsClassifier()
    param_grid_KNN = [{
        "n_neighbors": (1, 3, 5, 10),
        "weights": ("uniform", "distance"),
        "metric": ("minkowski", "chebyshev"), 
    }]
    gs_KNN = GridSearchCV(clf_KNN, param_grid_KNN, cv=CV_SPLITS)

    # Decision Tree
    clf_tree = DecisionTreeClassifier()
    param_grid_tree = [{
        "max_leaf_nodes": [3, 6, 9],
        "max_depth":[3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50],
    }]
    gs_tree = GridSearchCV(clf_tree, param_grid_tree, cv=CV_SPLITS)
    
    # Random Forest
    clf_forest = RandomForestClassifier()
    param_grid_forest = [{
        "n_estimators": [10, 25, 50, 100, 150],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [3, 6, 9],
        "max_leaf_nodes": [3, 6, 9],
    }]
    gs_forest = GridSearchCV(clf_forest, param_grid_forest, cv=CV_SPLITS)

    # Naive Bayes
    clf_GNB = GaussianNB()
    param_grid_GNB = {"var_smoothing": np.logspace(0, -9, num=100)}
    gs_GNB = GridSearchCV(clf_GNB, param_grid_GNB, cv=CV_SPLITS)

    return gs_KNN, gs_tree, gs_forest, gs_GNB


def run(dataset):
    df = prepare_dataset(dataset)
    print(f"Infrequent ammo types removed, final dataset shape: {df.shape}")

    np.random.seed(RANDOM_SEED)

    # Split dataset into data and class labels (ammo type)
    X_features = df.drop(columns=["stub_id", "ammo_type"])
    y = df["ammo_type"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    print(f"Size of training set:        {X_train.shape}")
    print(f"Size of test set:            {X_test.shape}")
    print(f"Size of training target set: {len(y_train)}")
    print(f"Size of test target set:     {len(y_test)}")

    # Use SMOTE for larger training set
    X_train_SMOTE, y_train_SMOTE = get_smote_data(X_train, y_train)
    print(f"Size of training set (SMOTE):        {X_train_SMOTE.shape}")
    print(f"Size of training target set (SMOTE): {len(y_train_SMOTE)}")

    # Define models to use
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "DT": DecisionTreeClassifier(max_depth=9),
        "RF": RandomForestClassifier(n_estimators=10, max_depth=9),
        "NB": GaussianNB(),
    }

    # Train models and report results
    training_results = train_models(models, X_train, y_train)
    training_results_SMOTE = train_models(models, X_train_SMOTE, y_train_SMOTE)
    print("Results (without SMOTE):\n", training_results)
    print("Results (with SMOTE):\n", training_results_SMOTE)

    gs_KNN, gs_tree, gs_forest, gs_GNB = tune_models(X_train, y_train)
    gs_KNN_SMOTE, gs_tree_SMOTE, gs_forest_SMOTE, gs_GNB_SMOTE = tune_models(X_train_SMOTE, y_train_SMOTE)

    # Report performance metric for tuned models
    for (i, tuned_model) in enumerate([gs_KNN, gs_tree, gs_forest, gs_GNB, \
                                       gs_KNN_SMOTE, gs_tree_SMOTE, gs_forest_SMOTE, gs_GNB_SMOTE]):
        
        smote_suffix = " SMOTE" if i >= len(models) else ""
        model_name = list(models.keys())[i % len(models)] + smote_suffix

        y_predicted = tuned_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_predicted)
        recall = recall_score(y_test, y_predicted, average="macro")
        precision = precision_score(y_test, y_predicted, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_predicted, average="macro")

        print(f"Tuned model performance ({model_name}):")
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
