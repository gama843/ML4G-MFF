#!/usr/bin/env python3
# author: ec66f26d-2216-11ec-986f-f39926f24a9c Daniel Kuchta
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")

class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        scaler = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_cols = [i for i in range(15)]
        num_cols = [i for i in range(15,21)]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, cat_cols),
                ("num", scaler, num_cols),
            ]
        )

        pipe = Pipeline(
            steps=[('preprocess', preprocessor), 
                   ('nystroem', Nystroem(kernel='poly', degree=4)),
                   ('logreg', LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000),)
                   ])

        param_grid = {
                    'nystroem__n_components' : [400, 700, 800, 900],
                    'logreg__C': [150, 200, 250, 300],
                    }

        # Train a model on the given dataset and store it in `model`.
        model = GridSearchCV(pipe, param_grid=param_grid, cv=10, n_jobs=-1, verbose=4).fit(train.data, train.target)        

        print('Best params:', model.best_params_)
        print('Best score:', model.best_score_)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)