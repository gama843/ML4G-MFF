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
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rentals in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

def spline(period, n_splines=None, degree=3):
    n_knots = n_splines + 1
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

def evaluate(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_root_mean_squared_error"],
        return_estimator=True,
    )
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Root Mean Squared Error: {rmse.mean():.2f} +/- {rmse.std():.3f}"
    )
    idx = np.argmax(cv_results['test_neg_root_mean_squared_error'])
    return cv_results['estimator'][idx] 

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:

        np.random.seed(args.seed)
        train = Dataset()

        cat_cols = []
        num_cols = [] 
        per_cols = [] 

        for index in range(train.data.shape[1]):
            if index in [2,3,5]:
                per_cols.append(index)
            elif (train.data[:,index] % 1 == 0).all():
                cat_cols.append(index)
            else:
                num_cols.append(index)

        timeseries_cv_splits = TimeSeriesSplit(
        n_splits=2,
        gap=72,
        max_train_size=900,
        test_size=100,
        )

        scaler = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, cat_cols),
                ("num", scaler, num_cols),
                ("spline_h", spline(24, n_splines=13), [3]),
                ("spline_d", spline(7, n_splines=3), [5]),
                ("spline_m", spline(12, n_splines=5), [2]),
            ]
        )
        
        alphas = np.logspace(-6, 6, 25)

        pipe = make_pipeline(
            preprocessor, 
            Nystroem(kernel="poly", degree=3, n_components=450, random_state=42),
            GridSearchCV(TweedieRegressor(power=1, max_iter=20000), param_grid={"alpha": alphas})
        )

        model = evaluate(pipe, train.data, train.target, cv=timeseries_cv_splits)
        
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)