#!/usr/bin/env python3
# author: ec66f26d-2216-11ec-986f-f39926f24a9c Daniel Kuchta
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Append a new feature to all input data, with value "1"
    features = np.concatenate([dataset.data, np.ones((442,1))], axis=1)
    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features, dataset.target, test_size=args.test_size, random_state=args.seed)
    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    weights = np.linalg.inv(np.transpose(X_train) @ X_train) @ np.transpose(X_train) @ Y_train
    # TODO: Predict target values on the test set.
    predictions = X_test @ weights
    # TODO: Manually compute root mean square error on the test set predictions.
    rmse = np.sqrt(np.sum(np.square(np.subtract(X_test @ weights, Y_test))) / len(X_test))

    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
