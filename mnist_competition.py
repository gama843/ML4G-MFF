#!/usr/bin/env python3
# authors: ec66f26d-2216-11ec-986f-f39926f24a9c
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

from sklearn.neural_network import MLPClassifier
import shutil

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="models/mnist_competition", type=str, help="Model path")

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)

def soft_voting(classifiers, x):
    predictions = np.zeros((len(x), 10))
    for clf in classifiers:
        predictions += clf.predict_proba(x)
    return np.argmax(predictions, axis=1)

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Train a model on the given dataset and store it in `model`.
        X_train = train.data / 255.0
        y_train = train.target
        mlps = []
        load = True
        mlps_count = 64
        
        if load:
            shutil.unpack_archive('models.zip', 'models')
            model_paths = [path for path in os.listdir('models') if path not in ['.DS_Store', '__MACOSX']]
            for path in model_paths:
                with lzma.open(os.path.join('models', path), "rb") as model_file:
                    model = pickle.load(model_file)
                mlps.append(model)
        
        else:
            if not os.path.exists('models'):
                os.makedirs('models')
            for j in range(mlps_count):
                mlp = MLPClassifier(hidden_layer_sizes=(200,200,100), activation='relu', solver='adam', alpha=0.0001).fit(X_train, y_train)
                # If you trained one or more MLPs, you can use the following code
                # to compress it significantly (approximately 12 times). The snippet
                # assumes the trained `MLPClassifier` is in the `mlp` variable.
                mlp._optimizer = None
                for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
                for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
                # Serialize the model.
                with lzma.open(args.model_path + str(j) + '.model', "wb") as model_file:
                    pickle.dump(mlp, model_file)
                mlps.append(mlp)
            shutil.make_archive('models', 'zip', 'models')

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        mlps = []
        shutil.unpack_archive('models.zip', 'models')
        model_paths = [path for path in os.listdir('models') if path not in ['.DS_Store', '__MACOSX']]
        for path in model_paths:
            with lzma.open(os.path.join('models', path), "rb") as model_file:
                model = pickle.load(model_file)
            mlps.append(model)

        # Generate `predictions` with the test set predictions.
        predictions = soft_voting(mlps, test.data / 255.0)

        return predictions

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
