#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_grads(data, targets, preds):
    return np.dot(data.T, preds - targets) / len(data)

def compute_f1_scores(targets, predictions):
    TP = np.sum((predictions == 1) & (targets == 1), axis=0)
    FP = np.sum((predictions == 1) & (targets == 0), axis=0)
    FN = np.sum((predictions == 0) & (targets == 1), axis=0)

    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)

    f1_scores = 2 * np.divide(precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)

    TP_micro = np.sum(TP)
    FP_micro = np.sum(FP)
    FN_micro = np.sum(FN)
    
    precision_micro = TP_micro / (TP_micro + FP_micro) if TP_micro + FP_micro > 0 else 0
    recall_micro = TP_micro / (TP_micro + FN_micro) if TP_micro + FN_micro > 0 else 0
    
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if precision_micro + recall_micro > 0 else 0
    macro_f1 = np.mean(f1_scores)

    return micro_f1, macro_f1

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    
    target = np.zeros([len(target_list), args.classes])
    for index, item in enumerate(target_list):
        for element in item:
            target[index, element] = 1
            
    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        ...
        for i in range(0, train_data.shape[0], args.batch_size):
            batch_indices = permutation[i:i + args.batch_size]
            batch_data = train_data[batch_indices]
            batch_targets = train_target[batch_indices]

            preds = sigmoid(np.dot(batch_data, weights))
            grads = compute_grads(batch_data, batch_targets, preds)
            weights -= args.learning_rate * grads

        train_preds = sigmoid(np.dot(train_data, weights))
        train_preds = (train_preds >= 0.5).astype(int)
        test_preds = sigmoid(np.dot(test_data, weights))
        test_preds = (test_preds >= 0.5).astype(int)

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.
        train_f1_micro, train_f1_macro = compute_f1_scores(train_target, train_preds)
        test_f1_micro, test_f1_macro = compute_f1_scores(test_target, test_preds)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")