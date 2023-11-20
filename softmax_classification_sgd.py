#!/usr/bin/env python3
# authors: ec66f26d-2216-11ec-986f-f39926f24a9c
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

def softmax(z):
    return np.exp(z-np.max(z))/(np.exp(z-np.max(z)).sum())

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)
    
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        counter = 0 
        batch_grad = np.zeros_like(weights)
        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.
        for index in permutation:
            x = train_data[index]
            y = train_target[index]

            pred = softmax(x.T @ weights)
            one_hot_y = np.zeros_like(pred)
            one_hot_y[y] = 1
            x = x.reshape(65,1)
            y = (pred-one_hot_y).reshape(1, 10)
            item_grad = x @ y
            batch_grad += item_grad            
            counter += 1

            if counter == args.batch_size:
                batch_grad /= args.batch_size
                weights += -args.learning_rate * batch_grad
                batch_grad = np.zeros_like(weights)
                counter = 0 
        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or cross-entropy loss, or KL loss) per example.
        y_pred = np.apply_along_axis(softmax, 1, train_data @ weights)
        y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
        train_accuracy = sklearn.metrics.accuracy_score(train_target, y_pred)

        y_pred = np.apply_along_axis(softmax, 1, test_data @ weights)
        y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, y_pred)

        train_loss = 0
        for x, y in zip(train_data, train_target):
            pred = softmax(x @ weights)
            item_loss = -np.log(pred[y])
            train_loss += item_loss
        train_loss = train_loss / len(train_data)

        test_loss = 0
        for x, y in zip(test_data, test_target):
            pred = softmax(x @ weights)
            item_loss = -np.log(pred[y])
            test_loss += item_loss
        test_loss = test_loss / len(test_data)

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
