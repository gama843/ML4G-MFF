#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    
    priors = np.bincount(train_target) / len(train_target)
    preds = [] 
    log_probs = []

    if args.naive_bayes_type == 'gaussian':    
        means = np.zeros((args.classes, train_data.shape[1]))
        variances = np.zeros_like(means)
        
        for cls in range(args.classes):
            cls_items = train_data[train_target == cls]
            means[cls] = cls_items.mean(axis=0)
            variances[cls] = cls_items.var(axis=0) + args.alpha
     
        for item in test_data:
            log_prob = [] 
            for cls in range(args.classes):
                cls_log_prob = np.log(priors[cls])
                cls_log_prob += np.sum(scipy.stats.norm.logpdf(item, means[cls], np.sqrt(variances[cls])))
                log_prob.append(cls_log_prob)
            
            preds.append(np.argmax(log_prob))
            log_probs.append(log_prob)

    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    
    if args.naive_bayes_type == 'multinomial':
        feature_probs = np.zeros((args.classes, train_data.shape[1]))
        for cls in range(args.classes):
            cls_items = train_data[train_target == cls]
            feature_probs[cls] = (np.sum(cls_items, axis=0) + args.alpha) / (np.sum(cls_items) + train_data.shape[1] * args.alpha)

        for item in test_data:
            log_prob = []
            for cls in range(args.classes):
                cls_log_prob = np.log(priors[cls])
                cls_log_prob += np.sum(item * np.log(feature_probs[cls]))
                log_prob.append(cls_log_prob)

            preds.append(np.argmax(log_prob))
            log_probs.append(log_prob)

    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    
    if args.naive_bayes_type == 'bernoulli':
        binarized_train_data = np.where(train_data >= 8, 1, 0)
        binarized_test_data = np.where(test_data >= 8, 1, 0)

        feature_probs = np.zeros((args.classes, train_data.shape[1]))
        for cls in range(args.classes):
            cls_items = binarized_train_data[train_target == cls]
            feature_probs[cls] = (cls_items.sum(axis=0) + args.alpha) / (len(cls_items) + 2 * args.alpha)

        for item in binarized_test_data:
            log_prob = []
            for cls in range(args.classes):
                cls_log_prob = np.log(priors[cls])
                cls_log_prob += np.sum(item * np.log(feature_probs[cls]) + (1 - item) * np.log(1 - feature_probs[cls]))
                log_prob.append(cls_log_prob)

            preds.append(np.argmax(log_prob))
            log_probs.append(log_prob)
        
    # In all cases, the class prior is the distribution of the train data classes.

    # TODO: Predict the test data classes, and compute
    # - the test set accuracy, and
    # - the joint log-probability of the test set, i.e.,
    #     \sum_{(x_i, t_i) \in test set} \log P(x_i, t_i).
    
    test_accuracy = np.mean(preds == test_target)
    test_log_probability = sum(log_probs[i][test_target[i]] for i in range(len(test_data)))

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))
