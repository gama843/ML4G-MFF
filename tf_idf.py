#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request
import re

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from collections import Counter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=45, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=1000, type=int, help="Test set size")
parser.add_argument("--train_size", default=500, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_y, test_y = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.
    # print(len(train_data))
    
    unique_words = set()
    unique_words_multi = dict()
    index = 0


    # find relevant unique words
    for doc in train_data:
        words = re.findall(r'\b\w+\b', doc)
        for word in words:
            if word in unique_words:
                if word not in unique_words_multi:
                    unique_words_multi[word] = index
                    index += 1
            else:
                unique_words.add(word)
    
    # count their occurrences across and per docs in train
    word_occurrences_across_docs = dict()
    word_occurrences_per_doc_train = []

    for doc_index, doc in enumerate(train_data):
        words = re.findall(r'\b\w+\b', doc)
        processed_words = set()
        word_occurrences_per_doc_train.append(dict())

        for word in words:
            if word not in processed_words:
                processed_words.add(word)
                if word in unique_words_multi:
                    if word in word_occurrences_across_docs:
                        word_occurrences_across_docs[word] += 1
                    else:
                        word_occurrences_across_docs[word] = 1
            if word in unique_words_multi:
                if word in word_occurrences_per_doc_train[doc_index]:
                    word_occurrences_per_doc_train[doc_index][word] += 1
                else:
                    word_occurrences_per_doc_train[doc_index][word] = 1

    # count their occurrences per docs in test
    word_occurrences_per_doc_test = []

    for doc_index, doc in enumerate(test_data):
        words = re.findall(r'\b\w+\b', doc)
        word_occurrences_per_doc_test.append(dict())

        for word in words:
            if word in unique_words_multi:
                if word in word_occurrences_per_doc_test[doc_index]:
                    word_occurrences_per_doc_test[doc_index][word] += 1
                else:
                    word_occurrences_per_doc_test[doc_index][word] = 1
    
    train_x = []
    test_x = []
    idfs = dict()

    # compute IDFs
    for word in unique_words_multi:
        idfs[word] = np.log(len(train_data) / (word_occurrences_across_docs[word] + 1))
    
    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to counts but normalized to sum to 1);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    
    for doc_index, doc in enumerate(train_data):
        
        features = np.zeros(len(unique_words_multi),)

        for word, val in word_occurrences_per_doc_train[doc_index].items():
            index = unique_words_multi[word]
            if args.tf:
                features[index] = val / np.sum(list(word_occurrences_per_doc_train[doc_index].values()))
            else:
                features[index] = 1

            # Then, if `args.idf` is set, multiply the document features by the
            # inverse document frequencies (IDF), where
            # - use the variant which contains `+1` in the denominator;
            # - the IDFs are computed on the train set and then reused without
            #   modification on the test set.        
            
            if args.idf:
                features[index] *= idfs[word]

        train_x.append(features)

    train_x = np.array(train_x)

    for doc_index, doc in enumerate(test_data):

        features = np.zeros(len(unique_words_multi),)

        for word, val in word_occurrences_per_doc_test[doc_index].items():
            index = unique_words_multi[word]
            if args.tf:
                features[index] = val / np.sum(list(word_occurrences_per_doc_test[doc_index].values()))
            else:
                features[index] = 1

            if args.idf:
                features[index] *= idfs[word]

        test_x.append(features)

    test_x = np.array(test_x)

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear")`
    # model on the train set, and classify the test set.
    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    model.fit(train_x, train_y)

    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    y_true = test_y
    y_pred = model.predict(test_x)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(args.tf, args.idf, f1_score))
