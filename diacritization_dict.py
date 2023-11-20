#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train_test.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

def get_class_from_char(char):
    if char in "áéíóúý":
        return 1
    if char in "čďěňřšťůž":
        return 2
    return 0

def pred_to_char(pred, char):
    if pred == 1:
        index = "aeiouy".find(char)
        return "áéíóúý"[index] if index >= 0 else char
    if pred == 2:
        index = "cdenrstuz".find(char)
        return "čďěňřšťůž"[index] if index >= 0 else char
    return char

def extract_features(data, target):
    indices = []
    features = [] 
    targets = []
    
    text = data.lower()
    candidate_chars = "acdeeinorstuuyz"

    for index in range(len(text)):
        if text[index] in candidate_chars:
            item_features = [text[index]]

            for offset in range(1, 6):
                item_features.append(text[index-offset:index-offset+1])
                item_features.append(text[index+offset:index+offset+1])

            for size in range(1, 6):
                for offset in range(-size, 1):
                    if index + offset > 0:
                        item_features.append(text[index + offset:index + offset + size + 1])
                    else:
                        item_features.append(text[0:index + offset + size + 1])
            
            features.append(item_features)
            targets.append(get_class_from_char(target[index].lower()))
            indices.append(index)
    
    return features, targets, indices

def generate_predictions(data, targets, indices):
    preds = list(data)
    for i in range(len(targets)):
        pred_char = pred_to_char(targets[i], data[indices[i]].lower())
        preds[indices[i]] = pred_char.upper() if data[indices[i]].isupper() else pred_char
    
    return ''.join(preds)

def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        train_data, train_target, _ = extract_features(train.data, train.target)
        train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=.2)

        # TODO: Train a model on the given dataset and store it in `model`.
        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(solver='saga', multi_class='multinomial', verbose=1)
        )

        model.fit(train_data, train_target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        dictionary = Dictionary()
        
        test_data, _, test_indices = extract_features(test.data, test.target)
        log_probs = model.predict_log_proba(test_data)
        test_target = model.predict(test_data)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = generate_predictions(test.data, test_target, test_indices)
        improved_preds = []
        candidate_chars = "acdeeinorstuuyz"

        prob_index = 0
        for word in predictions.split():
            prob_indices = []
            translated_word = word.translate(test.DIA_TO_NODIA).lower()

            for char in translated_word:
                if char in candidate_chars:
                    prob_indices.append(prob_index)
                    prob_index += 1

            if translated_word in dictionary.variants:
                
                variants = dictionary.variants[translated_word]
                
                if len(variants) == 1:
                    improved_preds.append(variants[0])
                
                else:

            
                    print(word, translated_word, prob_indices)
                    start = prob_indices[0]
                    end = prob_indices[-1] + 1

                    variant_scores = []
                    for variant in variants:
                        chars = []
                        for item in test_data[start:end]:
                            chars.append(item[0])
                        print(chars)
                        variant_score = 0
                        char_index = 0
                        for char in variant:
                            if char in test.LETTERS_DIA + test.LETTERS_NODIA:
                                
                                
                                char_class = get_class_from_char(char)
                                char_prob = log_probs[start:end][char_index][char_class]
                                print(char, char_prob)
                                variant_score += char_prob
                                char_index += 1
                        variant_scores.append(variant_score)
                        print("VARIANT:", variant, "SCORE:", variant_score)
                        print()
                    best_variant_index = np.argmax(variant_scores)
                    best_variant = variants[best_variant_index]
                    print("BEST:", best_variant)
                    print()
                    improved_preds.append(best_variant)

            else:
                improved_preds.append(word)
            
        improved_preds = ' '.join(improved_preds)

        result = ""

        for i in range(len(improved_preds)):
            if test.data[i].isupper():
                result += improved_preds[i].upper()
            else:
                result += improved_preds[i]
        
        print(result)

        return result

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)