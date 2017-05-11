#!/usr/bin/env python
# Python 3.

from collections import Counter
from collections import defaultdict

import sys
import string
import numpy as np

def process_str(s):
    rem_punc = str.maketrans('', '', string.punctuation)
    return s.translate(rem_punc).lower().split()

def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), words) )

    return dataset

def get_most_commons(dataset, skip=100, total=100):
    counter = Counter()
    for item in dataset:
        counter = counter + Counter(set(item[1]))

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)

def logistic(vector):
    return 1.0 / (1 + np.exp(-vector))

def logistic_regression(features, labels):
    n, d = features.shape
    w = np.zeros(d).T
    w_prev = np.ones(d).T
    i = 0

    # Parameters for logistic regression.
    # max_iter - Maximum iterations.
    # tol - Tolerance value.
    # alpha - Step size.
    # l - L2 regularization penalty.
    max_iter = 100; tol = 1e-6; alpha = 0.01; l = 0.01

    while True:
        if (np.linalg.norm(w - w_prev) < tol) or (i >= max_iter):
            break

        h = logistic(features @ w)
        loss_grad = (features.T @ (h - labels)) + (l * w)
        w_prev = w
        w = w - (alpha * loss_grad)
        i += 1

    return w

def logistic_pred(w, features):
    threshold = 0.5
    pred = np.where(logistic(features @ w) >= threshold, 1, 0)
    return pred

def calc_error(pred, labels):
    error = sum(np.where(pred != labels, 1, 0))
    return (error / labels.size)

def svm(features, labels):
    # test sub-gradient SVM
    total = features.shape[1]
    lam = 1.; D = total
    x = features; y = (labels-0.5)*2
    w = np.zeros(D); wpr = np.ones(D)
    eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
    while True:
        if np.linalg.norm(w-wpr) < tol or i > MAXI:
            break
        f = w @ x.T
        pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
        pL = np.mean(pL,axis=1) + lam*w
        wpr = w
        w = w - eta*pL
        i += 1

    return w

def svm_pred(w, features):
    return np.where((features @ w) >= 0, 1, 0)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        train_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        model_idx = int(sys.argv[3])

        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)

        common_words = get_most_commons(train_data, skip=100, total=4000)

        train_f, train_l = generate_vectors(train_data, common_words)
        test_f, test_l = generate_vectors(test_data, common_words)

        if model_idx == 1:
            w = logistic_regression(train_f, train_l)
            test_pred = logistic_pred(w, test_f)
            print('ZERO-ONE-LOSS-LR', calc_error(test_pred, test_l))
        elif model_idx == 2:
            w = svm(train_f, train_l)
            test_pred = svm_pred(w, test_f)
            print('ZERO-ONE-LOSS-SVM', calc_error(test_pred, test_l))
        else:
            print('Illegal modelIdx')
    else:
        print('usage: python hw3.py train.csv test.csv modelIdx')

