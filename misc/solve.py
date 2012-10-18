# pylint: disable=C0301
"""
Term project for t-61.3050: Machine Learning: Basic Principles (5 cr)
Autumn 2012

https://noppa.aalto.fi/noppa/kurssi/t-61.3050/term_project

(C) Olli Jarva

"""

import mlpy
import numpy as np
import sys
import json

def create_and_teach_svm(vectors, labels, param_gamma, param_c):
    """ Create and teach SVM """
    svm_type = "c_svc"
    kernel_type = "rbf"

    svm = mlpy.LibSvm(kernel_type=kernel_type, svm_type=svm_type, gamma=param_gamma, C=param_c)
    svm.learn(vectors, labels)

    return svm


def load(filename):
    """ Load data from file and split to labels and data vectors """
    data = np.loadtxt(filename, delimiter=",")
    # x: data, y: labels
    vectors, labels = data[:, 1:], data[:, 0].astype(np.int)

    return (vectors, labels)

def predict_character(character, svm):
    """ Predict single character """

    return int(svm.pred(character))


def main():
    """ Load data, parameters and run """

    vectors, labels = load("train.normalized.txt")
    test_vectors, test_labels = load("test.normalized.txt")


#"c": 4.0, "best_result": 0, "c_best": 0, "correct": 0.8991506927310685, "gamma": 0.044194173824159223,
#c=4.2, gamma=0.053033008589
    data = {"correct": 0, "wrong": 0, # overall statistics
             "ssvm_corrected": 0, "ssvm_errors": 0, # secondary SVMs
             "wrong_class": {}, # wrongly classified labels
             "wrong_class_after": {}, # after secondary SVMs
             "kfold_k": 20, # k-fold value
             "gamma": 0.053033008589, # gamma for "master" SVM
             "c": 4.2, # C for "master" SVM
             "special_sets_values": {},
            }


#{0: [4, 14, 13], 1: [7, 3], 2: [17], 3: [0], 4: [2], 5: [19, 17], 6: [14, 0], 7: [1, 13], 8: [11], 9: [8], 10: [1, 4, 7], 11: [2, 8],  12: [13], 13: [0, 20], 14: [0], 15: [6, 17],  16: [6], 17: [5, 21], 18: [6], 19: [5, 17],  20: [0, 13, 21], 21: [20], 22: [13], 23: [13],  24: [6], 25: [4]}


    # Train primary SVM
    svm = create_and_teach_svm(vectors, labels, data["gamma"], data["c"])

    for character in test_vectors:
        prediction = predict_character(character, svm)
        print prediction


if __name__ == '__main__':
    main()
