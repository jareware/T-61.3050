# pylint: disable=C0301
"""
Term project for t-61.3050: Machine Learning: Basic Principles (5 cr)
Autumn 2012

https://noppa.aalto.fi/noppa/kurssi/t-61.3050/term_project

(C) Olli Jarva

"""
import datetime
import os
import itertools
import mlpy
import numpy as np
import sys
import json



def optimize_parameters(vectors, labels, data):
    """ Try to improve parameters even further by searching neighborhoods """
    print "Optimizing gamma and C with n=%s" % (len(vectors))

    def run(k_folded, param_gamma, param_c):
        """ Single run of learning and prediction. Returns percentage of correct classifications """
        print "Running with ", param_gamma, param_c,

        x_a, y_a, clearx_a, cleary_a = k_folded
        correct = wrong = 0

        # x_a, y_a, clearx_a, cleary_a are lists of lists (k-folded dataset)
        for set_index in range(0, len(x_a)):

            svm = create_and_teach_svm(x_a[set_index], y_a[set_index], param_gamma, param_c)

            for test_index in range(0, len(clearx_a[set_index])):
                correct_answer = cleary_a[set_index][test_index]
                prediction = int(svm.pred(clearx_a[set_index][test_index]))

                if prediction == correct_answer:
                    correct += 1
                else:
                    wrong += 1
            print ".",
        print ""
        return float(correct) / (float(correct) + float(wrong))

    def run_grid_point(k_folded, gamma_power, c_power, data):
        """ Run single point from grid and store results """
        gamma_round = gamma_power
        c_round = c_power

        corpercent = run(k_folded, gamma_round, c_round)
        if corpercent > data["correct"]:
            data["correct"] = corpercent
            data["gamma"] = gamma_round
            data["c"] = c_round
            data["gamma_best_exponent"] = gamma_power
            data["c_best_exponent"] = c_power
            print "Found new parameters: c=%s, gamma=%s, per=%s" % (data["c"], data["gamma"], data["correct"])
        return data

    labels_distribution = get_distribution(labels)
    if len(labels_distribution) == 1:
        print "Invalid label distribution:", labels_distribution
        sys.exit(1)
    print "Label distribution:", labels_distribution


    if data.get("no-better-solution-found-4", False):
        return data

    # k-folding
    k_folded = kfold(vectors, labels, 3)

    variances = [0.1, 0.05, 0.01]
    variances.extend([-v for v in variances])
    
    gamma_range = [data["gamma"] * (1+variance) for variance in variances]
    c_range = [data["c"] * (1+variance) for variance in variances]

    cor_orig = data["correct"]

    for param_gamma, param_c in itertools.product(gamma_range, c_range):
         data = run_grid_point(k_folded, param_gamma, param_c, data)

    if data["correct"] == cor_orig:
        data["no-better-solution-found-4"] = True

    return data

def get_distribution(labels):
    """ calculate distribution of labels """
    values_tmp = {}
    for item in labels:
        if item not in values_tmp:
            values_tmp[item] = 0
        values_tmp[item] += 1
    return values_tmp

def kfold(vectors, labels, count):
    """ Returns k-folded sets as a tuple of 
         (training_x, training_y, testing_x, testing_y) """
    x_splitted = []
    y_splitted = []
    clearx_splitted = []
    cleary_splitted = []
    for training, testing in mlpy.cv_kfold(n=len(vectors), k=count):
        x_splitted.append([vectors[ind] for ind in training])
        y_splitted.append([labels[ind] for ind in training])
        clearx_splitted.append([vectors[ind] for ind in testing])
        cleary_splitted.append([labels[ind] for ind in testing])
    return (x_splitted, y_splitted, clearx_splitted, cleary_splitted)


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

def pick_values(vectors, labels, values):
    """ Pick entries from vectors and labels, if
        label is in "values" """

    data = {"x": [], "y": []}
    for index, value in enumerate(labels):
        if value in values:
            data["x"].append(vectors[index])
            data["y"].append(value)
    return data

def load_parameters(filename, current_parameters, key):
    try:
        parameters = json.load(open(filename))
        print "Loaded %s for %s" % (parameters, key)
        if len(parameters) == 0:
            return current_parameters
        if parameters.get("no-better-solution-found-4", False):
            current_parameters["no-better-solution-found-4"] = True

        if parameters["correct"] > current_parameters.get("correct", 0):
            return parameters

    except IOError:
        pass
    return current_parameters
    

def get_svm_parameters(vectors, labels, special_sets_values):
    """ Try to load SVM parameters from files. If parameters are not available,
        run grid search and store new parameters """

    special_svm_parameters = {}

    for key in special_sets_values:
        parameters = {}

        best_found = in_progress = False

        for counter in range(2, 25):
            filename = "params%s-oneway-%s-%s.json" % (counter, key, special_sets_values[key])
            if not os.path.exists(filename):
                break
            if os.path.exists(filename+".inprogress"):
                in_progress = True
                break
            if parameters.get("no-better-solution-found-4", False):
                best_found = True
                break
            parameters = load_parameters(filename, parameters, key)

        if best_found:
            print "Best parameter for %s found. Skip." % key
            continue

        if in_progress:
            print "File is locked. Skipping."
            continue

        if len(parameters) == 0:
            print "Invalid file: %s" % filename
            continue


        filename = "params%s-oneway-%s-%s.json" % (counter, key, special_sets_values[key])

        label_list = special_sets_values[key] + [key]
        values = pick_values(vectors, labels, label_list)

        open(filename+".inprogress", "w").write(str(datetime.datetime.now()))
        json.dump({}, open(filename, "w")) # Trivial locking
        parameters = optimize_parameters(values['x'], values['y'], parameters)
        json.dump(parameters, open(filename, "w"))
        os.remove(filename+".inprogress")

        special_svm_parameters[key] = parameters

    return special_svm_parameters

def main():
    """ Load data, parameters and run """

    vectors, labels = load("train.normalized.txt")

    data = {"correct": 0, "wrong": 0, # overall statistics
             "ssvm_corrected": 0, "ssvm_errors": 0, # secondary SVMs
             "wrong_class": {}, # wrongly classified labels
             "wrong_class_after": {}, # after secondary SVMs
             "kfold_k": 20, # k-fold value
             "gamma": 0.008249487, # gamma for "master" SVM
             "c": 18, # C for "master" SVM
             "special_sets_values": {0: [4, 14, 13], 1: [7, 3], 2: [17], 3: [0], 
                           4: [2], 5: [19, 17], 6: [14, 0], 7: [1, 13], 
                           8: [11], 9: [8], 10: [1, 4, 7], 11: [2, 8], 
                           12: [13], 13: [0, 20], 14: [0], 15: [6, 17], 
                           16: [6], 17: [5, 21], 18: [6], 19: [5, 17], 
                           20: [0, 13, 21], 21: [20], 22: [13], 23: [13], 
                           24: [6], 25: [4]}
            }


    data = {"correct": 0, "wrong": 0, # overall statistics
             "ssvm_corrected": 0, "ssvm_errors": 0, # secondary SVMs
             "wrong_class": {}, # wrongly classified labels
             "wrong_class_after": {}, # after secondary SVMs
             "kfold_k": 20, # k-fold value
             "gamma": 0.008249487, # gamma for "master" SVM
             "c": 18, # C for "master" SVM
             "special_sets_values": {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}
            }

    # Load secondary SVM parameters:
    special_svm_parameters = get_svm_parameters(vectors, labels, data["special_sets_values"])

if __name__ == '__main__':
    main()
