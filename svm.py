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



def optimize_parameters(vectors, labels):
    """ Simple grid search for finding RFB kernel parameters """
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
        gamma_round = 2 ** gamma_power
        c_round = 2 ** c_power

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

    # k-folding
    k_folded = kfold(vectors, labels, 3)

    data = {"c": 0,       # values used for teaching SVM
            "gamma": 0,
            "correct": 0,
            "best_result": 0, # Values used for grid search
            "gamma_best": 0, # actual parameter values
            "c_best": 0,
            "gamma_best_exponent": 0, # exponent value
            "c_best_exponent": 0
           }

    # Coarse search
    for gamma_power in range(-10, 2, 2):
        for c_power in range(-1, 11, 2):
            data = run_grid_point(k_folded, gamma_power, c_power, data)
    
    # Finer search
    for gamma_power in np.arange(data["gamma_best_exponent"] - 1, data["gamma_best_exponent"] + 1, 0.5):
        for c_power in np.arange(data["c_best_exponent"] - 1, data["c_best_exponent"] + 1, 0.5):
            data = run_grid_point(k_folded, gamma_power, c_power, data)

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

def get_svm_parameters(vectors, labels, special_sets_values):
    """ Try to load SVM parameters from files. If parameters are not available,
        run grid search and store new parameters """

    special_svm_parameters = {}
    for key in special_sets_values:
        filename = "data/params-oneway-%s-%s.json" % (key, special_sets_values[key])

        run = False
        try:
            parameters = json.load(open(filename))
            print "Loaded %s for %s" % (parameters, key)
        except IOError:
            run = True

        if run:
            label_list = special_sets_values[key] + [key]
            values = pick_values(vectors, labels, label_list)

            json.dump({}, open(filename, "w")) # Trivial locking
            parameters = optimize_parameters(values['x'], values['y'])
            json.dump(parameters, open(filename, "w"))

        special_svm_parameters[key] = parameters

    return special_svm_parameters


def predict_character(**kwargs):
    """ Predict single character """
    data = kwargs["data"]

    correct_answer = kwargs["cleary"][kwargs["index"]]
    prediction = int(kwargs["svm"].pred(kwargs["clearx"][kwargs["index"]]))

    if prediction != correct_answer:
        if correct_answer not in data["wrong_class"]:
            data["wrong_class"][correct_answer] = {}
        if prediction not in data["wrong_class"][correct_answer]:
            data["wrong_class"][correct_answer][prediction] = 0
        data["wrong_class"][correct_answer][prediction] += 1
 
    final_prediction = prediction
    if prediction in kwargs["special_sets_values"]:
        secondary_prediction = int(kwargs["special_svms"][prediction].pred(kwargs["clearx"][kwargs["index"]]))
        final_prediction = secondary_prediction
        if secondary_prediction != prediction:
            print "Changed", prediction, "to", secondary_prediction, ". Correct answer is", correct_answer
            if secondary_prediction == correct_answer:
                data["ssvm_corrected"] += 1
            else:
                if prediction == correct_answer:
                    data["ssvm_errors"] += 1

    if final_prediction != correct_answer:
        if correct_answer not in data["wrong_class_after"]: 
            data["wrong_class_after"][correct_answer] = {}
        if final_prediction not in data["wrong_class_after"][correct_answer]:
            data["wrong_class_after"][correct_answer][final_prediction] = 0
        data["wrong_class_after"][correct_answer][final_prediction] += 1

    if prediction == correct_answer:
        data["correct"] += 1
    else:
        data["wrong"] += 1
    return data


def main():
    """ Load data, parameters and run """

    vectors, labels = load("data/train.normalized.txt")

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

    # Load secondary SVM parameters:
    special_svm_parameters = get_svm_parameters(vectors, labels, data["special_sets_values"])

    data["special_svm_parameters"] = special_svm_parameters
    # k-folding
    x_splitted, y_splitted, clearx_splitted, cleary_splitted = kfold(vectors, labels, data["kfold_k"])

    # Crossvalidation: loop over k-folded sets:
    for round_index in range(0, data["kfold_k"]):
        # Secondary SVMs and secondary SVM datasets:
        special_svms = {}

        # Train secondary SVMs
        for key in data["special_sets_values"]:
            values = pick_values(x_splitted[round_index], y_splitted[round_index], data["special_sets_values"][key] + [key])
            svm_tmp = create_and_teach_svm(values['x'], values['y'], special_svm_parameters[key]["gamma"], special_svm_parameters[key]["c"])
            special_svms[key] = svm_tmp

        # Train primary SVM
        svm = create_and_teach_svm(x_splitted[round_index], y_splitted[round_index], data["gamma"], data["c"])

        # Run testing dataset:
        for index in range(0, len(clearx_splitted[round_index])):
            data = predict_character(clearx=clearx_splitted[round_index], cleary=cleary_splitted[round_index], index=index, data=data, svm=svm, special_svms=special_svms, special_sets_values=data["special_sets_values"])

        data["corpercent"] = float(data["correct"]) / (float(data["correct"]) + float(data["wrong"]))
        data["corpercent_after"] = float(data["correct"] + data["ssvm_corrected"] - data["ssvm_errors"]) / (float(data["correct"]) + float(data["wrong"]))
        print "per=%s, per_after=%s, ssvm_corrected=%s, ssvm_errors=%s, kfold_counter=%s/%s" % (data["corpercent"], data["corpercent_after"], data["ssvm_corrected"], data["ssvm_errors"], round_index + 1, data["kfold_k"])
    print "gamma=%s, C=%s, per=%s" % (data["gamma"], data["c"], data["corpercent"])
    print_statistics(data)


def print_statistics(data):
    """ Print basic statistics and data """
    print "Correct before secondary SVMs: %s" % data["corpercent"]
    print "Correct after secondary SVMs: %s" % data["corpercent_after"]

    print "Characters classified incorrectly:"
    print data["wrong_class"]

    print "Characters classified incorrectly after secondary SVMs"
    print data["wrong_class_after"]




if __name__ == '__main__':
    main()
