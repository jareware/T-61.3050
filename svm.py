import mlpy
import numpy as np
import sys

data = np.loadtxt("data/train.normalized.txt", delimiter=",")

x, y = data[:, 1:], data[:, 0].astype(np.int)

"""
clearx = x[40000:]
cleary = y[40000:]
x = x[:40000]
y = y[:40000]
"""

svm_type = "c_svc"
kernel_type = "rbf"

#n=:2000, clear=40000: % 0.594330855
gamma = ogamma = 0.008249487
C = oC = 18

gamma_fix = C_fix = 0

#for C_fix in [-0.1, 0, 0.1]:
#   for gamma_fix in [-gamma/250, 0, gamma/250]:

# Run k-fold crossvalidation:
for tr, ts in mlpy.cv_kfold(n=len(x), k=20):
    x_round = [x[ind] for ind in tr]
    y_round = [y[ind] for ind in tr]
    clearx_round = [x[ind] for ind in ts]
    cleary_round = [y[ind] for ind in ts]
    gamma = ogamma + gamma_fix
    C = oC + C_fix
    svm = mlpy.LibSvm(kernel_type=kernel_type, svm_type=svm_type, gamma=gamma, C=C, probability=True)
    svm.learn(x_round, y_round)

    correct = wrong = 0

    for i in range(0, len(clearx_round)):
      pr = svm.pred(clearx_round[i])
      if int(pr) == cleary_round[i]:
        correct += 1
      else:
        wrong += 1

    corpercent = float(correct)/(float(correct)+float(wrong))
    print "gamma", gamma, "C", C, "%",corpercent
