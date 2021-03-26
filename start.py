import numpy as np
import pandas as pd
from algorithms import SVM_classifier, logistic_regression, KernelPCA
from kernels import mismatch_kernel, normalize_gram


# Loading the data
Xtr0 = pd.read_csv("Xtr0.csv", index_col="Id")
Xtr1 = pd.read_csv("Xtr1.csv", index_col="Id")
Xtr2 = pd.read_csv("Xtr2.csv", index_col="Id")

Xte0 = pd.read_csv("Xte0.csv", index_col="Id")
Xte1 = pd.read_csv("Xte1.csv", index_col="Id")
Xte2 = pd.read_csv("Xte2.csv", index_col="Id")

Ytr0 = pd.read_csv("Ytr0.csv", index_col="Id")
Ytr1 = pd.read_csv("Ytr1.csv", index_col="Id")
Ytr2 = pd.read_csv("Ytr2.csv", index_col="Id")

# Our best parameters
N_split = 2000
k, m = 9, 1
C = 0.4


# Making the prediction for each dataset
sequences0 = list(Xtr0["seq"]) + list(Xte0["seq"])
y_train0 = np.array(list(Ytr0["Bound"]))
K0 = mismatch_kernel(sequences0, k, m)
alpha0, y_pred0 = SVM_classifier(K0, N_split, y_train0[:N_split], C=C, algo='2-SVM')


sequences1 = list(Xtr1["seq"]) + list(Xte1["seq"])
y_train1 = np.array(list(Ytr1["Bound"]))
K1 = mismatch_kernel(sequences1, k, m)
alpha1, y_pred1 = SVM_classifier(K1, N_split, y_train1[:N_split], C=C, algo='2-SVM')

sequences2 = list(Xtr2["seq"]) + list(Xte2["seq"])
y_train2 = np.array(list(Ytr2["Bound"]))
K2 = mismatch_kernel(sequences2, k, m)
alpha2, y_pred2 = SVM_classifier(K2, N_split, y_train2[:N_split], C=C, algo='2-SVM')

y_predict = np.concatenate((y_pred0.T[0], y_pred1.T[0], y_pred2.T[0]))
Yte = pd.DataFrame({'Bound': y_predict})
Yte.index.name = 'Id'
Yte.to_csv("Yte.csv")