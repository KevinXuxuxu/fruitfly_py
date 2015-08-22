__author__ = "Xu Fangzhou"
__email__ = "kevin.xu.fangzhou@gmail.com"

import os
execfile(os.environ.get("PYSPARK_INIT"))

import scipy.io as sio
import numpy as np
import utilities as util
import time
import spams
from munkres import Munkres
import pyspark.mllib.recommendation as pmr
# must install munkres 1.0.6: https://pypi.python.org/pypi/munkres/

def dictLearnInit(X, K, method, p = 0):
    if method in ["random", "exact"]:
        ind = np.random.randint(low = 0, high = X.shape[1], size = K)
        D = X[:, ind]
        return D

resolution = '32by16'
mat_contents = sio.loadmat('./data/finalData/data32by16.mat')
X = mat_contents['X']

(m, n) = X.shape

# truncating ...
X[X < 0] = 0
X[X > 1] = 1
Lambda = 0 # sparsity control for the coefficients alpha

XOrged = [] # organize X into (user, production, rating) tuple
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        XOrged += [(i, j, X[i][j])]

tic = time.time()
Ks = [21]
for K in Ks:
    path = "./" + resolution + "/mllib/K=" + str(K) + "/"
    try:
        os.listdir(path)
    except Exception as e:
        os.mkdir(path)

    test_cases = 5
    DwR = []

    for i in range(test_cases):
        ratings = sc.parallelize(XOrged)
        model = pmr.ALS.train(ratings, K, nonnegative=True)
        D = np.array(model.userFeatures().sortByKey().map(lambda (w,fs): fs).collect())
        lparam = {'lambda1': Lambda,
                    'pos': True,
                    'mode': 2,
                    'numThreads': -1}
        alpha = spams.lasso(X, np.asfortranarray(D), **lparam)
        R = np.mean(0.5 * sum((X - D * alpha) ** 2) + Lambda * sum(abs(alpha)))
        DwR += [(D, R)]

    dwr = sc.parallelize(DwR)

    def findBest(E1, E2):
        if E1[1] > E2[1]:
            return E2
        return E1

    (D, R) = dwr.reduce(findBest)
    sio.savemat(path + "bestMat", {'Dbest': D, "R": R})
toc = time.time()
print "time passed "+str(toc-tic)+" s."
