__author__ = "Xu Fangzhou"
__email__ = "kevin.xu.fangzhou@gmail.com"

import os
execfile(os.environ.get("PYSPARK_INIT"))

import scipy.io as sio
import numpy as np
#import utilities as util
import time
import spams
from munkres import Munkres
# must install munkres 1.0.6: https://pypi.python.org/pypi/munkres/

def dictLearnInit(X, K, method, p = 0):
    if method in ["random", "exact"]:
        ind = np.random.randint(low = 0, high = X.shape[1], size = K)
        D = X[:, ind]
        return D

def dissimilarityDict(D1, D2, method):
    if method == 'euclidean':
        disMat = np.zeros((D1.shape[1], D2.shape[1]))
        # inefficient here
        for i in xrange(0, D1.shape[1]):
            for j in xrange(0, D2.shape[1]):
                disMat[i][j] = np.sum((D1[:, i] - D2[:, j]) ** 2)
        matrix = disMat.tolist()
        munkres = Munkres()
        indexes = munkres.compute(matrix)
        matchInd = np.zeros((D1.shape[1], ), dtype=int)
        cost = 0
        for row, col in indexes:
            matchInd[i] = col
            cost += matrix[row][col]
    return (matchInd, cost)


resolution = '32by16'
mat_contents = sio.loadmat('./data/finalData/data32by16.mat')
X = mat_contents['X']
geneNames = mat_contents['geneNames']
template = mat_contents['template']

width = 32
height = 16
# different from MATLAB
ind = np.where(template[:, :, 0].ravel() == 1)[0]
Y = np.zeros((height * width, X.shape[1]))
Y[ind, :] = X
Load = 1

(m, n) = X.shape

# truncating ...
X[X < 0] = 0
X[X > 1] = 1

numPatterns = range(21, 22) # dictionary sizes
randomStart = 1 # initialize dictionary learning by randomly selecting K images from the data
noiseStart = 0 # initialize dictionary learning from random noise image lambda = 2*size(X, 1)/405;
Lambda = 0 # sparsity control for the coefficients alpha
gamma1 = 0 # sparsity control on the dictionary patterns
doShift = 0 # do we want to shift the image

len_numPatterns = len(numPatterns)
for k in range(0, len_numPatterns):

    K = numPatterns[k]
    print 'Pattern ' + str(K) + ':'

    path = ''
    if doShift == 1:
        print 'To be continued'
    else:
        if randomStart == 1:
            D0 = util.dictLearnInit(X, K, 'random', 0)
            path = './' + resolution + '/randomStart/K=' + str(K) + '/'
        # print path
    # os.makedirs(path)
    param = {'mode': 2,
             'K': K, # learns a dictionary with K elements
             'lambda1': Lambda, # number of threads
             'numThreads': -1,
             'batchsize': min(1024, n), # positive dictionary
             'posD': True, # positive dictionary
             'iter': 500, # number of iteration
             'modeD': 0,
             'verbose': 0, # print out update information?
             #'pos': 1, # positive alpha
             'posAlpha': 1, # positive alpha
             'gamma1': gamma1, # penalizing parameter on the dictionary patterns
             'D': np.asfortranarray(D0) # set initial values
    }

    X = np.asfortranarray(X, dtype=float)
    # print X.shape

    print "getting Dtemplate..."
    tic = time.time()
    Dtemplate = spams.trainDL(X, **param)
    toc = time.time()
    print "finished. time passed "+str(toc-tic)+" s"

    # for each fixed dictionary K, we will repeat dictionary
    # learning for 100 times, each with a different initial value
    test_cases = 1
    R = []
    for i in range(test_cases):
        R += [X]
    rdd = sc.parallelize(R)

    def altp(x,p): # alter the D attribute in the param
        p['D'] = np.asfortranarray( util.dictLearnInit(x, p['K'], 'random', 0) )
        return (x,p)

    def calcR((X,p,D,alpha)):
        return np.mean(0.5 * sum((X - D * alpha) ** 2) + p['lambda1'] * sum(abs(alpha)))

    def processD(d):
        (permInd, cost) = dissimilarityDict(Dtemplate, d, 'euclidean')
        return d[:, permInd]

    def findBest(E1, E2):
        if E1[1] > E2[1]:
            return E2
        return E1

    lparam = {'lambda1': Lambda,
                  'pos': True,
                  'mode': 2,
                  'numThreads': -1
        }

    print "start rdd transition."
    tic = time.time()
    p = (rdd.map(lambda x: altp(x, param))
            .map(lambda (x,p): (x,p,spams.trainDL(x,**p)))
            .map(lambda (x,p,D): (x,p,D,spams.lasso(x, D, **lparam)))
            .map(lambda e: (e[2], calcR(e)))
            #.map(lambda (d, r): (processD(d), r))
            .reduce(findBest)
    )
    toc = time.time()
    print "all finished. time passed "+str(toc-tic)+" s"
    print p

    sio.savemat(path + "bestDict.mat", {'DTemplate': Dtemplate, 'Dbest': p[0], 'R': p[1]})
