__author__ = "Xu Fangzhou"
__email__ = "kevin.xu.fangzhou@gmail.com"

import scipy.io as sio
import numpy as np
import utilities as util
import time
import spams
import csv

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
    tic = time.time()

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
    Dtemplate = spams.trainDL(X, **param)

    # for each fixed dictionary K, we will repeat dictionary
    # learning for 100 times, each with a different initial value
    test_cases = 10
    R = np.zeros((test_cases, ))
    for i in xrange(0, test_cases):
        if randomStart == 1:
            D0 = util.dictLearnInit(X, K, 'random', 0)
        param['D'] = np.asfortranarray(D0)
        lparam = {'lambda1': Lambda,
                  'pos': True,
                  'mode': 2,
                  'numThreads': -1
        }
        D = spams.trainDL(X, **param)
        alpha = spams.lasso(X, D, **lparam)
        R[i] = np.mean(0.5 * sum((X - D * alpha) ** 2) + param['lambda1'] * sum(abs(alpha)))
        print R[i]

        #(permInd, cost) = util.dissimilarityDict(Dtemplate, D, 'euclidean')
        #D = D[:, permInd] # permute the columns of D to match the template Dtemplate

        if i >= 1:
            if R[i] < Rbest:
                Dbest = D
                Rbest = R[i]
        else:
            Dbest = D
            Rbest = R[0]

    # print path
    toc = time.time()
    print 'Elapsed time is ' + str(toc - tic) + ' seconds.'
    print Dbest
    sio.savemat(path + "bestDict.mat", {'Ftemplate': Dtemplate, 'Dbest': Dbest, 'R': Rbest})

    wt = csv.writer(open("output.csv", 'w'))
    wt.writerow([i for i in range(1,K+1)])
    for i in range(0, len(Dbest[0])):
        wt.writerow([Dbest[j][i] for j in range(0,K)])
