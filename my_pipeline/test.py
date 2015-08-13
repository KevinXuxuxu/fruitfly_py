import numpy as np
import spams

param = {'K': 100, 'lambda1': 0.15, 'numThreads': 4, 'batchsize': 400, 'iter': 10}
X = np.zeros((5, 5), dtype=float)
X = np.asfortranarray(X)
D = spams.trainDL(X, **param)
