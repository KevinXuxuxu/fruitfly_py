import numpy as np
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

# Test dissimilarityDict
# if __name__ == "__main__":
#     A = np.asarray([[1, 2], [1, 2]])
#     B = np.asarray([[2, 5], [2, 5]])
#     (ind, cost) = dissimilarityDict(A, B, 'euclidean')
#     print ind, cost