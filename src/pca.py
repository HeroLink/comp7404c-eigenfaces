import numpy as np
import numpy.linalg as LA


def mean_face(X):
    """ Compute the mean face """
    return np.mean(X, axis=0)


def cov_matrix(X, mean):
    """ Compute the covariance matrix  """
    sub = X - mean
    # cov= np.dot(sub.T, sub)
    # return cov / len(X)

    # Trick: (N,d x d,N = N,N), calculating eigenvectors cost 2.6s. But the result is problematic.
    # (d,N x N,d = d,d), calculating eigenvectors cost 8mins 14s.
    # Quesiton: d eigenvalues VS N eigenvalues
    # Answer: Top k is the same

    cov = np.dot(sub, sub.T)
    return cov / len(X)


def eigen(X, mean):
    """ Compute the eigenvalues and eigenvectors """
    # Compute covariance matrix
    cov = cov_matrix(X, mean)

    # Compute normalized eigenvectors and eigenvalues of covariance matrix
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh
    # eigh: 3m 6s, eig: 8m 14s
    # import os
    # if not os.path.exists('data'):
    #     os.mkdir('data')
    # if not os.path.exists('data/eigval.npy') or not os.path.exists('data/eigvec.npy'):
    #     eigval, eigvec = LA.eigh(cov)
    #     # Save eigenvalues and eigenvectors
    #     np.save('data/eigval.npy', eigval)
    #     np.save('data/eigvec.npy', eigvec)
    # else:
    #     eigval = np.load('data/eigval.npy')
    #     eigvec = np.load('data/eigvec.npy')
    # return eigval, eigvec

    eigval, eigvec = LA.eigh(cov)
    # Convert to the original eigenvectors, (d, N)
    eigvec = np.dot(X.T, eigvec)

    # NOTE: Normalize should be element-wise, not row-wise
    # Row-wise normalization
    # eigvec = eigvec / LA.norm(eigvec, axis=1).reshape(-1, 1)
    # Element-wise normalization
    # eigvec = eigvec / LA.norm(eigvec)

    # Column-wise normalization
    eigvec = eigvec / LA.norm(eigvec, axis=0)
    return eigval, eigvec


def PCA(X, mean, k=150):
    """ Principal components analysis with top k components """
    eigval, eigvec = eigen(X, mean)
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigval)[::-1][:k]
    eigface = eigvec[:, idx]
    # Compute weights
    weights = np.dot(X - mean, eigface)
    return eigface, weights


def project(X, mean, eigface):
    """ Project input image onto eigenfaces """
    # Subtract mean face
    sub = X - mean
    return np.dot(sub, eigface)


def euclidean(proj, weights):
    """ Compute the Euclidean distance between projection and training weights """
    dist = []
    if len(proj.shape) == 1:
        proj = proj.reshape(1, -1)
    for p in proj:
        sub = p - weights
        d = LA.norm(sub, axis=1)
        dist.append(d)
        # print(d.shape)
    return np.array(dist)


def pipeline(X_train, X_test, y_train, k=150):
    """ Pipeline for face recognition using eigenfaces """
    # Compute mean face
    mean = mean_face(X_train)
    # Compute eigenfaces and weights
    eigface, weights = PCA(X_train, mean, k)
    # Project development images onto eigenfaces
    proj = project(X_test, mean, eigface)
    # Compute Euclidean distance
    dist = euclidean(proj, weights)
    # Find the smallest distance of each dev image
    idx = np.argmin(dist, axis=1)
    # Get the predicted labels
    y_pred = y_train[idx]
    return y_pred
