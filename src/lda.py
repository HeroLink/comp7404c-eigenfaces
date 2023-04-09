import numpy as np
import numpy.linalg as LA
from utils import report


def class_mean_num(X, y, classes):
    """ Compute the number of samples and means of each class """
    class_num = []
    class_mean = []
    for c in classes:
        # x_i shape (#class_i_samples, n_features)
        x_i = X[y == c]
        class_num.append(x_i.shape[0])
        # mean of class i, shape (n_features,)
        mu_i = np.mean(x_i, axis=0)
        class_mean.append(mu_i)
    return class_num, class_mean


def between_class_scatter(X, class_num, class_mean):
    """ Compute the between-class scatter matrix """
    n_features = X.shape[1]
    # Compute global mean, shape (n_features,) -> (n_features, 1)
    global_mean = np.mean(X, axis=0).reshape(-1, 1)
    S_b = np.zeros((n_features, n_features))
    for i, mean in enumerate(class_mean):
        n_i = class_num[i]
        # mean shape (n_features,) -> (n_features, 1)
        mean = np.reshape(mean, (-1, 1))
        sub = mean - global_mean
        S_b += n_i * np.dot(sub, sub.T)
    return S_b


def within_class_scatter(X, y, classes, class_mean):
    """ Compute the within-class scatter matrix """
    n_features = X.shape[1]
    S_w = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        x_i = X[y == c]
        sub = x_i - class_mean[i]
        S_i = np.dot(sub.T, sub)
        S_w += S_i
    return S_w


def lda(X, y, k=None):
    """ Linear Discriminant Analysis (LDA)
        X shape (n_samples, n_features)
        y shape (n_samples,)
        k should be less than min(n_classes - 1, n_features) """
    if not k:
        k = min(len(np.unique(y)) - 1, X.shape[1])

    # Compute mean and number of samples in each class
    classes = np.unique(y)
    class_num, class_mean = class_mean_num(X, y, classes)
    # assert n_samples == np.sum(class_n_smp)

    # Compute between-class scatter matrix
    S_b = between_class_scatter(X, class_num, class_mean)

    # Compute within-class scatter matrix
    S_w = within_class_scatter(X, y, classes, class_mean)

    # Compute normalized eigenvalues and eigenvectors of Sw^(-1)Sb
    # A shape (n_features, n_features)
    A = np.dot(LA.inv(S_w), S_b)
    U, s, Vt = LA.svd(A)
    eigval = s ** 2
    eigvec = Vt.T
    
    # Select the first k eigenvectors as matrix for fisher's criterion
    # W shape (k, n_features) -> (n_features, k)
    W = eigvec[:, :k]

    # Project the data onto the LDA space
    X_lda = np.dot(X, W)
    return X_lda, W
