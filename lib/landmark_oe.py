import torch
from torch.autograd import grad
import numpy as np
from sklearn.datasets import make_moons
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_utils.logging_util import my_custom_logger
from preprocessing_utils.oracle import Oracle
from preprocessing_utils.TripletData import triplet_error
from preprocessing_utils.TripletData import triplet_error_batches
import logging


def landmark_oe_with_data(data: np.ndarray, trip_num: int, dim: int, epochs: int, batch_size: int, learning_rate: float,
                          device: torch.device, logger: logging.Logger):
    """
    given the ground-truth data in a numpy array, it produces the ordinal embedding. The method internally generates
    a triplet answering oracle based on the ground-truth. The maximum number of triplets is given as a parameter.
    :param data: the ground-truth data
    :param trip_num: maximum number of triplet queries
    :param dim: embedding dimension
    :param epochs: number of epoch for the first phase (ranking)
    :param batch_size: used for triplet generation of oracle
    :param learning_rate: the multiplier for the first optimization phase
    :param device: torch device
    :param logger: logger for logging the results
    :return:  triplet containing the embedding output, triplet answers of the queried triplets, training triplet error
    based on the queried triplets
    """
    oc = Oracle(data=data)
    n = data.shape[0]
    k = dim + 3
    c = int(np.ceil(trip_num / (k * n * np.log2(n))))

    logger.info('Producing landmark triplets')
    emb, triplets_compact, elapsed = landmark_oe_torch(oc=oc, n=n, c=c, k=k, dim=dim, batch_size=batch_size,
                                                       epochs=epochs, learning_rate=learning_rate
                                                       , device=device, logger=logger)

    emb = emb.cpu().detach().numpy()
    triplets_compact = triplets_compact.cpu().detach().numpy()
    trip_err = triplet_error_batches(emb, triplets_compact)
    return emb, elapsed, trip_err


def landmark_oe_torch(oc: Oracle, n: int, dim: int, c: int, k: int, epochs: int, batch_size: int, learning_rate: float,
                      logger: logging.Logger, device=None):
    """
    Given the oracle, it asks the required triplets, then it embeds the points
    :param oc: triplet answering oracle
    :param n: number of items
    :param dim: dimension of embedding
    :param c: multiplier for the number of triplets to ask (see the paper)
    :param k: nunber of landmark points
    :param epochs: number of epochs for the ranking optimization
    :param batch_size: used for generating triplets
    :param learning_rate: learning rate of ranking optimization (first phase)
    :param device: torch device
    :param logger: logger for logging the results
    :return:  triplet containing the embedding output, triplet answers of the queried triplets, training triplet error
    based on the queried triplets
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    landmark_indices, triplets_answers, pairs_num = oc.gen_landmark_triplets(n=n, k=k, c=c, device=device,
                                                                             bs=batch_size)
    logger.info('performing LOE')
    start = time.time()
    R = make_R(landmark_indices, triplets_answers, pairs_num, n, epochs, learning_rate, k)
    if (torch.sum(torch.isnan(R)) > 0):
        print('nan values, too large lr!')
        elapsed = time.time() - start
        triplets_compact = torch.zeros(landmark_indices.shape).long()
        triplets_compact[:, 0] = landmark_indices[:, 0]
        triplets_compact[:, 1] = landmark_indices[:, 2] * (triplets_answers + 1) / 2 + \
                                 landmark_indices[:, 1] * (1 - triplets_answers) / 2
        triplets_compact[:, 2] = landmark_indices[:, 1] * (triplets_answers + 1) / 2 + \
                                 landmark_indices[:, 2] * (1 - triplets_answers) / 2
        Z = torch.rand((n, dim))
        return Z, triplets_compact, elapsed
    w_hat = R[:k, :k]
    D, s = infer_dist_mat(w_hat)
    F = R[k:n, 0: k] + s.repeat(n - k, 1)
    X, Y = lmds(D, F, dim)
    Z = torch.cat((X, Y))
    elapsed = time.time() - start
    triplets_compact = torch.zeros(landmark_indices.shape).long()
    triplets_compact[:, 0] = landmark_indices[:, 0]
    triplets_compact[:, 1] = landmark_indices[:, 2] * (triplets_answers + 1) / 2 + \
                             landmark_indices[:, 1] * (1 - triplets_answers) / 2
    triplets_compact[:, 2] = landmark_indices[:, 1] * (triplets_answers + 1) / 2 + \
                             landmark_indices[:, 2] * (1 - triplets_answers) / 2
    return Z, triplets_compact, elapsed


def make_R(landmark_indices: torch.Tensor, triplet_answers: torch.Tensor, pairs_num: int, n: int,
           epochs: int, learning_rate: float, k: int):
    # produces the matrix R (see the paper) that contains ranking of every item with respect to landmark items
    device = landmark_indices.device
    R = torch.zeros(size=(n, k), dtype=torch.float).to(device)

    for pivot in range(k):
        X = landmark_indices[pivot * pairs_num:(pivot + 1) * pairs_num, 1:3]
        y = triplet_answers[pivot * pairs_num:(pivot + 1) * pairs_num]
        w = torch.zeros(size=(n, 1), dtype=torch.float).to(device)
        w.requires_grad = True
        prev_ll = -float("inf")
        for it in range(epochs):
            ll = log_likelihood(w, X, y)  # + torch.sum(w ** 2)
            if (ll < prev_ll):
                print(ll, prev_ll)
                print('stoped at it=', it)
                break
            prev_ll = ll.clone().detach()
            gradient = grad(ll, w)[0]
            with torch.no_grad():
                w += learning_rate * gradient
        R[:, pivot] = w.squeeze()

    return R


def proj_PSD(x: torch.Tensor):
    # Computes the projection of an n x n matrix X onto PSD cone

    x = proj_sym(x)
    (eigenvalues, eigenvectors) = torch.eig(x, eigenvectors=True)
    mid_mat = torch.diag(eigenvalues[:, 0].squeeze())
    mid_mat[mid_mat < 0] = 0
    out = torch.matmul(torch.matmul(eigenvectors, mid_mat), eigenvectors.transpose(1, 0))

    return out


def proj_K2(x: torch.Tensor):
    # Computes the projection of an n x n matrix X onto cone K2
    device = x.device
    n = x.shape[0]
    V = (torch.eye(n) - torch.ones((n, n)) / n).to(device)
    mult = torch.matmul(torch.matmul(V, x.float()), V)
    return x - proj_PSD(mult)


def logistic_f(x: torch.double):
    return 1 / (1 + torch.exp(-x))


def proj_sym(x: torch.Tensor):
    # symmetric projection
    return (x + torch.transpose(x, 1, 0)) / 2


def log_likelihood(w, X, y):
    # log-likelihood for the ranking optimization
    w_expanded = w[X[:, 0]] - w[X[:, 1]]

    vec_1 = torch.log(logistic_f(w_expanded)).squeeze()
    vec_2 = torch.log(1 - logistic_f(w_expanded)).squeeze()

    sum_vector = torch.sum(vec_1 * (y + 1) / 2 + vec_2 * (1 - y) / 2)

    return sum_vector


def projSymHollow(x):
    # symmetric projection with removed diagonal
    return proj_sym(x - torch.diag(torch.diag(x)))


def infer_dist_mat(w_hat: torch.Tensor):
    # given the estimated rankings with respect to the landmarks as w_hat, it produces the partial distance matrix
    device = w_hat.device
    w_diff = torch.transpose(w_hat, 1, 0) - w_hat
    k = w_hat.shape[0]
    A = (pw_diff(k)).to(device)
    b = torch.cat(
        (-w_diff[torch.triu(torch.ones(k, k), 1) == 1], torch.Tensor([-torch.sum(w_hat) / (k - 1)]).to(device)))

    A_inv = torch.pinverse(A)
    s_sol = torch.matmul(A_inv, b)
    J = (torch.ones(size=(k, k)) - torch.eye(k)).to(device)
    s_cap = torch.matmul(J, torch.diag(s_sol))
    C = proj_sym(s_cap + w_hat).to(device)
    eig_norm_2 = torch.sort(torch.norm(torch.eig(C).eigenvalues, dim=1), descending=True).values[1]
    s = s_sol + eig_norm_2 * torch.ones((1, k)).to(device)
    D = proj_EDM(w_hat + torch.matmul(J, torch.diag(s.squeeze())))

    return D, s


def proj_EDM(X: torch.Tensor):
    # projects X into the space of Euclidean distance matrices
    device = X.device
    max_iters = 1000
    proj_tol = 1e-10
    n = X.shape[0]
    P = torch.zeros(size=(n, n)).to(device)
    Y = X
    iter = 0
    converged = False

    while iter < max_iters and (not converged):
        X = projSymHollow(Y)
        Y = proj_K2(X - P)
        P = Y - (X - P)
        conv = torch.max(Y - X) / torch.max(Y)
        converged = (conv <= proj_tol)
        iter = iter + 1

    if not converged:
        print('projEDM did not converge in %d iterations\n', iter)

    return projSymHollow(Y)


def pw_diff(k: int):
    # pair-wise difference matrix (see the paper)
    A = torch.zeros(size=((int(k * (k - 1) / 2)), k))
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            A[count, i] = 1
            A[count, j] = -1
            count = count + 1
    A = torch.cat((A, torch.ones(size=(1, k))))
    return A


def lmds(E, F, d):
    """
    Produces the landmark MDS based on the partial distance matrices
    :param E: k x k matrix of distances between landmarks
    :param F: (n-k) x k matrix of distances of non-landmarks to landmarks
    :param d: embedding dimension
    :return: X is the left part of embedding, Y is the right part
    """
    device = E.device
    nk = F.shape[0]
    k = F.shape[1]

    # Distance matrix to Gram matrix
    V = (torch.eye(k) - torch.ones((k, k)) / k).to(device)
    A = -0.5 * torch.matmul(torch.matmul(V, E), V).to(device)
    B = -0.5 * torch.matmul((F - torch.matmul(torch.ones((nk, k)).to(device), E) / k), V)

    # Embedding
    lam, Q = torch.eig(A, eigenvectors=True)
    lam = torch.norm(lam, dim=1)
    larger_inds = torch.argsort(lam, descending=True)[0:d]

    lam = lam[larger_inds]
    Q = Q[:, larger_inds]
    X = torch.matmul(Q, torch.diag(torch.sqrt(lam)))
    Y = torch.matmul(torch.matmul(B, Q), torch.diag(1 / torch.sqrt(lam)))

    return X, Y
