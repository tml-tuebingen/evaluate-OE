# coding: utf-8
# !/usr/bin/env python

import numpy as np
from sklearn import neighbors
from sklearn.utils.graph_shortest_path import graph_shortest_path


def gen_knn_graph_with_sp(data, num_neighbors):
    """
    Description: Generate knn graph with the shortest path distance given data, #neighbors.
    :param data: #TODO
    :param num_neighbors:
    :return:
    """
    kng = neighbors.kneighbors_graph(data, num_neighbors, mode='distance', n_jobs=8)
    sp_dist_matrix = graph_shortest_path(kng, method='auto', directed=False)
    return sp_dist_matrix


def gen_triplets_from_distance_mat(sp_dist_matrix, indices):
    """
    Description: Generate triplet data given distance matrix and random indices.
    :param sp_dist_matrix: #TODO
    :param indices:
    :return:
    """
    num_triplets = indices.shape[0]  # Compute the number of triplets.
    triplet_set = np.zeros((num_triplets, 3), dtype=int)  # Initializing the triplet set

    triplet_set[:, 0] = indices[:, 0]  # Initialize index 1 randomly.

    d1 = sp_dist_matrix[indices[:, 0], :][:, indices[:, 1]]
    d2 = sp_dist_matrix[indices[:, 0], :][:, indices[:, 2]]

    det = np.sign(d1 - d2)

    triplet_set[:, 1] = ((indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
    triplet_set[:, 2] = ((indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)
    triplet_set = triplet_set.astype(dtype=int)

    return triplet_set
