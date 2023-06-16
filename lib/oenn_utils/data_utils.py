# coding: utf-8
# !/usr/bin/env python
import numpy as np
import math
import sys
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph_shortest_path import graph_shortest_path
from torch.utils.data import DataLoader
import torch


def gen_triplets_from_knn(data, indices, num_neighbors=50):
    """
    Description: Generate triplet data given distance matrix and random indices.
    :param data: #TODO
    :param indices:
    :param num_neighbors:
    :return:
    """
    print('Generating the knn graph')
    sys.stdout.flush()
    kng = kneighbors_graph(data, num_neighbors, mode='distance', n_jobs=8)
    print('Computing the shortest path metric on the knn graph')
    sys.stdout.flush()
    sp_dist_matrix = graph_shortest_path(kng, method='auto', directed=False)
    del kng

    num_triplets = indices.shape[0]  # Compute the number of triplets.
    triplet_set = np.zeros((num_triplets, 3), dtype=int)  # Initializing the triplet set

    triplet_set[:, 0] = indices[:, 0]  # Initialize index 1 randomly.

    d1 = sp_dist_matrix[indices[:, 0], indices[:, 1]]
    d2 = sp_dist_matrix[indices[:, 0], indices[:, 2]]

    det = np.sign(d1 - d2)

    triplet_set[:, 1] = ((indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
    triplet_set[:, 2] = ((indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)
    triplet_set = triplet_set.astype(dtype=int)

    return triplet_set


def gen_triplets_from_knn_in_batches(data, random_triplet_indices, num_neighbors=50, batch_size=10000):
    """
    Description: Generate triplet data given distance matrix and random indices.
    :param data: #TODO
    :param random_triplet_indices:
    :param num_neighbors:
    :param batch_size:
    :return:
    """

    kng = kneighbors_graph(data, num_neighbors, mode='distance', n_jobs=8)
    sp_dist_matrix = graph_shortest_path(kng, method='auto', directed=False)
    del kng

    num_triplets = random_triplet_indices.shape[0]  # Compute the number of triplets.
    number_of_batches = np.int(np.ceil(num_triplets / batch_size))  # Number of batches

    triplet_set = np.zeros((num_triplets, 3), dtype=int)  # Initializing the triplet set

    for i in range(number_of_batches):
        if i == (number_of_batches - 1):
            indices = random_triplet_indices[(i * batch_size):, :]
            triplet_set[(i * batch_size):, 0] = indices[:, 0]

            d1 = sp_dist_matrix[indices[:, 0], indices[:, 1]]
            d2 = sp_dist_matrix[indices[:, 0], indices[:, 2]]

            det = np.sign(d1 - d2)

            triplet_set[(i * batch_size):, 1] = (
                        (indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
            triplet_set[(i * batch_size):, 2] = (
                        (indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)

        else:
            indices = random_triplet_indices[(i * batch_size):((i+1) * batch_size), :]
            triplet_set[(i * batch_size):((i+1) * batch_size), 0] = indices[:, 0]

            d1 = sp_dist_matrix[indices[:, 0], indices[:, 1]]
            d2 = sp_dist_matrix[indices[:, 0], indices[:, 2]]

            det = np.sign(d1 - d2)

            triplet_set[(i * batch_size):((i+1) * batch_size), 1] = ((indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
            triplet_set[(i * batch_size):((i+1) * batch_size), 2] = ((indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)
    triplet_set = triplet_set.astype(dtype=int)

    triplet_set[:, 0] = random_triplet_indices[:, 0]  # Initialize index 1 randomly.

    return triplet_set


def triplet_error(emb, trips):
    """
    Description: Given the embeddings and triplet constraints, compute the triplet error.
    :param emb:
    :param trips:
    :return:
    """
    d1 = np.sum((emb[trips[:, 0], :] - emb[trips[:, 1], :])**2, axis=1)
    d2 = np.sum((emb[trips[:, 0], :] - emb[trips[:, 2], :])**2, axis=1)
    error_list = d2 < d1
    ratio = sum(error_list) / trips.shape[0]
    return ratio, error_list


def triplet_error_unseen(test_emb, train_emb, unseen_trips):
    """
    Description: Given the embeddings and triplet constraints, compute the triplet error.
    :param test_emb:
    :param train_emb:
    :param unseen_trips
    :return:
    """
    d1 = np.sum((test_emb[unseen_trips[:, 0], :] - train_emb[unseen_trips[:, 1], :])**2, axis=1)
    d2 = np.sum((test_emb[unseen_trips[:, 0], :] - train_emb[unseen_trips[:, 2], :])**2, axis=1)
    error_list = d2 < d1
    ratio = sum(error_list) / unseen_trips.shape[0]
    return ratio, error_list


def binary_vec(num, length):
    """"
    Description:
    Takes a natural number (n) and a specified length (l) and returns the (l-dim) binary representation of n.
    """
    bin_val = bin(num)
    val = list(bin_val[2:])
    while len(val) < length:
        val = [0] + val
    return val


def get_binary_array(n, digits):
    """"
    Description:
    Takes a number (n) and a length (l) and returns the (l-dim) binary representations of all numbers 1 to n.
    """
    bin_array = np.zeros((n, digits), dtype=int)

    for ind in range(n):
        bin_array[ind, :] = np.array(binary_vec(ind, digits), dtype=int)
    return bin_array


def gen_triplet_indices(n, num_trips):
    """
    Description: Generate random triplet indices
    :param n: #points in the data
    :param num_trips: #triplets
    :return: random triplet indices. Shape (#triplets, 3)
    """
    all_triplet_indices = np.random.randint(n, size=(num_trips, 3))

    return all_triplet_indices


def gen_triplet_data_unseen(data, unseen_data, num_triplets):
    """
    Description: Random triplets containing data points from the unseen data.
    # TODO: There is redundancy in the code, remove it.
    :param data:
    :param unseen_data:
    :param num_triplets:
    :return:
    """
    indices = np.random.randint(data.shape[0], size=(num_triplets, 3))  # Compute random
    triplet_set = indices.copy()
    triplet_set[:, 0] = np.random.randint(unseen_data.shape[0], size=(num_triplets, ))

    d1 = np.sum((unseen_data[triplet_set[:, 0], :] - data[indices[:, 1], :]) ** 2, axis=1)
    d2 = np.sum((unseen_data[triplet_set[:, 0], :] - data[indices[:, 2], :]) ** 2, axis=1)

    det = np.sign(d1 - d2)

    triplet_set[:, 1] = (((indices[:, 1] + indices[:, 2]) - (np.multiply(det, indices[:, 1])) + (np.multiply(det, indices[:,2]))) / 2)
    triplet_set[:, 2] = (((indices[:, 1] + indices[:, 2]) + (np.multiply(det, indices[:, 1])) - (np.multiply(det, indices[:,2]))) / 2)

    triplet_set = triplet_set.astype(dtype=int)
    return triplet_set


def gen_triplet_data_cosine(data, indices):
    """
    Description: Generate triplets based on cosine similarity.
    :param data:
    :param indices:
    :return:
    """
    num_triplets = indices.shape[0]  # Compute the number of triplets.
    triplet_set = np.zeros((num_triplets, 3), dtype=int)  # Initializing the triplet set

    triplet_set[:, 0] = indices[:, 0]  # Initialize index 1 randomly.

    norm_indices_12 = norm(data[indices[:, 0], :], axis=1) * norm(data[indices[:, 1], :], axis=1)
    norm_indices_13 = norm(data[indices[:, 0], :], axis=1) * norm(data[indices[:, 2], :], axis=1)

    d1 = ((data[indices[:, 0], :] * data[indices[:, 1], :]).sum(axis=1))/norm_indices_12
    d2 = (data[indices[:, 0], :] * data[indices[:, 2], :]).sum(axis=1)/norm_indices_13

    det = np.sign(d1 - d2)
    #  Cosine is a similarity, don't forget to reverse the order if using the code for distance!
    triplet_set[:, 1] = ((indices[:, 1] + indices[:, 2] + det * indices[:, 1] + det * indices[:, 2]) / 2)
    triplet_set[:, 2] = ((indices[:, 1] + indices[:, 2] - det * indices[:, 1] - det * indices[:, 2]) / 2)
    triplet_set = triplet_set.astype(dtype=int)

    return triplet_set


def gen_triplet_data_torch(data, num_triplets, batch_size):
    """
    Description: Generate triplets at once in batches so we won't run into memory issues and training takes less time.
    # TODO: There is redundancy in the code, remove it.
    :param data: 2D numpy array with shape (#points, #dimensions)
    :param random_triplet_indices: A set of triplets with random integers in [#points]. Shape (#triplets, 3)
    :param batch_size: Arbitrary integer, typically chosen as 10000 or 50000
    :return: triplet_set: Triplet data. Shape (#triplets, 3)
    """
    number_of_batches = int(np.ceil(num_triplets/batch_size))
    m = number_of_batches * batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # num_triplets = random_triplet_indices.shape[0]  # Compute the number of triplets.
    triplet_set = torch.Tensor(size=(m, 3)).long().to(device)  # Initializing the triplet set
    # random_triplet_indices = torch.Tensor(random_triplet_indices).long().to(device)
    data = torch.Tensor(data).to(device)
    n = data.shape[0]

    for i in range(number_of_batches):

        indices = torch.randint(0, n, size=(batch_size, 3)).long().to(device)

        triplet_set[(i * batch_size):((i+1) * batch_size), 0] = indices[:, 0]

        d1 = torch.sum((data[indices[:, 0], :] - data[indices[:, 1], :]) ** 2, dim=1)
        d2 = torch.sum((data[indices[:, 0], :] - data[indices[:, 2], :]) ** 2, dim=1)

        # print(d1)
        det = torch.sign(d1 - d2)

        triplet_set[(i * batch_size):((i+1) * batch_size), 1] = ((indices[:, 1] + indices[:, 2] - det * indices[
                                                                                                        :, 1] + det * indices[:, 2]) / 2)
        triplet_set[(i * batch_size):((i+1) * batch_size), 2] = ((indices[:, 1] + indices[:, 2] + det * indices[
                                                                                                        :, 1] - det * indices[:, 2]) / 2)
    triplet_set_cpu = triplet_set.cpu().numpy()
    triplet_set_cpu = triplet_set_cpu.astype(dtype=int)
    del triplet_set
    return triplet_set_cpu


# def gen_triplet_data_torch(data, random_triplet_indices, batch_size):
#     """
#     Description: Generate triplets at once in batches so we won't run into memory issues and training takes less time.
#     # TODO: There is redundancy in the code, remove it.
#     :param data: 2D numpy array with shape (#points, #dimensions)
#     :param random_triplet_indices: A set of triplets with random integers in [#points]. Shape (#triplets, 3)
#     :param batch_size: Arbitrary integer, typically chosen as 10000 or 50000
#     :return: triplet_set: Triplet data. Shape (#triplets, 3)
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     num_triplets = random_triplet_indices.shape[0]  # Compute the number of triplets.
#     number_of_batches = np.int(np.ceil(num_triplets/batch_size))  # Number of batches
#     # triplet_set = torch.zeros(size=(num_triplets, 3)).to(device).long()  # Initializing the triplet set
#     triplet_set = torch.Tensor(random_triplet_indices).to(device).long()
#     data = torch.Tensor(data).to(device)
#     for i in range(number_of_batches):
#         if i == (number_of_batches - 1):
#             indices = triplet_set[(i * batch_size):, :]
#             # triplet_set[(i * batch_size):, 0] = indices[:, 0]
#
#             d1 = torch.sum((data[indices[:, 0], :] - data[indices[:, 1], :]) ** 2, dim=1)
#             d2 = torch.sum((data[indices[:, 0], :] - data[indices[:, 2], :]) ** 2, dim=1)
#
#             det = torch.sign(d1 - d2)
#
#             triplet_set[(i * batch_size):, 1] = (
#                         (indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) /2)
#             triplet_set[(i * batch_size):, 2] = (
#                         (indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)
#
#         else:
#             indices = triplet_set[(i * batch_size):((i+1) * batch_size), :]
#             # triplet_set[(i * batch_size):((i+1) * batch_size), 0] = indices[:, 0]
#
#             d1 = torch.sum((data[indices[:, 0], :] - data[indices[:, 1], :]) ** 2, dim=1)
#             d2 = torch.sum((data[indices[:, 0], :] - data[indices[:, 2], :]) ** 2, dim=1)
#
#             det = torch.sign(d1 - d2)
#
#             triplet_set[(i * batch_size):((i+1) * batch_size), 1] = ((indices[:, 1] + indices[:, 2] - det * indices[
#                                                                                                             :, 1] + det * indices[:, 2]) / 2)
#             triplet_set[(i * batch_size):((i+1) * batch_size), 2] = ((indices[:, 1] + indices[:, 2] + det * indices[
#                                                                                                             :, 1] - det * indices[:, 2]) / 2)
#     triplet_set_cpu = triplet_set.cpu().numpy()
#     del triplet_set
#     triplet_set_cpu = triplet_set_cpu.astype(dtype=int)
#
#     return triplet_set_cpu


def gen_triplet_data(data, random_triplet_indices, batch_size):
    """
    Description: Generate triplets at once in batches so we won't run into memory issues and training takes less time.
    # TODO: There is redundancy in the code, remove it.
    :param data: 2D numpy array with shape (#points, #dimensions)
    :param random_triplet_indices: A set of triplets with random integers in [#points]. Shape (#triplets, 3)
    :param batch_size: Arbitrary integer, typically chosen as 10000 or 50000
    :return: triplet_set: Triplet data. Shape (#triplets, 3)
    """
    num_triplets = random_triplet_indices.shape[0]  # Compute the number of triplets.
    number_of_batches = np.int(np.ceil(num_triplets/batch_size))  # Number of batches
    triplet_set = np.zeros((num_triplets, 3), dtype=int)  # Initializing the triplet set
    for i in range(number_of_batches):
        if i == (number_of_batches - 1):
            indices = random_triplet_indices[(i * batch_size):, :]
            triplet_set[(i * batch_size):, 0] = indices[:, 0]

            d1 = np.sum((data[indices[:, 0], :] - data[indices[:, 1], :]) ** 2, axis=1)
            d2 = np.sum((data[indices[:, 0], :] - data[indices[:, 2], :]) ** 2, axis=1)

            det = np.sign(d1 - d2)

            triplet_set[(i * batch_size):, 1] = (
                        (indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
            triplet_set[(i * batch_size):, 2] = (
                        (indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)

        else:
            indices = random_triplet_indices[(i * batch_size):((i+1) * batch_size), :]
            triplet_set[(i * batch_size):((i+1) * batch_size), 0] = indices[:, 0]

            d1 = np.sum((data[indices[:, 0], :] - data[indices[:, 1], :]) ** 2, axis=1)
            d2 = np.sum((data[indices[:, 0], :] - data[indices[:, 2], :]) ** 2, axis=1)

            det = np.sign(d1 - d2)

            triplet_set[(i * batch_size):((i+1) * batch_size), 1] = ((indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
            triplet_set[(i * batch_size):((i+1) * batch_size), 2] = ((indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)
    triplet_set = triplet_set.astype(dtype=int)

    return triplet_set


def gen_triplet_set_in_batches(data, batch_triplet_indices):
    """
    Description: #TODO
    :param data:
    :param batch_triplet_indices:
    :return: triplet_set
    """
    # Number of triplets we will generate --> the number in each batch
    num_triplets = batch_triplet_indices.shape[0]
    triplet_set = np.zeros((num_triplets, 3), dtype=int)  # Initializing the triplet set
    indices = batch_triplet_indices

    d1 = np.sum((data[indices[:, 0], :] - data[indices[:, 1], :]) ** 2, axis=1)
    d2 = np.sum((data[indices[:, 0], :] - data[indices[:, 2], :]) ** 2, axis=1)

    triplet_set[:, 0] = indices[:, 0]

    det = np.sign(d1 - d2)
    triplet_set[:, 1] = ((indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
    triplet_set[:, 2] = ((indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)
    triplet_set = triplet_set.astype(dtype=int)

    return triplet_set


def get_nearest_neighbors(data, num_neighbors):
    """
    Description: Generate nearest neighbor indices for the whole data set.
    :param data:
    :param num_neighbors:
    :return:
    """
    neighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=8).fit(data)
    neighbor_indices = np.zeros((data.shape[0], num_neighbors))
    # neighbor_indices = neighbors.kneighbors(data, return_distance=False)
    batch_size = 10000
    number_of_batches = np.int(np.ceil(data.shape[0]/batch_size))
    for i in range(0, number_of_batches):
        if i == (number_of_batches - 1):
            print('Batch number ' + str(i))
            neighbor_indices[(i*batch_size):, :] = neighbors.kneighbors(data[(i*batch_size):, :],
                                                                        return_distance=False)
        else:
            print('Batch number ' + str(i))
            neighbor_indices[(i * batch_size):((i+1) * batch_size), :] = neighbors.kneighbors(data[(i * batch_size):(
                    (i+1) * batch_size), :], return_distance=False)

    return neighbor_indices


def prep_data_for_nn(vec_data, labels, triplet_num, batch_size, metric, number_of_neighbours):
    # If sampling is random
    print('Sampling is chosen as random')
    sys.stdout.flush()

    num_neighbors = number_of_neighbours

    print('Generating triplets...')
    sys.stdout.flush()

    triplet_indices = TripletDataset(data=vec_data, labels=labels,
                                        num_trips=triplet_num,
                                        batch_size=batch_size,
                                        metric=metric,
                                        num_n=num_neighbors)

    print('Generating triplets completed.')
    sys.stdout.flush()

    all_triplets = triplet_indices.trips_data  # For computing triplet error

    print('Data loader is being created')
    sys.stdout.flush()

    batch_triplet_indices_loader = DataLoader(triplet_indices, batch_size=batch_size, shuffle=True,
                                              num_workers=8)
    triplet_loaders = {'random': batch_triplet_indices_loader}

    print('Data loader creation is complete.')
    sys.stdout.flush()

    return all_triplets, triplet_loaders


def gen_selective_triplet_indices(num_trips, neighbor_indices):
    """
    Description: Generate triplets according to nearest neighbor sampling strategy
    :param num_trips:
    :param neighbor_indices:
    :return:
    """
    print('Generating samples')
    n = neighbor_indices.shape[0]  # Total number of points
    num_neighbors = neighbor_indices.shape[1]  # Number of neighbors selected for each point.

    all_triplet_indices = np.zeros((num_trips, 3))  # Initializing the triplets
    # The first index is randomly sampled from all the points
    indices_one = np.random.randint(n, size=(num_trips, 1)).squeeze()

    # The second index is randomly sampled from the nearest neighbor of index 1.
    nearest_neighbors = neighbor_indices[indices_one[:], :]  # Arranges the nearest neighbors according to

    random_nn_indices = np.random.randint(1, num_neighbors, size=(num_trips, 1)).squeeze()  # Exclude the first point.
    all_triplet_indices[:, 0] = indices_one
    all_triplet_indices[:, 1] = nearest_neighbors[np.arange(len(nearest_neighbors)), random_nn_indices]
    all_triplet_indices[:, 2] = np.random.randint(n, size=(num_trips, 1)).squeeze()
    print('')
    return all_triplet_indices.astype(int)


class TripletDataset(Dataset):
    """"
    Description: #TODO
    """
    def __init__(self, data, labels, num_trips, batch_size, metric, num_n, test=False, test_data=np.array([])):
        super(TripletDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.n = data.shape[0]
        self.digits = int(math.ceil(math.log2(self.n)))
        print('print digits inside trip loader is ' + str(self.digits))
        sys.stdout.flush()
        self.bin_array = get_binary_array(self.n, self.digits)

        if test:
            self.trips_data = gen_triplet_data_unseen(data, test_data, num_trips)
        else:
            indices = gen_triplet_indices(self.n, num_trips)
            print('Metric chosen is ' + metric)
            sys.stdout.flush()

            self.trips_data = gen_triplet_data(self.data, indices, batch_size)

    def __getitem__(self, index):
        return np.reshape(self.bin_array[self.trips_data[index, ]], [1, -1])

    def __len__(self):
        return self.trips_data.shape[0]

    def triplet_error(self, emb):
        d1 = np.sum((emb[self.trips_data[:, 0], :] - emb[self.trips_data[:, 1], :]) ** 2, axis=1)
        d2 = np.sum((emb[self.trips_data[:, 0], :] - emb[self.trips_data[:, 2], :]) ** 2, axis=1)
        error_list = d2 <= d1
        ratio = sum(error_list) / self.trips_data.shape[0]
        return ratio


class SelectiveTripletDataset(Dataset):
    """"
    Description: #TODO
    """
    def __init__(self, data, labels, num_trips, batch_size, neighbor_indices, metric, num_n):
        super(SelectiveTripletDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.n = data.shape[0]
        self.digits = int(math.ceil(math.log2(self.n)))
        self.bin_array = get_binary_array(self.n, self.digits)
        self.neighbor_indices = neighbor_indices

        indices = gen_selective_triplet_indices(num_trips, self.neighbor_indices)
        if metric == 'cosine':
            self.trips_data = gen_triplet_data_cosine(self.data, indices)
        if metric == 'knn':
            self.trips_data = gen_triplets_from_knn(self.data, indices, num_neighbors=num_n)
        else:
            self.trips_data = gen_triplet_data(self.data, indices, batch_size)

    def __getitem__(self, index):
        return np.reshape(self.bin_array[self.trips_data[index, ]], [1, -1])

    def __len__(self):
        return self.trips_data.shape[0]

    def triplet_error(self, emb):
        d1 = np.sum((emb[self.trips_data[:, 0], :] - emb[self.trips_data[:, 1], :]) ** 2, axis=1)
        d2 = np.sum((emb[self.trips_data[:, 0], :] - emb[self.trips_data[:, 2], :]) ** 2, axis=1)
        error_list = d2 < d1
        ratio = sum(error_list) / self.trips_data.shape[0]
        return ratio

