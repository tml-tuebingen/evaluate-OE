# coding: utf-8
# !/usr/bin/env python
import numpy as np
import math
from torch.utils.data import Dataset
from sklearn.datasets import make_blobs
from scipy.spatial import procrustes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV as grid_search
from sklearn.model_selection import train_test_split
import torch


def fft(distance_mat, k, initial_anchor_index):
    """
    D: distance matrix (n x n)
    k: number of points to choose
    out: indices of centroids
    """
    # distance_mat = torch.Tensor(distance_mat)
    # num_samples = distance_mat.shape[0]
    # visited = []
    c = initial_anchor_index
    visited = np.zeros((k), dtype=int)
    visited[0] = c

    for j in range(1,k):
        dist = np.min(distance_mat[visited[0:j]], axis=0)
        # print(dist.values.shape)
        # print(dist.shape, len(visited))
        a = np.argmax(dist)
        # print(len(visited), 'anchors found so far ...')
        # print(np.sort(dist)[::-1])
        # print(a)
        # print(visited)
        visited[j] = a
        # for l in a:
        #     if l not in visited:
        #         visited.append(l)
        #         break
        # j += 1
    return visited


def triplet_error(emb, trips):
    """
    Description: Given the embeddings and triplet constraints, compute the triplet error.
    :param emb:
    :param trips:
    :return:
    """
    d1 = np.sum((emb[trips[:, 0], :] - emb[trips[:, 1], :]) ** 2, axis=1)
    d2 = np.sum((emb[trips[:, 0], :] - emb[trips[:, 2], :]) ** 2, axis=1)
    error_list = d2 < d1
    ratio = sum(error_list) / trips.shape[0]
    return ratio, error_list

def triplet_error_batches(emb, trips):
    batch_size = 1000000
    triplet_num = trips.shape[0]
    batches = 1 if batch_size > triplet_num else triplet_num // batch_size
    train_triplet_error = 0
    for batch_ind in range(batches):
        batch_trips = trips[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
        batch_triplet_error, _ = triplet_error(emb, batch_trips)
        train_triplet_error += batch_triplet_error
    train_triplet_error = train_triplet_error / batches

    return train_triplet_error

def triplet_error_torch(emb, trips):
    """
    Description: Given the embeddings and triplet constraints, compute the triplet error.
    :param emb:
    :param trips:
    :return:
    """
    d1 = torch.sum((emb[trips[:, 0], :] - emb[trips[:, 1], :]) ** 2, dim=1)
    d2 = torch.sum((emb[trips[:, 0], :] - emb[trips[:, 2], :]) ** 2, dim=1)

    error_list = d2 < d1
    ratio = sum(error_list) / float(trips.shape[0])
    return ratio, error_list


def procrustes_disparity(emb_true, ord_emb):
    """
    Description: Compute the procrustes disparity using translation, scaling, reflection, and rotation.
    It computes the sum of the pointwise squared differences. If the dimension of ord_emb is lower then
    of emb_true, then zero-columns are added to the ord_emb, and vice-versa.
    :param emb_true: Ground-truth embedding
    :param ord_emb: result of an ordinal embedding approach with lower or equal dimension
    :return: disparity value
    """
    # add zero columns to the embedding with smaller dimension
    if emb_true.shape[1] == ord_emb.shape[1]:
        _, _, disparity = procrustes(emb_true, ord_emb)
    elif emb_true.shape[1] > ord_emb.shape[1]:
        nmb_new_cols = emb_true.shape[1] - ord_emb.shape[1]
        new_cols = np.zeros(shape=(ord_emb.shape[0],nmb_new_cols))
        ord_emb = np.c_[ord_emb, new_cols]
        _, _, disparity = procrustes(emb_true, ord_emb)
    else:
        nmb_new_cols = ord_emb.shape[1] - emb_true.shape[1]
        new_cols = np.zeros(shape=(emb_true.shape[0], nmb_new_cols))
        emb_true = np.c_[emb_true, new_cols]
        _, _, disparity = procrustes(emb_true, ord_emb)
    return disparity


def knn_classification_error(ord_emb, true_emb, labels):
    """
    Description: Compute the kNN classification error on a test set (70/30 split)  trained on
    the ground-truth embedding and the ordinal embedding.
    :param ord_emb: ordinal embedding
    :param true_emb: ground-truth embedding
    :param labels: labels of the data points
    :return classification error on test data on ordinal embedding and ground-truth embedding.
    """
    n_neighbors = int(np.log2(ord_emb.shape[0]))

    x_ord_train, x_ord_test, x_train, x_test, y_train, y_test = train_test_split(ord_emb, true_emb, labels, train_size=0.7)
    # n_neighbors = {'n_neighbors':[1, 3, 5, 10, 15, 20, 25]}
    ordinal_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    original_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    # ordinal_classifier = grid_search(kNN(), param_grid=n_neighbors, cv=3)
    # original_classifier = grid_search(kNN(), param_grid=n_neighbors, cv=3)
    ordinal_classifier.fit(x_ord_train, y_train)
    original_classifier.fit(x_train, y_train)
    return 1 - ordinal_classifier.score(x_ord_test, y_test), 1 - original_classifier.score(x_test, y_test)

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
    Description #TODO
    :param n:
    :param num_trips:
    :return:
    """
    all_triplet_indices = np.random.randint(n, size=(num_trips, 3))

    # bad_indices = np.logical_or(all_triplet_indices[:, 0] == all_triplet_indices[:, 1],
    #                             all_triplet_indices[:, 1] == all_triplet_indices[:, 2],
    #                             all_triplet_indices[:, 0] == all_triplet_indices[:, 2])
    #
    # all_triplet_indices = np.delete(all_triplet_indices, np.where(bad_indices), axis=0)

    all_triplet_indices = all_triplet_indices[:num_trips, ]
    return all_triplet_indices


def gen_triplet_data(data, random_triplet_indices, batch_size):
    """
    Description: Generate triplets at once in batches so we won't run into memory issues and training takes less time.
    # TODO: There is redundancy in the code, remove it.
    :param data:
    :param random_triplet_indices:
    :param batch_size:
    :return:
    """
    num_triplets = random_triplet_indices.shape[0]  # Compute the number of triplets.
    number_of_batches = np.int(np.ceil(num_triplets / batch_size))  # Number of batches
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
            indices = random_triplet_indices[(i * batch_size):((i + 1) * batch_size), :]
            triplet_set[(i * batch_size):((i + 1) * batch_size), 0] = indices[:, 0]

            d1 = np.sum((data[indices[:, 0], :] - data[indices[:, 1], :]) ** 2, axis=1)
            d2 = np.sum((data[indices[:, 0], :] - data[indices[:, 2], :]) ** 2, axis=1)

            det = np.sign(d1 - d2)

            triplet_set[(i * batch_size):((i + 1) * batch_size), 1] = (
                    (indices[:, 1] + indices[:, 2] - det * indices[:, 1] + det * indices[:, 2]) / 2)
            triplet_set[(i * batch_size):((i + 1) * batch_size), 2] = (
                    (indices[:, 1] + indices[:, 2] + det * indices[:, 1] - det * indices[:, 2]) / 2)
    triplet_set = triplet_set.astype(dtype=int)
    return triplet_set


class TripletBatchesDataset(Dataset):
    """"
    Description: #TODO
    """

    def __init__(self, data, labels, num_trips, batch_size, device):
        super(TripletBatchesDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.n = data.shape[0]
        self.digits = int(math.ceil(math.log2(self.n)))
        #self.bin_array = torch.Tensor(get_binary_array(self.n, self.digits)).to(device)
        indices = gen_triplet_indices(self.n, num_trips)
        self.trips_data_indices = gen_triplet_data(self.data, indices, batch_size)

        # self.trips_data_digits = self.bin_array[self.trips_data_indices, :].view((num_trips, -1))
        self.num_batches = num_trips // batch_size
        # print(self.trips_data_digits.shape)

    def __getitem__(self, index):
        selected_indices = self.trips_data_indices[index * self.batch_size: (index + 1) * self.batch_size, :]

        return self.trips_data_digits[selected_indices, :]

    def __len__(self):
        return self.num_batches

    def triplet_error(self, emb, sub_sample=True):
        total_triplets = self.trips_data_indices.shape[0]

        #if sub_sample:
        #    samples = np.random.permutation(total_triplets)[0:5000]
        #else:
        #    samples = np.random.permutation(total_triplets)

        d1 = np.sum((emb[self.trips_data_indices[:, 0], :] - emb[self.trips_data_indices[:, 1], :]) ** 2,
                    axis=1)
        d2 = np.sum((emb[self.trips_data_indices[:, 0], :] - emb[self.trips_data_indices[:, 2], :]) ** 2,
                    axis=1)
        error_list = d2 < d1

        ratio = sum(error_list) / self.trips_data_indices.shape[0]
        return ratio


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n = 10
    # data, labels = make_moons(n_samples=n, shuffle=True, noise=0.1)
    dim = 2
    epochs = 500
    bs = 1000
    lr_range = [50000]

    logn = int(np.log2(n))

    # bs_range = [500000, 2 * logn * n * dim]
    data, labels = make_blobs(n_samples=n, n_features=dim, centers=3)

    triplet_batches_dataset = TripletBatchesDataset(data, labels, 2 * logn * n * dim, bs, device)

    print(triplet_batches_dataset.trips_data_indices.shape)
