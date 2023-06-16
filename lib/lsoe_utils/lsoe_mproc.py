import numpy as np
from scipy.spatial import distance_matrix
import random
import torch
from torch.autograd import grad
from preprocessing_utils.TripletData import fft, gen_triplet_data, triplet_error, gen_triplet_indices
from lib.soe import soe_sgd, soe_adam
import math
import time
from torch import optim
# import torch.multiprocessing as mp
import multiprocessing as mp
import sys


def get_subset_sizes(dataset_size, init_subset_size):
    """
    Desc: Get an list of subset sizes starting from an initial subset size via a doubling strategy. See Pseudocode
    for more.
    :param dataset_size: num_samples
    :param init_subset_size: according to algorithm: = 50 * L (num_landmarks)
    :return: List of subset sizes to iterate over
    """
    embedding_size_intervals = []  # List of subset sizes.
    # always use min(dataset/2, 1000) as the max subset size as reported in the paper
    init_subset_size = min(int(dataset_size / 2), init_subset_size, 1000)
    embedding_size_intervals.append(init_subset_size)

    # make multiple subset sizes required for Phase I
    while init_subset_size <= min(int(dataset_size / 2), 1000):
        init_subset_size *= 2
        embedding_size_intervals.append(init_subset_size) if init_subset_size <= dataset_size / 2 else None
    return embedding_size_intervals


def get_landmark_subset(data, number_of_landmarks):
    # get the landmarks subset randomly

    indices = np.random.choice(data.shape[0], size=number_of_landmarks, replace=False)
    # random_complement = list(set(list(range(data.shape[0]))).difference(set(indices)))
    landmarks_matrix = data[indices, :]

    return landmarks_matrix, indices


def get_subsets(data, landmark_indices, subset_size):
    # get the landmarks subset randomly and return random complement as well
    # landmarks
    indices = landmark_indices

    # all other indices
    random_complement = list(set(list(range(data.shape[0]))).difference(set(indices)))

    # select subset from the rest of the dataset
    subset_indices = random.sample(random_complement, k=subset_size - len(indices))

    # add in the landmark as well. The first L indices will be landmarks and the remaining are the rest of the subset.
    final_subset_indices = indices.tolist() + subset_indices

    subset_items = data[final_subset_indices, :]
    return subset_items, final_subset_indices


def compute_landmark_to_data_distance(landmarks, subset):
    distance_mat = distance_matrix(landmarks, subset)
    return distance_mat


def sort_distances_pt_to_landmark(distance_mat):
    """
    Description: Sort distances points to landmarks
    :param distance_mat:
    :return:
    """
    sorting_indices = np.argsort(distance_mat, axis=0)
    sorted_distances = np.sort(distance_mat, axis=0)
    # subset_indices = np.tile(subset_indices, (distance_mat.shape[0], 1))
    # print(subset_indices)
    # print(sorting_indices)
    # sorted_indices = subset_indices[np.arange(subset_indices.shape[0])[:, np.newaxis], sorting_indices]
    return sorting_indices, sorted_distances


def sort_landmark_to_pt_distances(distance_mat):
    """
        Description: Sort distances landmarks to points
        :param distance_mat:
        :return:
        """
    sorting_indices = np.argsort(distance_mat, axis=1)
    sorted_distances = np.sort(distance_mat, axis=1)
    # subset_indices = np.tile(subset_indices, (distance_mat.shape[0], 1))
    # print(subset_indices)
    # print(sorting_indices)
    # sorted_indices = subset_indices[np.arange(subset_indices.shape[0])[:, np.newaxis], sorting_indices]
    return sorting_indices, sorted_distances


def generate_triplets_from_indices(sorted_landmarks_to_subset, sorted_indices_subset_to_landmarks, landmark_indices):
    triplets = []
    indices_mat1 = sorted_landmarks_to_subset
    indices_mat2 = sorted_indices_subset_to_landmarks
    for each_landmark in range(indices_mat1.shape[0]):
        for less_than_index in range(1, indices_mat1.shape[1] - 1):
            # for greater_than_index in range(0, indices_mat.shape[1]-less_than_index-1):
            triplets.append([indices_mat1[each_landmark, 0], indices_mat1[each_landmark, less_than_index],
                             indices_mat1[each_landmark, less_than_index + 1]])
    for each_pt in range(indices_mat2.shape[1]):
        for less_index in range(indices_mat2.shape[0] - 1):
            triplets.append([each_pt, indices_mat2[less_index, each_pt],
                             indices_mat2[less_index + 1, each_pt]]) if each_pt != indices_mat2[
                less_index, each_pt] else None
    return triplets


def first_phase_soe(num_landmarks, subset_size, data, dataset_size, embedding_dim,
                    epochs, target_loss, first_phase_lr, batch_size, device, logger):
    # get embedding size interval
    embedding_size_intervals = get_subset_sizes(dataset_size, subset_size)
    print('Embedding size intervals: ', embedding_size_intervals)
    # get landmarks
    landmarks, landmark_indices = get_landmark_subset(data, number_of_landmarks=num_landmarks)

    # first phase data buffer
    interval_final_epoch_losses = []
    interval_data_subsets = []
    interval_subset_reconstructions = []
    interval_subset_indices = []
    interval_triplet_error = []
    i = 0
    total_time = 0

    # first phase flow
    logger.info('Running SOE on different subset Size Until the Criteria Defined is Satisified.')
    for index, each_subset_size in enumerate(embedding_size_intervals):
        # Get the subset data. Subsets must contain landmarks. The function takes care of this. Subset indices are
        # from the original set of indices
        subset_data, subset_indices = get_subsets(data, landmark_indices=landmark_indices, subset_size=each_subset_size)
        logger.info('First Phase Iteration Index: ' + str(index))
        logger.info('Landmarks:' + str(landmarks.shape))
        logger.info('Obtained subset size:' + str(subset_data.shape))
        logger.info('Landmark indices: ' + str(len(landmark_indices)))
        logger.info('Obtained subset indices: ' + str(len(subset_indices)))

        num_trips = np.int(2 * each_subset_size * math.log2(each_subset_size) * embedding_dim)
        # print(each_subset_size)
        # print(num_trips)
        trip_indices = np.asarray(gen_triplet_indices(each_subset_size, num_trips))  # Random relative triplets

        trips = gen_triplet_data(subset_data, trip_indices, batch_size=trip_indices.shape[0])
        # print(trips)
        # run soe on the subset data
        embedding_of_subset, loss_history, triplet_error_history, time_taken, _ = soe_adam(triplets=trips,
                                                                                           n=each_subset_size,
                                                                                           dim=embedding_dim,
                                                                                           epochs=epochs,
                                                                                           batch_size=min(
                                                                                               trips.shape[0],
                                                                                               batch_size * (2 ** i)),
                                                                                           learning_rate=first_phase_lr * (
                                                                                                       2 ** i),
                                                                                           device=device, logger=logger,
                                                                                           error_change_threshold=-1)
        total_time += time_taken

        final_epoch_loss = loss_history[-1]
        #  print(embedding_of_subset)
        trip_error, error_list = triplet_error(embedding_of_subset, trips)
        logger.info('Number of Triplets: ' + str(trips.shape[0]))
        logger.info('Training triplet Error: ' + str(trip_error))
        logger.info('Subset Size: ' + str(each_subset_size) + ' Loss: ' + str(final_epoch_loss))
        begin_time = time.time()
        if final_epoch_loss < target_loss:
            i += 1
            interval_final_epoch_losses.append(final_epoch_loss)
            interval_data_subsets.append(subset_data)
            interval_subset_reconstructions.append(embedding_of_subset)
            interval_subset_indices.append(subset_indices)
            interval_triplet_error.append(trip_error)
            # print('Final epoch loss was less than target. Continue doubling the subset')
            end_time = time.time()
            total_time += (end_time - begin_time)
            continue
        else:
            if len(interval_final_epoch_losses) < 1:
                interval_final_epoch_losses.append(final_epoch_loss)
                interval_data_subsets.append(subset_data)
                interval_subset_reconstructions.append(embedding_of_subset)
                interval_subset_indices.append(subset_indices)
                interval_triplet_error.append(trip_error)
                # print('Final epoch loss was larger than target loss. We donot double the subset size')
                continue
            end_time = time.time()
            total_time += (end_time - begin_time)
            break

    logger.info('First Phase is Finished.')
    return landmarks, interval_subset_indices[-1], interval_data_subsets[-1], interval_subset_reconstructions[-1], \
           interval_final_epoch_losses[-1], interval_triplet_error[-1], total_time


def first_phase(num_landmarks, subset_size, data, dataset_size, embedding_dim,
                epochs, target_loss, first_phase_lr, bs, device, logger):
    # get embedding size interval
    embedding_size_intervals = get_subset_sizes(dataset_size, subset_size)
    print('Embedding size intervals: ', embedding_size_intervals)
    # get landmarks
    landmarks, landmark_indices = get_landmark_subset(data, number_of_landmarks=num_landmarks)

    # first phase data buffer
    interval_final_epoch_losses = []
    interval_data_subsets = []
    interval_subset_reconstructions = []
    interval_subset_indices = []
    interval_triplet_error = []
    i = 0
    total_time = 0
    subset_data = []
    # first phase flow
    logger.info('Running SOE on different subset Size Until the Criteria Defined is Satisified.')
    for index, each_subset_size in enumerate(embedding_size_intervals):
        # Get the subset data. Subsets must contain landmarks. The function takes care of this. Subset indices are
        # from the original set of indices
        subset_data, subset_indices = get_subsets(data, landmark_indices=landmark_indices, subset_size=each_subset_size)
        logger.info('First Phase Iteration Index: ' + str(index))
        logger.info('Landmarks:' + str(landmarks.shape))
        logger.info('Obtained subset size:' + str(subset_data.shape))
        logger.info('Landmark indices: ' + str(len(landmark_indices)))
        logger.info('Obtained subset indices: ' + str(len(subset_indices)))

        begin_time = time.time()
        # compute distances between landmarks and subset. Recall landmarks are also part of subset
        landmark_to_data_distance = compute_landmark_to_data_distance(landmarks=landmarks, subset=subset_data)

        # sort the distance matrix; sorted_indices are indices relative to the subset size.
        sorted_indices_pt_to_l, sorted_distances_pt_to_l = sort_distances_pt_to_landmark(
            distance_mat=landmark_to_data_distance)

        sorted_indices_l_to_pt, sorted_distances_l_to_pt = sort_landmark_to_pt_distances(
            distance_mat=landmark_to_data_distance)

        # now generate LNM-MDS triplets from the distance matrix
        generated_trips = generate_triplets_from_indices(sorted_indices_l_to_pt, sorted_indices_pt_to_l,
                                                         landmark_indices)
        trips = np.asarray(generated_trips)  # W.r.t relative indices

        end_time = time.time()
        total_time += (end_time - begin_time)

        # run soe on the subset data
        embedding_of_subset, loss_history, _, time_soe, _ = soe_adam(triplets=trips, n=each_subset_size,
                                                                     dim=embedding_dim,
                                                                     epochs=epochs,
                                                                     batch_size=min(bs * (2 ** i), trips.shape[0]),
                                                                     learning_rate=first_phase_lr * (2 ** i),
                                                                     device=device, logger=logger)
        total_time += time_soe
        final_epoch_loss = loss_history[-1]

        trip_error, error_list = triplet_error(embedding_of_subset, trips)
        logger.info('Number of Triplets: ' + str(trips.shape[0]))
        logger.info('Training triplet Error: ' + str(trip_error))
        logger.info('Subset Size: ' + str(each_subset_size) + ' Loss: ' + str(final_epoch_loss))
        begin_time = time.time()
        if final_epoch_loss < target_loss:
            i += 1
            interval_final_epoch_losses.append(final_epoch_loss)
            interval_data_subsets.append(subset_data)
            interval_subset_reconstructions.append(embedding_of_subset)
            interval_subset_indices.append(subset_indices)
            interval_triplet_error.append(trip_error)
            # print('Final epoch loss was less than target. Continue doubling the subset')
            continue
        else:
            if len(interval_final_epoch_losses) < 1:
                interval_final_epoch_losses.append(final_epoch_loss)
                interval_data_subsets.append(subset_data)
                interval_subset_reconstructions.append(embedding_of_subset)
                interval_subset_indices.append(subset_indices)
                interval_triplet_error.append(trip_error)
                # print('Final epoch loss was larger than target loss. We donot double the subset size')
                end_time = time.time()
                total_time += (end_time - begin_time)
                continue
            end_time = time.time()
            total_time += (end_time - begin_time)
            break
    logger.info('First Phase is Finished.')
    # Evaluation

    # Evaluation
    logger.info('Evaluation of the first phase Embedding: ')
    random_trip_indices = gen_triplet_indices(n=interval_data_subsets[-1].shape[0], num_trips=10000)
    test_triplet_data = gen_triplet_data(data=interval_data_subsets[-1], random_triplet_indices=random_trip_indices,
                                         batch_size=10000)

    test_triplet_error, _ = triplet_error(interval_subset_reconstructions[-1], test_triplet_data)

    logger.info('Test triplet error in the first phase is ' + str(test_triplet_error))

    return landmarks, interval_subset_indices[-1], interval_data_subsets[-1], interval_subset_reconstructions[-1], \
           interval_final_epoch_losses[-1], interval_triplet_error[-1], test_triplet_error, total_time


def loss_second_phase_loss(embedding_dim, anchors_in_shells, distances_to_p, distances_to_q, lr, logger):
    x = torch.rand((1, embedding_dim), requires_grad=True)
    # print('X: ', X)
    # all_x = x.repeat(np.array(list(anchors_in_shells)).shape[0], 1)

    # print('rep_X: ', all_X.shape)
    a = torch.from_numpy(np.array(list(anchors_in_shells), dtype=np.float))
    b = torch.from_numpy(np.array(list(distances_to_p), dtype=np.float))
    c = torch.from_numpy(np.array(list(distances_to_q), dtype=np.float))
    total_time = 0
    rep_num = np.array(list(anchors_in_shells)).shape[0]
    for it in range(0, 100):
        begin_time = time.time()
        # loss = torch.sum(torch.max(torch.from_numpy(np.zeros(b.cpu().numpy().shape)),
        #                           (torch.norm(all_x-a) - ((b + c) / 2)) ** 2 - ((b - c) / 2) ** 2))
        all_x = x[np.zeros(rep_num, dtype=int)]
        # print(b.shape, torch.norm(all_x-a, dim=1).shape)
        loss_vec = (torch.norm(all_x - a, dim=1) - ((b + c) / 2)) ** 2 - ((b - c) / 2) ** 2
        pos_inds = loss_vec > 0
        loss = torch.sum(loss_vec[pos_inds])

        gradient = grad(loss, x)[0]
        # print(gradient)
        # with torch.no_grad():
        x = x - lr * gradient
        end_time = time.time()
        total_time += (end_time - begin_time)
        all_x = x[np.zeros(rep_num, dtype=int)]
        # all_x = x.repeat(np.array(list(anchors_in_shells)).shape[0], 1)

        # logger.info('Loss at Iter ' + str(it) + ' Equals ' + str(loss.item()))
    return x.detach().numpy(), total_time


def loss_second_phase_loss_adam(embedding_dim, anchors_in_shells, distances_to_p, distances_to_q, lr, logger):
    # random_embedding = torch.rand((1, embedding_dim), )

    random_embeddings = torch.rand(size=(1, embedding_dim), dtype=torch.float)
    x = torch.Tensor(random_embeddings)
    x.requires_grad = True
    # print('X: ', X)
    optimizer = optim.Adam([x], lr=lr, amsgrad=True)
    # print('rep_X: ', all_X.shape)
    a = torch.from_numpy(np.array(list(anchors_in_shells), dtype=np.float))
    b = torch.from_numpy(np.array(list(distances_to_p), dtype=np.float))
    c = torch.from_numpy(np.array(list(distances_to_q), dtype=np.float))
    total_time = 0
    for it in range(0, 100):
        begin_time = time.time()
        all_x = x.repeat(np.array(list(anchors_in_shells)).shape[0], 1)
        point_loss = torch.sum(torch.max(torch.from_numpy(np.zeros(b.cpu().numpy().shape)),
                                         (torch.norm(all_x - a, dim=1) - ((b + c) / 2)) ** 2 - ((b - c) / 2) ** 2))
        # batch_loss = torch.sum(point_loss)

        # gradient = tor
        # gradient = grad(loss, x)[0]
        # x = x - lr * gradient

        optimizer.zero_grad()
        point_loss.backward()
        optimizer.step()

        end_time = time.time()
        total_time += (end_time - begin_time)

        # x.requires_grad = False
        # all_x = x.repeat(np.array(list(anchors_in_shells)).shape[0], 1)

        # logger.info('Change: ', torch.sum(torch.norm(all_x[1, :]-x.item())))
        # logger.info('Loss at Iter ' + str(it) + ' Equals ' + str(point_loss.item())) if it%10==0 else None
    return x.detach().numpy(), total_time


from multiprocessing import Pool
from functools import partial


def run_multiprocessing(func, ind, n_processors):
    # ctx = mp.get_context('spawn')
    mp.set_start_method('spawn', force=True)
    with Pool(processes=n_processors) as pool:
        return pool.map(func, ind)


import sys


def embed_ind(ind, embedded_points, dt_mat, embedded_indices, my_oracle, dim, logger, lr):
    if (ind % 1000 == 0):
        print('unembedded:', ind)
        sys.stdout.flush()

    initial_anchor_index = random.sample(range(len(embedded_indices)), 1)[0]

    # begin_time = time.time()
    array_size = min(2 * (dim + 1), embedded_points.shape[0])
    temp_fft_anchors_per_index = fft(dt_mat, array_size, initial_anchor_index)
    # print(temp_fft_anchors_per_index)
    # temp_fft_anchors_per_index = np.random.permutation(embedded_points.shape[0])
    # print('fft time', time.time() - begin_time)
    # total_time += (end_time - begin_time)
    # print('fftres', temp_fft_anchors_per_index)
    subset_indices = np.asarray(embedded_indices)
    fft_anchors_per_index = subset_indices[temp_fft_anchors_per_index]
    # print(fft_anchors_per_index)

    p = []
    q = []
    shells = []

    for anchor_index in fft_anchors_per_index:
        # begin_time = time.time()
        first_p, first_q, dt_to_p, dt_to_q = my_oracle.bulk_query_pq(un_embedded_point_index=ind,
                                                                     anchor_index=anchor_index,
                                                                     already_embedded_index=embedded_indices,
                                                                     already_embeddings=embedded_points)
        # print('pq-query time', time.time() - begin_time)
        if first_p is None or first_q is None:
            continue

        # print('P is', first_p)
        p.append(first_p)

        # print('Q is', first_q)
        q.append(first_q)

        relative_anchor_index = temp_fft_anchors_per_index[fft_anchors_per_index == anchor_index]
        # print(embedded_points[relative_anchor_index, :].squeeze().shape)
        shells.append([embedded_points[relative_anchor_index, :].squeeze(), dt_to_p.cpu().numpy().squeeze(),
                       dt_to_q.cpu().numpy().squeeze()])
        # logger.info('Index ' + str(each_index) + ' Length of p + q: ' + str(len(p) + len(q)))

    shells = np.asarray(shells)

    # tinme this
    embedding, additional_time = loss_second_phase_loss_adam(embedding_dim=dim, anchors_in_shells=shells[:, 0],
                                                             distances_to_p=shells[:, 1],
                                                             distances_to_q=shells[:, 2], lr=lr, logger=logger)
    return embedding


def embedding_rest_indices(my_oracle, non_embedded_indices, embedded_indices, first_phase_embedded_points, dim, lr,
                           logger):
    logger.info('Number of Unembedded Indices: ' + str(len(non_embedded_indices)))
    embedded_points = first_phase_embedded_points
    non_embedded_points_index = []
    embeddings_of_unembedded_points = []
    total_time = 0

    # print(non_embedded_indices)
    n_processors = 4
    '''
    pass the task function, followed by the parameters to processors
    '''
    dt_mat = distance_matrix(embedded_points, embedded_points)
    emb_one_arg = partial(embed_ind, embedded_points=embedded_points, dt_mat=dt_mat,
                          embedded_indices=embedded_indices,
                          my_oracle=my_oracle, dim=dim, logger=logger, lr=lr)
    begin_time = time.time()
    embeddings_of_unembedded_points = run_multiprocessing(emb_one_arg, non_embedded_indices, n_processors)
    embeddings_of_unembedded_points = np.concatenate(embeddings_of_unembedded_points)
    # print(embeddings_of_unembedded_points)
    # print(embeddings_of_unembedded_points.shape)
    total_time = (time.time() - begin_time)

    return non_embedded_indices, embeddings_of_unembedded_points, total_time
