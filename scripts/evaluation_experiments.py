import numpy as np
import logging
import torch
import argparse
import os
import sys
import math
from collections import OrderedDict
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import soe, ste, ckl, tste, gnmds, forte, landmark_oe
from preprocessing_utils.data_select_utils import select_dataset
from preprocessing_utils.TripletData import TripletBatchesDataset
from logging_utils import logging_util
from config_utils.config_eval import load_config
from preprocessing_utils.TripletData import procrustes_disparity
from preprocessing_utils.TripletData import knn_classification_error
from preprocessing_utils.TripletData import triplet_error_batches
import joblib
from lib.lsoe_utils.oracle import Oracle

from lib.lsoe_utils.lsoe_mproc import embedding_rest_indices as second_phase
from lib.lsoe_utils.lsoe_mproc import first_phase, first_phase_soe

from lib.oenn_utils import data_utils as data_utils_oenn
from lib.oenn_utils import training_routine_v3


def parse_args():
    """
    To run this file use "CUDA_VISIBLE_DEVICES=3 python train_soe.py -config configs/soe/soe_evaluation.json". See
    the config file in
    the path for an example of how to construct config files.
    """
    parser = argparse.ArgumentParser(description='Run SOE Experiments')
    parser.add_argument('-config', '--config_path', type=str, default='configs/soe/uniform_baseline.json',
                        required=True,
                        help='Input the Config File Path')
    parser.add_argument('-data', '--data_set', type=str, default='not_selected', required=False,
                        help='Manually change dataset')
    args = parser.parse_args()
    return args


def main(args):
    config = load_config(args.config_path)
    algorithm = config['algorithm']
    error_change_threshold = config['error_change_threshold']
    input_dim_range = config['tradeoff_set']['input_dimension']
    output_dimensions_range = config['tradeoff_set']['output_dimension']
    nmb_points_range = config['tradeoff_set']['number_of_points']
    batch_size_range = config['tradeoff_set']['batch_size']
    learning_rate_range = config['tradeoff_set']['learning_rate']
    triplet_multiplier_range = config['tradeoff_set']['triplets_multiplier']

    if args.data_set == 'not_selected':
        dataset_name = config['dataset_selected']
    else:
        dataset_name = args.data_set

    try:
        input_equals_output = config['tradeoff_set']['input_equals_output']
    except:
        input_equals_output = False

    epochs = config['nb_epochs']
    optimizer = config['optimizer']
    n_test_triplets = config['n_test_triplets']

    log_dir = config['log']['path']

    separator = '_'
    experiment_name = algorithm + \
                      '_data_' + dataset_name + \
                      '_input_dim_' + separator.join([str(i) for i in input_dim_range]) + \
                      '_output_dim_' + separator.join([str(i) for i in output_dimensions_range]) + \
                      '_n_pts_' + separator.join([str(i) for i in nmb_points_range]) + \
                      '_bs_' + separator.join([str(i) for i in batch_size_range]) + \
                      '_change_criterion_' + str(error_change_threshold) + \
                      '_lr_' + separator.join([str(i) for i in learning_rate_range]) + \
                      '_triplet_num_' + separator.join([str(i) for i in triplet_multiplier_range]) + \
                      '_ep_' + str(epochs)
        # create a log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging_path = os.path.join(log_dir, experiment_name + '.log')
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiments: ' + experiment_name)
    logger.info('Dataset Name: ' + dataset_name)
    logger.info('Error Change Threshold: ' + str(error_change_threshold))
    logger.info('Epochs: ' + str(epochs))

    tradeoff_results = defaultdict(dict)

    order = ['Input Dimension', 'Output Dimension', 'Number of Points',
             'Batch Size', 'Learning Rate', 'Triplet Multiplier',
             ['Train Error', 'Test Error', 'Procrustes Error', 'Knn Orig Error', 'Knn Ordinal Error', 'Time',
              'Embedding', 'Labels', 'Train Triplets']]
    experiment_range = OrderedDict({'input_dim': input_dim_range, 'output_dim': output_dimensions_range,
                                    'number_of_points': nmb_points_range, 'batch_size': batch_size_range,
                                    'learning_rate': learning_rate_range,
                                    'triplet_multiplier': triplet_multiplier_range})

    for input_dim_index, input_dim in enumerate(input_dim_range):
        for dimensions_index, embedding_dimension in enumerate(output_dimensions_range):
            for subset_index, nmb_points in enumerate(nmb_points_range):
                for batch_size_index, batch_size in enumerate(batch_size_range):
                    for lr_index, learning_rate in enumerate(learning_rate_range):
                        for trip_mindex, triplet_multiplier in enumerate(triplet_multiplier_range):

                            if (not input_equals_output) or (input_equals_output and input_dim == embedding_dimension):
                                logger.info('Learning Rate: ' + str(learning_rate))
                                logger.info('Number of Points: ' + str(nmb_points))
                                logger.info('Input Dimension: ' + str(input_dim))
                                logger.info('Output Dimension: ' + str(embedding_dimension))
                                logger.info('Number of Test Triplets: ' + str(n_test_triplets))
                                logger.info('Triplet Multiplier: ' + str(triplet_multiplier))
                                logger.info('Batch Size: ' + str(batch_size))

                                embedding, train_triplets, labels, train_error, test_error, \
                                procrustes_error, knn_orig, knn_embed, time_taken, \
                                loss_history, triplet_error_history, time_history = run_method(config, dataset_name,
                                                                                               algorithm, nmb_points,
                                                                                               input_dim,
                                                                                               embedding_dimension,
                                                                                               learning_rate,
                                                                                               batch_size,
                                                                                               triplet_multiplier,
                                                                                               optimizer, epochs,
                                                                                               n_test_triplets, logger,
                                                                                               error_change_threshold)

                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 0] = train_error
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 1] = test_error
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 2] = procrustes_error
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 3] = knn_orig
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 4] = knn_embed
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 5] = time_taken
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 6] = embedding
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 7] = labels
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 8] = loss_history
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 9] = triplet_error_history
                                tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_mindex, 10] = time_history
                                # tradeoff_results[input_dim_index, dimensions_index, subset_index,
                                #                  batch_size_index, lr_index, trip_mindex, 8] = train_triplets

    for input_dim_index, input_dim in enumerate(input_dim_range):
        for dimensions_index, embedding_dimension in enumerate(output_dimensions_range):
            for subset_index, nmb_points in enumerate(nmb_points_range):
                for batch_size_index, batch_size in enumerate(batch_size_range):
                    for lr_index, learning_rate in enumerate(learning_rate_range):
                        for trip_mindex, triplet_multiplier in enumerate(triplet_multiplier_range):
                            if (not input_equals_output) or (input_equals_output and input_dim == embedding_dimension):
                                logger.info('Input Dimension ' + str(input_dim) +
                                            ' Output Dimension ' + str(embedding_dimension) +
                                            ' Number of Points ' + str(nmb_points) +
                                            ' Batch Size ' + str(batch_size) +
                                            ' Learning Rate ' + str(learning_rate) +
                                            ' Triplet Multiplier ' + str(triplet_multiplier))
                                logger.info(' Train Error ' +
                                            str(tradeoff_results[input_dim_index, dimensions_index,
                                                                 subset_index, batch_size_index, lr_index, trip_mindex, 0])
                                            +
                                            ' Test Error ' +
                                            str(tradeoff_results[input_dim_index, dimensions_index,
                                                                 subset_index, batch_size_index, lr_index, trip_mindex, 1]))

                                logger.info(' Procrustes Error ' +
                                            str(tradeoff_results[input_dim_index, dimensions_index,
                                                                 subset_index, batch_size_index, lr_index, trip_mindex, 2]))

                                logger.info(' kNN original Emb Loss ' +
                                            str(tradeoff_results[input_dim_index, dimensions_index,
                                                                 subset_index, batch_size_index, lr_index, trip_mindex, 3])
                                            +
                                            ' kNN on Ordinal Emb Loss ' +
                                            str(tradeoff_results[input_dim_index, dimensions_index,
                                                                 subset_index, batch_size_index, lr_index, trip_mindex, 4]))

                                logger.info(' Time ' +
                                            str(tradeoff_results[input_dim_index, dimensions_index,
                                                                 subset_index, batch_size_index, lr_index, trip_mindex, 5]))
                                logger.info('-' * 20)

    data_dump = [order, experiment_range, tradeoff_results]
    joblib.dump(data_dump, os.path.join(log_dir, experiment_name + '.pkl'))


def run_method(config, dataset_name, algorithm, n_points, input_dim, embedding_dimension, learning_rate, batch_size,
               triplet_multiplier, optimizer, epochs, n_test_triplets, logger, error_change_threshold):
    vec_data, labels = select_dataset(dataset_name, n_samples=n_points, input_dim=input_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_points = vec_data.shape[0]

    triplet_num = np.int(np.ceil(triplet_multiplier * n_points * math.log2(n_points) * embedding_dimension))
    train_triplets = []
    loss_history, triplet_error_history, time_history = [], [], []

    batch_size = min(batch_size, triplet_num)

    logger.info('Computing Embedding...')
    logger.info('Number of Points: ' + str(n_points))
    logger.info('Number of Triplets: ' + str(triplet_num))
    logger.info('Input Dimension: ' + str(input_dim))
    logger.info('Output Dimension: ' + str(embedding_dimension))
    time_taken = 0
    train_error = -1  # active methods wont have a train error
    if optimizer == 'adam' and algorithm == 'soe':
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = soe.soe_adam(triplets=train_triplets,
                                                                                        n=n_points,
                                                                                        dim=embedding_dimension,
                                                                                        epochs=epochs,
                                                                                        batch_size=batch_size,
                                                                                        learning_rate=learning_rate,
                                                                                        device=device, logger=logger,
                                                                                        error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)

    elif optimizer == 'sgd' and algorithm == 'soe':
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = soe.soe_sgd(triplets=train_triplets,
                                                                                       n=n_points,
                                                                                       dim=embedding_dimension,
                                                                                       iterations=epochs, bs=batch_size,
                                                                                       lr=learning_rate,
                                                                                       device=device, logger=logger)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)
    elif algorithm == 'ste':
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = ste.ste_adam(triplets=train_triplets,
                                                                                        n=n_points,
                                                                                        dim=embedding_dimension,
                                                                                        epochs=epochs,
                                                                                        batch_size=batch_size,
                                                                                        learning_rate=learning_rate,
                                                                                        device=device, logger=logger,
                                                                                        error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)
    elif algorithm == 'tste':
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = tste.t_ste_adam(triplets=train_triplets,
                                                                                           n=n_points,
                                                                                           emb_dim=embedding_dimension,
                                                                                           epochs=epochs,
                                                                                           batch_size=batch_size,
                                                                                           learning_rate=learning_rate,
                                                                                           device=device, logger=logger,
                                                                                           error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)
    elif algorithm == 'triplet_loss':
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = soe.triplet_loss_adam(
            triplets=train_triplets,
            n=n_points, dim=embedding_dimension, iterations=epochs, batch_size=batch_size,
            lr=learning_rate, device=device, logger=logger, error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)
    elif algorithm == 'gnmds':
        regularizer = config['regularizer']
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = gnmds.gnmds(triplets=train_triplets,
                                                                                       reg_lbda=regularizer,
                                                                                       n=n_points,
                                                                                       dim=embedding_dimension,
                                                                                       epochs=epochs,
                                                                                       batch_size=batch_size,
                                                                                       learning_rate=learning_rate,
                                                                                       device=device, logger=logger,
                                                                                       error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)
    elif algorithm == 'forte':
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = forte.rank_d_pgd(triplets=train_triplets,
                                                                                            n=n_points,
                                                                                            dim=embedding_dimension,
                                                                                            epochs=epochs,
                                                                                            batch_size=batch_size,
                                                                                            learning_rate=learning_rate,
                                                                                            device=device,
                                                                                            logger=logger,
                                                                                            error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)
    elif algorithm == 'ckl':
        regularizer = config['regularizer']
        mu = config['mu']
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = ckl.ckl_k(triplets=train_triplets,
                                                                                     reg_lbda=regularizer, mu=mu,
                                                                                     n=n_points,
                                                                                     dim=embedding_dimension,
                                                                                     epochs=epochs,
                                                                                     batch_size=batch_size,
                                                                                     learning_rate=learning_rate,
                                                                                     device=device, logger=logger,
                                                                                     error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)

    elif algorithm == 'ckl_x':
        mu = config['mu']
        logger.info('Generating triplets...')

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        train_triplets = train_triplets_dataset.trips_data_indices

        x, loss_history, triplet_error_history, time_taken, time_history = ckl.ckl_x(triplets=train_triplets, mu=mu,
                                                                                     n=n_points,
                                                                                     dim=embedding_dimension,
                                                                                     epochs=epochs,
                                                                                     batch_size=batch_size,
                                                                                     learning_rate=learning_rate,
                                                                                     device=device, logger=logger,
                                                                                     error_change_threshold=error_change_threshold)
        # compute triplet error for train and test data
        train_error = triplet_error_batches(x, train_triplets)
    elif algorithm == 'loe':
        x, time_taken, train_error = landmark_oe.landmark_oe_with_data(data=vec_data, dim=embedding_dimension,
                                                                       trip_num=triplet_num,
                                                                       learning_rate=learning_rate, epochs=epochs,
                                                                       batch_size=batch_size,
                                                                       device=device, logger=logger)
    elif algorithm == 'oenn':

        number_of_neighbours = 50  # config['number_of_neighbours']
        metric = 'eu'  # config['metric']
        all_triplets, triplet_loaders = data_utils_oenn.prep_data_for_nn(vec_data=vec_data,
                                                                         labels=labels,
                                                                         triplet_num=triplet_num,
                                                                         batch_size=batch_size,
                                                                         metric=metric,
                                                                         number_of_neighbours=number_of_neighbours)

        hl_size = int(120 + (2 * embedding_dimension * math.log2(n_points)))
        x, loss_history, triplet_error_history, time_taken, time_history = training_routine_v3.create_and_train_triplet_network(
            dataset_name=dataset_name,
            ind_loaders=triplet_loaders,
            n=n_points,
            dim=embedding_dimension,
            layers=3,
            learning_rate=learning_rate,
            epochs=epochs,
            hl_size=hl_size,
            batch_size=batch_size,
            number_of_triplets=triplet_num,
            logger=logger,
            error_change_threshold=error_change_threshold)
        train_error = triplet_error_batches(x, all_triplets)

    elif algorithm == 'lloe':
        num_landmarks = config['optimizer_params']['num_landmarks']
        subset_size = config['optimizer_params']['subset_size']
        phase1_learning_rate = config['optimizer_params']['phase1_learning_rate']
        phase2_learning_rate = config['optimizer_params']['phase2_learning_rate']
        target_loss = config['optimizer_params']['target_loss']

        number_of_landmarks = min(int(num_landmarks * n_points), 100)
        subset_size = subset_size * number_of_landmarks
        landmarks, first_phase_indices, \
        first_phase_subset_size, first_phase_reconstruction, \
        first_phase_loss, first_phase_triplet_error, time_first_phase = first_phase_soe(
            num_landmarks=number_of_landmarks,
            subset_size=subset_size,
            data=vec_data, dataset_size=n_points,
            embedding_dim=embedding_dimension, epochs=epochs,
            first_phase_lr=phase1_learning_rate,
            device=device,
            target_loss=target_loss,
            batch_size=batch_size,
            logger=logger)
        embedded_indices = first_phase_indices
        embedded_points = first_phase_reconstruction
        non_embedded_indices = list(set(range(vec_data.shape[0])).difference(set(embedded_indices)))
        my_oracle = Oracle(data=vec_data)
        logger.info('Second Phase: ')
        logger.info('Oracle Created...')
        logger.info('Computing LLOE - Phase 2...')
        print(time_first_phase)
        # second phase for embedding point by point update
        second_phase_embeddings_index, \
        second_phase_embeddings, time_second_phase = second_phase(my_oracle=my_oracle,
                                                                  non_embedded_indices=non_embedded_indices,
                                                                  embedded_indices=embedded_indices,
                                                                  first_phase_embedded_points=embedded_points,
                                                                  dim=embedding_dimension,
                                                                  lr=phase2_learning_rate, logger=logger)
        # combine the first phase and second phase points and index
        x = np.zeros((vec_data.shape[0], embedding_dimension))
        # phase 1 points
        x[embedded_indices] = embedded_points
        # second phase points
        x[second_phase_embeddings_index] = second_phase_embeddings
        time_taken = time_first_phase + time_second_phase

    logger.info('Time Taken for experiment ' + str(time_taken) + ' seconds.')
    logger.info('Evaluating the computed embeddings...')

    test_triplets_dataset = TripletBatchesDataset(vec_data, labels, n_test_triplets, 1000, device)
    test_error = test_triplets_dataset.triplet_error(x)
    procrustes_error = procrustes_disparity(vec_data, x)
    knn_error_ord_emb, knn_error_true_emb = knn_classification_error(x, vec_data, labels)

    # log the errors
    logger.info('Train Error: ' + str(train_error))
    logger.info('Test Error: ' + str(test_error))
    logger.info('Procrustes Disparity: ' + str(procrustes_error))
    logger.info('kNN Classification Error on ground-truth: ' + str(knn_error_true_emb))
    logger.info('kNN Classification Error on embedding: ' + str(knn_error_ord_emb))
    return x, train_triplets, labels, train_error, test_error, procrustes_error, knn_error_true_emb, knn_error_ord_emb, time_taken, loss_history, triplet_error_history, time_history


if __name__ == "__main__":
    main(parse_args())
