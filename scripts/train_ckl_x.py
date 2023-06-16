import numpy as np
import logging
import torch
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
from itertools import product
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import ckl
from preprocessing_utils.data_select_utils import select_dataset
from preprocessing_utils.TripletData import TripletBatchesDataset
from logging_utils import logging_util
from config_utils.config_eval import load_config
from preprocessing_utils.TripletData import procrustes_disparity
from preprocessing_utils.TripletData import knn_classification_error


def parse_args():
    """
    To run this file use "CUDA_VISIBLE_DEVICES=3 python train_soe.py -config configs/ckl/uniform_baseline.json". See
    the config file in
    the path for an example of how to construct config files.
    """
    parser = argparse.ArgumentParser(description='Run CKL Experiments')
    parser.add_argument('-config', '--config_path', type=str, default='configs/ckl/uniform_baseline.json',
                        required=True,
                        help='Input the Config File Path')
    args = parser.parse_args()
    return args


def main(args):

    config = load_config(args.config_path)
    dataset_name = config['dataset_selected']
    error_change_threshold = config['error_change_threshold']
    batch_size = config['batch_size']
    learning_rate = config['optimizer_params']['learning_rate']
    mu = config['mu']
    epochs = config['nb_epochs']
    input_dim = config['input_dimension']
    embedding_dimension = config['output_dimension']
    n_samples = config['number_of_points']
    number_of_test_triplets = config['n_test_triplets']
    triplet_multiplier = config['triplets_multiplier']
    log_dir = config['log']['path']
    hyper_search = config['hyper_search']['activation']
    optimizer = config['optimizer']

    if hyper_search:
        run_hyperparameter_search(config=config)
    else:
        vec_data, labels = select_dataset(dataset_name, n_samples=n_samples, input_dim=input_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        n = vec_data.shape[0]
        logn = int(np.log2(n))
        triplet_num = triplet_multiplier * logn * n * embedding_dimension

        batch_size = min(batch_size, triplet_num)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        experiment_name = 'ckl_x_' + \
                          'data_' + dataset_name + \
                          '_error_change_threshold_' + str(error_change_threshold) + \
                          '_input_dim_' + str(input_dim) + \
                          '_output_dim_' + str(embedding_dimension) + \
                           '_originaldimension_' + str(vec_data.shape[1]) + \
                          '_triplet_num_' + str(triplet_multiplier) + \
                          '_n_pts_' + str(n) + \
                          '_lr_' + str(learning_rate) + \
                          '_mu_' + str(mu) + \
                          '_optimizer_' + str(optimizer) + \
                          '_bs_' + str(batch_size)

        # create a logging file for extensive logging
        logging_path = os.path.join(log_dir, experiment_name + '.log')
        logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)

        logger.info('Name of Experiments: ' + experiment_name)
        logger.info('Logging Path:' + logging_path)
        logger.info('Dataset Name: ' + dataset_name)
        logger.info('Error Change Threshold: ' + str(error_change_threshold))
        logger.info('Epochs: ' + str(epochs))
        logger.info('Learning Rate: ' + str(learning_rate))
        logger.info('Mu: ' + str(mu))
        logger.info('Number of Points: ' + str(n))
        logger.info('Input Dimension: ' + str(input_dim))
        logger.info('Output Dimension: ' + str(embedding_dimension))
        logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
        logger.info('Triplet Multiplier: ' + str(triplet_multiplier))
        logger.info('Batch Size: ' + str(batch_size))

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)

        logger.info('Computing CKL_x...')

        x, loss_history, triplet_error_history, time_taken, time_history = ckl.ckl_x(triplets=train_triplets_dataset.trips_data_indices, n=n,
                                                                       dim=embedding_dimension, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                                                       device=device, logger=logger, mu=mu, error_change_threshold=error_change_threshold)


        logger.info('Evaluating the computed embeddings...')
        # compute triplet error for train and test data
        train_error = train_triplets_dataset.triplet_error(x)
        test_triplets_dataset = TripletBatchesDataset(vec_data, labels, number_of_test_triplets, 1000, device)
        test_error = test_triplets_dataset.triplet_error(x)
        procrustes_error = procrustes_disparity(vec_data, x)
        knn_error_ord_emb, knn_error_true_emb = knn_classification_error(x, vec_data, labels)

        # sample points for tsne visualization
        subsample = np.random.permutation(n)[0:500]
        x = x[subsample, :]
        sub_labels = labels[subsample]

        x_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)
        fig, ax = plt.subplots(1, 1)

        ax.scatter(x_embedded[:, 0], x_embedded[:, 1], s=3, c=sub_labels)
        fig.savefig(os.path.join(log_dir, experiment_name + '.png'))

        logger.info('Name of Experiments: ' + experiment_name)
        logger.info('Epochs: ' + str(epochs))
        logger.info('Time Taken: ' + str(time_taken) + ' seconds.')
        logger.info('Train Error: ' + str(train_error))
        logger.info('Test Error: ' + str(test_error))
        logger.info('Procrustes Disparity: ' + str(procrustes_error))
        logger.info('kNN Classification Error on ground-truth: ' + str(knn_error_true_emb))
        logger.info('kNN Classification Error on embedding: ' + str(knn_error_ord_emb))

        results = {'train_error': train_error, 'test_error': test_error, 'procrustes': procrustes_error, 'knn_true': knn_error_true_emb,
                   'knn_ord_emb': knn_error_ord_emb, 'labels': labels,
                   'loss_history': loss_history, 'error_history': triplet_error_history,
                   'ordinal_embedding': x, 'time_taken': time_taken}
        joblib.dump(results, os.path.join(log_dir, experiment_name + '.pkl'))


def run_hyperparameter_search(config):

    dataset_name = config['dataset_selected']
    batch_size = config['batch_size']
    iterations = config['nb_epochs']
    input_dim = config['input_dimension']
    n_samples = config['number_of_points']
    number_of_test_triplets = config['n_test_triplets']
    log_dir = config['log']['path']
    triplet_multiplier_range = config['hyper_search']['triplets_multiplier']
    learning_rate_range = config['hyper_search']['learning_rate']
    regularizer_range = config['hyper_search']['regularizer']
    mu_range = config['hyper_search']['mu']
    optimizer = config['optimizer']
    dimensions_range = config['hyper_search']['output_dimension']

    separator = '_'
    experiment_name = 'ckl_hyper_search_' + \
                      'data_' + dataset_name + \
                      '_input_dim_' + str(input_dim) + \
                      '_n_pts_' + str(n_samples) + \
                      '_num_test_trips_' + str(number_of_test_triplets) + \
                      '_output_dim_' + separator.join([str(i) for i in dimensions_range]) + \
                      '_lr_' + separator.join([str(i) for i in learning_rate_range]) + \
                      '_reg_' + separator.join([str(i) for i in regularizer_range]) + \
                      '_mu_' + separator.join([str(i) for i in mu_range]) + \
                      '_bs_' + str(batch_size) + \
                      '_triplet_number_' + separator.join([str(i) for i in triplet_multiplier_range])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging_path = os.path.join(log_dir, experiment_name + '.log')
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiment: ' + experiment_name)
    logger.info('Logging Path:' + logging_path)
    logger.info('Dataset Name: ' + dataset_name)
    logger.info('Epochs: ' + str(iterations))

    best_params_train = {}
    best_params_test = {}
    all_results = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vec_data, labels = select_dataset(dataset_name, n_samples=n_samples, input_dim=input_dim) # input_dim is only argument for uniform. Ignored otherwise

    n = vec_data.shape[0]
    logn = int(np.log2(n))

    for (emb_dim, triplet_multiplier) in product(dimensions_range, triplet_multiplier_range):
        all_results[(emb_dim, triplet_multiplier)] = {}
        best_train_error = 1
        best_test_error = 1

        triplet_num = triplet_multiplier * logn * n * emb_dim

        batch_size = min(batch_size, triplet_num)
        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, batch_size, device)
        logger.info('Testing on: ' + dataset_name + '. Embedding dimension is ' + str(emb_dim))
        logger.info(' ')

        for (learning_rate, regularizer, mu) in product(learning_rate_range, regularizer_range, mu_range):

            logger.info(10 * '-' + ' New parameters' + 10 * '-')
            logger.info('Learning Rate: ' + str(learning_rate))
            logger.info('Regularizer: ' + str(regularizer))
            logger.info('Mu: ' + str(mu))
            logger.info('Number of Points: ' + str(n))
            logger.info('Input Dimension: ' + str(input_dim))
            logger.info('Output Dimension: ' + str(emb_dim))
            logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
            logger.info('Triplet Multiplier: ' + str(triplet_multiplier))
            logger.info('Batch Size: ' + str(batch_size))

            logger.info('Computing CKL...')

            x, loss_history, triplet_error_history, time_taken, time_history = ckl.ckl_k(triplets=train_triplets_dataset.trips_data_indices, n=n,
                                                                           dim=emb_dim, epochs=iterations, batch_size=batch_size, reg_lbda=regularizer,
                                                                           mu=mu,
                                                                           learning_rate=learning_rate, device=device, logger=logger)

            logger.info('Evaluating the computed embeddings...')
            # compute triplet error for train and test data
            train_error = train_triplets_dataset.triplet_error(x, sub_sample=False)
            logger.info('Triplet Error on Training Triplets: ' + str(train_error))
            test_triplets_dataset = TripletBatchesDataset(vec_data, labels, number_of_test_triplets, 1000, device)
            test_error = test_triplets_dataset.triplet_error(x)
            #procrustes_error = procrustes_disparity(vec_data, X)
            #knn_error_ord_emb, knn_error_true_emb = knn_classification_error(X, vec_data, labels)

            logger.info('Time Taken: ' + str(time_taken) + ' seconds.')
            logger.info('Train Error: ' + str(train_error))
            logger.info('Test Error: ' + str(test_error))
            #logger.info('Procrustes Disparity: ' + str(procrustes_error))
            #logger.info('kNN Classification Error on ground-truth: ' + str(knn_error_true_emb))
            #logger.info('kNN Classification Error on embedding: ' + str(knn_error_ord_emb))

            results = {'train_error': train_error, 'test_error': test_error, 'loss_history': loss_history, 'error_history': triplet_error_history,
                       'last_embedding': x}

            all_results[(emb_dim, triplet_multiplier)].update({(learning_rate, regularizer, mu): results})

            if test_error < best_test_error:
                best_params_test[(emb_dim, triplet_multiplier)] = {'learning_rate': learning_rate, 'regularizer': regularizer,
                                                                   'mu': mu,
                                                                   'optimizer': optimizer, 'error': test_error}
                best_test_error = test_error
            if train_error < best_train_error:
                best_params_train[(emb_dim, triplet_multiplier)] = {'learning_rate': learning_rate, 'regularizer': regularizer,
                                                                    'mu': mu,
                                                                    'optimizer': optimizer, 'error': train_error}
                best_train_error = train_error

        result_name = 'ckl_convergence_' + \
                      'data_' + dataset_name + \
                      '_input_dim_' + str(input_dim) + \
                      '_n_pts_' + str(n_samples) + \
                      '_output_dim_' + str(emb_dim) + \
                      '_bs_' + str(batch_size) + \
                      '_triplet_number_' + str(triplet_multiplier)
        all_results['labels'] = labels
        joblib.dump(all_results[(emb_dim, triplet_multiplier)], os.path.join(log_dir, result_name + '.pkl'))


    # print all results as well again
    logger.info(10 * '-' + 'ALL RESULTS ' + 10 * '-')
    for (emb_dim, triplet_multiplier) in product(dimensions_range, triplet_multiplier_range):
        results = all_results[(emb_dim, triplet_multiplier)]
        logger.info('Results for emb dimension ' + str(emb_dim) + ' and triplet multiplier ' + str(triplet_multiplier))
        for (learning_rate, regularizer, mu) in product(learning_rate_range, regularizer_range, mu_range):
            logger.info('learning rate ' + str(learning_rate) + ' regularizer ' + str(regularizer) + ' mu ' + str(mu)
                        + ' -- train error: ' + str(results[(learning_rate, regularizer, mu)]['train_error']) + ' test error: '
                        + str(results[(learning_rate, regularizer, mu)]['test_error']))

    # print best parameter settings
    for (emb_dim, triplet_multiplier) in product(dimensions_range, triplet_multiplier_range):
        logger.info('Best Parameters for emb dimension ' + str(emb_dim) + ' and triplet multiplier ' + str(triplet_multiplier))
        best_on_train = best_params_train[(emb_dim, triplet_multiplier)]
        best_on_test = best_params_test[(emb_dim, triplet_multiplier)]
        logger.info('achieved ' + str(best_on_train['error']) + ' train error with learning rate: ' + str(best_on_train['learning_rate']) + ', reg: '
                    + str(best_on_train['regularizer']) + ', mu: ' + str(best_on_train['mu']))
        logger.info('achieved ' + str(best_on_test['error']) + ' test error with learning rate: ' + str(best_on_test['learning_rate']) + ', reg: '
                    + str(best_on_test['regularizer']) + ', mu: ' + str(best_on_test['mu']))

if __name__ == "__main__":
    main(parse_args())
