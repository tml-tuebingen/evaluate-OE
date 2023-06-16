import numpy as np
import logging
import torch
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
import os
import sys
from itertools import product
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.oenn_utils import data_utils
from lib.oenn_utils import training_routine_v3
from preprocessing_utils.data_select_utils import select_dataset
from logging_utils import logging_util
from config_utils.config_eval import load_config
from preprocessing_utils.TripletData import procrustes_disparity, triplet_error_torch, triplet_error
from preprocessing_utils.TripletData import knn_classification_error



def parse_args():
    """
    To run this file use "CUDA_VISIBLE_DEVICES=3 python train_soe.py -config configs/oenn/uniform_baseline.json". See
    the config file in
    the path for an example of how to construct config files.
    """
    parser = argparse.ArgumentParser(description='Run OENN Experiments')
    parser.add_argument('-config', '--config_path', type=str, default='configs/oenn/uniform_baseline.json', required=True,
                        help='Input the Config File Path')
    args = parser.parse_args()
    return args


def main(args):
    # usual options
    config = load_config(args.config_path)
    model_name = config['model_name']
    dataset_name = config['dataset_selected']
    error_change_threshold = config['error_change_threshold']
    batch_size = config['batch_size']
    learning_rate = config['optimizer_params']['learning_rate']
    epochs = config['nb_epochs']
    input_dim = config['input_dimension']
    embedding_dimension = config['output_dimension']
    n_samples = config['number_of_points']
    number_of_test_triplets = config['n_test_triplets']
    triplet_multiplier = config['triplets_multiplier']
    log_dir = config['log']['path']
    optimizer = config['optimizer']

    # NOTE: These are options for metric and sampling used. We keep them constant.
    sampling = 'random'  # config['sampling'] # random sampling of triplets or selective sampling
    sampling_ratio = 1.0  # config['sampling_ratio'] # fraction of triplets that are chosen randomly
    number_of_neighbours = 50  # config['number_of_neighbours']
    metric = 'eu'  # config['metric']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # network configurations
    num_layers = config['num_layers']
    hl_scale = config['hl_scale']
    hyper_search = config['hyper_search']['activation']
    if hyper_search:
        run_hypersearch(config)
    else:
        # select the dataset
        vec_data, labels = select_dataset(dataset_name, input_dim=input_dim, n_samples=n_samples)

        n_points = vec_data.shape[0]

        hl_size = int(120 + (1 * hl_scale * embedding_dimension * math.log2(n_points)))  # Hidden layer size
        triplet_num = np.int(np.ceil(triplet_multiplier * n_points * math.log2(n_points) * embedding_dimension))

        all_triplets, triplet_loaders = data_utils.prep_data_for_nn(vec_data=vec_data,
                                                                    labels=labels,
                                                                    triplet_num=triplet_num,
                                                                    batch_size=batch_size,
                                                                    metric=metric,
                                                                    number_of_neighbours=number_of_neighbours)

        experiment_name = 'oenn_data_' + dataset_name + '_input_dim_' \
                          + str(input_dim) + '_dimensions_' \
                          + str(embedding_dimension) + '_triplet_num_' + str(triplet_num) + '_n_' + str(n_points)

        # logging path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging_path = os.path.join(log_dir, experiment_name + '.log')
        logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)

        logger.info('Name of Experiments: ' + experiment_name)
        logger.info('Logging Path:' + logging_path)
        logger.info('Dataset Name: ' + dataset_name)
        logger.info('Error Change Threshold: ' + str(error_change_threshold))
        logger.info('Epochs: ' + str(epochs))
        logger.info('Learning Rate: ' + str(learning_rate))
        logger.info('Number of Points: ' + str(n_points))
        logger.info('Input Dimension: ' + str(input_dim))
        logger.info('Output Dimension: ' + str(embedding_dimension))
        logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
        logger.info('Triplet Multiplier: ' + str(triplet_multiplier))
        logger.info('Batch Size: ' + str(batch_size))

        x, \
        loss_history, \
        triplet_error_history, \
        time_taken, \
        time_history = training_routine_v3.create_and_train_triplet_network(
            dataset_name=dataset_name,
            ind_loaders=triplet_loaders,
            n=n_points,
            dim=embedding_dimension,
            layers=num_layers,
            learning_rate=learning_rate,
            epochs=epochs,
            hl_size=hl_size,
            batch_size=batch_size,
            number_of_triplets=triplet_num,
            logger=logger, error_change_threshold=error_change_threshold)

        logger.info('Evaluating the computed embeddings...')

        train_error, _ = triplet_error(x, all_triplets)

        random_triplet_indices = data_utils.gen_triplet_indices(n_points, number_of_test_triplets)
        random_triplets = data_utils.gen_triplet_data(vec_data, random_triplet_indices, 10000)
        test_error, _ = triplet_error(x, random_triplets)
        procrustes_error = procrustes_disparity(vec_data, x)
        knn_error_ord_emb, knn_error_true_emb = knn_classification_error(x, vec_data, labels)

        subsample = np.random.permutation(n_points)[0:500]
        X = x[subsample, :]
        sublabel = labels[subsample]

        X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
        fig, ax = plt.subplots(1, 1)

        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=3, c=sublabel)
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


def run_hypersearch(config):

    model_name = config['model_name']
    dataset_name = config['dataset_selected']
    batch_size = config['batch_size']
    number_of_test_triplets = config['n_test_triplets']
    epochs = config['nb_epochs']
    n_samples = config['number_of_points']
    log_dir = config['log']['path']
    optimizer = config['optimizer']
    input_dim = config['input_dimension']
    error_change_threshold = config['error_change_threshold']

    # NOTE: These are options for metric and sampling used. We keep them constant.
    sampling = 'random'  # config['sampling'] # random sampling of triplets or selective sampling
    sampling_ratio = 1.0  # config['sampling_ratio'] # fraction of triplets that are chosen randomly
    number_of_neighbours = 50  # config['number_of_neighbours']
    metric = 'eu'  # config['metric']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # neural network parameters
    num_layers = config['num_layers']
    hl_scale = config['hl_scale']

    triplet_multiplier_range = config['hyper_search']['triplets_multiplier']
    learning_rate_range = config['hyper_search']['learning_rate']
    dimensions_range = config['hyper_search']['output_dimension']

    separator = '_'
    experiment_name = 'oenn_hyper_search_' + \
                      'data_' + dataset_name + \
                      '_model_' + model_name + \
                      '_input_dim_' + str(input_dim) + \
                      '_n_pts_' + str(n_samples) + \
                      '_num_test_trips_' + str(number_of_test_triplets) + \
                      '_output_dim_' + separator.join([str(i) for i in dimensions_range]) + \
                      '_lr_' + separator.join([str(i) for i in learning_rate_range]) + \
                      '_bs_' + str(batch_size) + \
                      '_layers_' + str(num_layers) + \
                      '_hl_scale' + str(hl_scale) + \
                      '_triplet_number_' + separator.join([str(i) for i in triplet_multiplier_range])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging_path = os.path.join(log_dir, experiment_name + '.log')
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiment: ' + experiment_name)
    logger.info('Logging Path:' + logging_path)
    logger.info('Dataset Name: ' + dataset_name)
    logger.info('Epochs: ' + str(epochs))

    best_params_train = {}
    best_params_test = {}
    all_results = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vec_data, labels = select_dataset(dataset_name, n_samples=n_samples,
                                      input_dim=input_dim)  # input_dim is only argument for uniform. Ignored otherwise
    n_points = vec_data.shape[0]
    logn = int(np.log2(n_points))

    for (emb_dim, triplet_multiplier) in product(dimensions_range, triplet_multiplier_range):
        all_results[(emb_dim, triplet_multiplier)] = {}
        best_train_error = 1
        best_test_error = 1

        triplet_num = triplet_multiplier * logn * n_points * emb_dim

        batch_size = min(batch_size, triplet_num)
        hl_size = int(120 + (1 * hl_scale * emb_dim * logn))  # Hidden layer size

        all_triplets, triplet_loaders = data_utils.prep_data_for_nn(vec_data=vec_data,
                                                                    labels=labels,
                                                                    triplet_num=triplet_num,
                                                                    batch_size=batch_size,
                                                                    metric=metric,
                                                                    number_of_neighbours=number_of_neighbours)

        logger.info('Testing on: ' + dataset_name + '. Embedding dimension is ' + str(emb_dim))
        logger.info(' ')
        for learning_rate in learning_rate_range:

            logger.info(10 * '-' + ' New parameters' + 10 * '-')
            logger.info('Learning Rate: ' + str(learning_rate))
            logger.info('Number of Points: ' + str(n_points))
            logger.info('Input Dimension: ' + str(input_dim))
            logger.info('Output Dimension: ' + str(emb_dim))
            logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
            logger.info('Triplet Multiplier: ' + str(triplet_multiplier))
            logger.info('Batch Size: ' + str(batch_size))

            logger.info('Computing OENN...')

            x, \
            loss_history, \
            triplet_error_history, \
            time_taken, time_history = training_routine_v3.create_and_train_triplet_network(
                dataset_name=dataset_name,
                ind_loaders=triplet_loaders,
                n=n_points,
                dim=emb_dim,
                layers=num_layers,
                learning_rate=learning_rate,
                epochs=epochs,
                hl_size=hl_size,
                batch_size=batch_size,
                number_of_triplets=triplet_num,
                logger=logger, error_change_threshold=error_change_threshold)


            logger.info('Evaluating the computed embeddings...')
            train_error, _ = triplet_error(x, all_triplets)
            random_triplet_indices = data_utils.gen_triplet_indices(n_points, number_of_test_triplets)
            random_triplets = data_utils.gen_triplet_data(vec_data, random_triplet_indices, 10000)
            random_triplets = torch.Tensor(random_triplets).to(device).long()
            test_error, _ = triplet_error_torch(x, random_triplets)
            test_error = test_error.item()
            procrustes_error = procrustes_disparity(vec_data, x)
            knn_error_ord_emb, knn_error_true_emb = knn_classification_error(x, vec_data, labels)

            logger.info('Epochs: ' + str(epochs))
            logger.info('Time Taken: ' + str(time_taken) + ' seconds.')
            logger.info('Train Error: ' + str(train_error))
            logger.info('Test Error: ' + str(test_error))
            logger.info('Procrustes Disparity: ' + str(procrustes_error))
            logger.info('kNN Classification Error on ground-truth: ' + str(knn_error_true_emb))
            logger.info('kNN Classification Error on embedding: ' + str(knn_error_ord_emb))

            results = {'train_error': train_error, 'test_error': test_error, 'loss_history': loss_history,
                       'error_history': triplet_error_history,
                       'last_embedding': x}
            all_results[(emb_dim, triplet_multiplier)].update({learning_rate: results})

            if test_error < best_test_error:
                best_params_test[(emb_dim, triplet_multiplier)] = {'learning_rate': learning_rate,
                                                                   'optimizer': optimizer, 'error': test_error}
                best_test_error = test_error
            if train_error < best_train_error:
                best_params_train[(emb_dim, triplet_multiplier)] = {'learning_rate': learning_rate,
                                                                    'optimizer': optimizer, 'error': train_error}
                best_train_error = train_error

        result_name = 'oenn_convergence_' + \
                      'data_' + dataset_name + \
                      '_input_dim_' + str(input_dim) + \
                      '_n_pts_' + str(n_points) + \
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
        for learning_rate in learning_rate_range:
            logger.info('learning rate ' + str(learning_rate)
                        + ' -- train error: ' + str(results[learning_rate]['train_error']) + ' test error: '
                        + str(results[learning_rate]['test_error']))

    # print best parameter settings
    for (emb_dim, triplet_multiplier) in product(dimensions_range, triplet_multiplier_range):
        logger.info(
            'Best Parameters for emb dimension ' + str(emb_dim) + ' and triplet multiplier ' + str(triplet_multiplier))
        best_on_train = best_params_train[(emb_dim, triplet_multiplier)]
        best_on_test = best_params_test[(emb_dim, triplet_multiplier)]
        logger.info('achieved ' + str(best_on_train['error']) + ' train error with learning rate: ' + str(
            best_on_train['learning_rate'])
                    )
        logger.info('achieved ' + str(best_on_test['error']) + ' test error with learning rate: ' + str(
            best_on_test['learning_rate'])
                    )


if __name__ == "__main__":
    main(parse_args())
