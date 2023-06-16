import argparse
import numpy as np
import torch
import logging
import os
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import product
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.lsoe_utils.oracle import Oracle
from lib.lsoe_utils.lsoe_mproc import embedding_rest_indices as second_phase
from lib.lsoe_utils.lsoe_mproc import first_phase, first_phase_soe
from preprocessing_utils.TripletData import triplet_error, gen_triplet_indices, gen_triplet_data
from preprocessing_utils.data_select_utils import select_dataset
from logging_utils import logging_util
from config_utils.config_eval import load_config
from preprocessing_utils.TripletData import procrustes_disparity
from preprocessing_utils.TripletData import knn_classification_error


def parse_args():
    """
    To run this file use "CUDA_VISIBLE_DEVICES=3 python train_soe.py -config configs/soe/soe_evaluation.json". See
    the config file in
    the path for an example of how to construct config files.
    """
    parser = argparse.ArgumentParser(description='Run LLOE_Phase2 Experiments')
    parser.add_argument('-config', '--config_path', type=str, default='../configs/lloe/uniform_baseline.json', required=True,
                        help='Input the Config File Path')
    args = parser.parse_args()
    return args


def main(args):

    config = load_config(args.config_path)
    dataset_name = config['dataset_selected']
    batch_size = config['batch_size']
    phase1_learning_rate = config['optimizer_params']['phase1_learning_rate']
    phase2_learning_rate = config['optimizer_params']['phase2_learning_rate']
    num_landmarks = config['optimizer_params']['num_landmarks']
    subset_size = config['optimizer_params']['subset_size']
    target_loss = config['optimizer_params']['target_loss']
    epochs = config['nb_epochs']
    input_dim = config['input_dimension']
    embedding_dimension = config['output_dimension']
    n_samples = config['number_of_points']
    number_of_test_triplets = config['n_test_triplets']
    log_dir = config['log']['path']
    hyper_search = config['hyper_search']['activation']

    if hyper_search:
        run_hyper_search(config=config)
    else:
        vec_data, labels = select_dataset(dataset_name=dataset_name,
                                      input_dim=input_dim, n_samples=n_samples)

        n_points = vec_data.shape[0]  # do not remove
        number_of_landmarks = min(int(num_landmarks * n_points), 100)
        subset_size = subset_size * number_of_landmarks

        experiment_name = 'lsoe_' + 'data_' + dataset_name \
                          + '_input_dim_' + str(input_dim) \
                          + '_emb_dimension_' + str(embedding_dimension) \
                          + '_originaldimension_' + str(vec_data.shape[1]) \
                          + '_n_' + str(n_samples) \
                          + '_landmarks_' + str(number_of_landmarks) \
                          + '_bs_ ' + str(batch_size) \
                          + '_pplr_' + str(phase2_learning_rate) \
                          + '_soe_lr_' + str(phase1_learning_rate) \
                          + '_epochs_' + str(epochs)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging_path = os.path.join(log_dir, experiment_name + '.log')
        logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)

        logger.info('Name of Experiments: ' + experiment_name)
        logger.info('Dataset Name:' + dataset_name)
        logger.info('Number of Points: ' + str(n_samples))
        logger.info('Dataset Dimension:' + str(input_dim))
        logger.info('Number of Landmarks:' + str(number_of_landmarks))
        logger.info('Number of Subset Size:' + str(subset_size))
        logger.info('First Phase Epochs: ' + str(epochs))

        # set the gpu id
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        logger.info('Computing SOE - Phase 1...')


        # first phase of the algorithm
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

        logger.info('First Phase Loss: ' + str(first_phase_loss))
        logger.info('First Phase Triplet Error: ' + str(first_phase_triplet_error))
        logger.info('First Phase Number of Landmarks: ' + str(landmarks.shape))
        logger.info('First Phase Indices Number: ' + str(len(first_phase_indices)))
        logger.info('First Phase Reconstruction Size: ' + str(first_phase_reconstruction.shape))

        embedded_indices = first_phase_indices
        embedded_points = first_phase_reconstruction
        non_embedded_indices = list(set(range(vec_data.shape[0])).difference(set(embedded_indices)))
        my_oracle = Oracle(data=vec_data)
        logger.info('Second Phase: ')
        logger.info('Oracle Created...')

        logger.info('Computing LLOE - Phase 2...')
        # second phase for embedding point by point update
        second_phase_embeddings_index, \
        second_phase_embeddings, time_second_phase = second_phase(my_oracle=my_oracle,
                                                                  non_embedded_indices=non_embedded_indices,
                                                                  embedded_indices=embedded_indices,
                                                                  first_phase_embedded_points=embedded_points,
                                                                  dim=embedding_dimension,
                                                                  lr=phase2_learning_rate, logger=logger)

        # combine the first phase and second phase points and index
        final_embedding = np.zeros((vec_data.shape[0], embedding_dimension))
        # phase 1 points
        final_embedding[embedded_indices] = embedded_points
        # second phase points
        final_embedding[second_phase_embeddings_index] = second_phase_embeddings
        time_taken = time_first_phase + time_second_phase

        logger.info('Size of Dataset: ' + str(vec_data.shape[0]))
        logger.info('Size of First Phase Indices: ' + str(len(embedded_indices)))
        logger.info('Size of Second Phase Indices: ' + str(len(second_phase_embeddings_index)))

        # Evaluation
        logger.info('Evaluation of the Complete Embedding Dataset: ')
        random_trip_indices = gen_triplet_indices(n=vec_data.shape[0], num_trips=number_of_test_triplets)
        test_triplet_data = gen_triplet_data(data=vec_data, random_triplet_indices=random_trip_indices, batch_size=1000)

        test_error, embedding_error_list = triplet_error(final_embedding, test_triplet_data)
        procrustes_error = procrustes_disparity(vec_data, final_embedding)
        knn_error_ord_emb, knn_error_true_emb = knn_classification_error(final_embedding, vec_data, labels)

        # sample points for tsne visualization
        subsample = np.random.permutation(n_points)[0:500]
        x = final_embedding[subsample, :]
        sub_labels = labels[subsample]

        x_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)
        fig, ax = plt.subplots(1, 1)

        ax.scatter(x_embedded[:, 0], x_embedded[:, 1], s=3, c=sub_labels)
        fig.savefig(os.path.join(log_dir, experiment_name + '.png'))

        logger.info('Name of Experiments: ' + experiment_name)
        logger.info('Epochs: ' + str(epochs))
        logger.info('Time Taken: ' + str(time_taken) + ' seconds.')
        logger.info('Test Error: ' + str(test_error))
        logger.info('Procrustes Disparity: ' + str(procrustes_error))
        logger.info('kNN Classification Error on ground-truth: ' + str(knn_error_true_emb))
        logger.info('kNN Classification Error on embedding: ' + str(knn_error_ord_emb))

        results = {'test_error': test_error, 'procrustes': procrustes_error, 'knn_true': knn_error_true_emb,
                   'knn_ord_emb': knn_error_ord_emb, 'labels': labels,
                   'ordinal_embedding': final_embedding, 'time_taken': time_taken}
        joblib.dump(results, os.path.join(log_dir, experiment_name + '.pkl'))


def run_hyper_search(config):

    dataset_name = config['dataset_selected']
    batch_size = config['batch_size']
    epochs = config['nb_epochs']
    input_dim = config['input_dimension']
    n_samples = config['number_of_points']
    number_of_test_triplets = config['n_test_triplets']
    log_dir = config['log']['path']

    phase1_learning_rate_range = config['hyper_search']['phase1_learning_rate']
    phase2_learning_rate_range = config['hyper_search']['phase2_learning_rate']
    dimensions_range = config['hyper_search']['output_dimension']

    separator = '_'
    experiment_name = 'lloe_full_hyper_search_' + \
                      'data_' + dataset_name + \
                      '_input_dim_' + str(input_dim) + \
                      '_n_pts_' + str(n_samples) + \
                      '_num_test_trips_' + str(number_of_test_triplets) + \
                      '_output_dim_' + separator.join([str(i) for i in dimensions_range]) + \
                      '_phase1lr_' + separator.join([str(i) for i in phase1_learning_rate_range]) + \
                      '_phase2lr_' + separator.join([str(i) for i in phase2_learning_rate_range]) + \
                      '_bs_' + str(batch_size)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging_path = os.path.join(log_dir, experiment_name + '.log')
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiment: ' + experiment_name)
    logger.info('Logging Path:' + logging_path)
    logger.info('Dataset Name: ' + dataset_name)
    logger.info('Epochs: ' + str(epochs))

    best_params_test = {}
    all_results = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vec_data, labels = select_dataset(dataset_name, n_samples=n_samples, input_dim=input_dim) # input_dim is only argument for uniform. Ignored otherwise

    n_samples = vec_data.shape[0] # do not remove
    number_of_landmarks = int(0.1 * n_samples)
    subset_size = 10 * number_of_landmarks

    for emb_dim in dimensions_range:
        all_results[emb_dim] = {}
        best_test_error = 1

        logger.info('Testing on: ' + dataset_name + '. Embedding dimension is ' + str(emb_dim))
        logger.info(' ')

        for (phase1_learning_rate, phase2_learning_rate) \
                in product(phase1_learning_rate_range, phase2_learning_rate_range):

                logger.info(10*'-'+' New parameters' + 10*'-')
                logger.info('phase1_Learning Rate: ' + str(phase1_learning_rate))
                logger.info('phase2_Learning Rate: ' + str(phase2_learning_rate))
                logger.info('Number of Points: ' + str(n_samples))
                logger.info('Input Dimension: ' + str(input_dim))
                logger.info('Output Dimension: ' + str(emb_dim))
                logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
                logger.info('Batch Size: ' + str(batch_size))

                logger.info('Computing LOE_FULL...')

                # first phase of the algorithm
                landmarks, first_phase_indices, \
                first_phase_subset_size, first_phase_reconstruction, \
                first_phase_loss, first_phase_triplet_error, time_first_phase = first_phase_soe(num_landmarks=number_of_landmarks,
                                                                                            subset_size=subset_size,
                                                                                            data=vec_data, dataset_size=n_samples,
                                                                                            embedding_dim=emb_dim, epochs=epochs,
                                                                                            target_loss=0.1,
                                                                                            first_phase_lr=phase1_learning_rate,
                                                                                            device=device,
                                                                                            batch_size=batch_size, logger=logger)
                logger.info('First Phase Loss: ' + str(first_phase_loss))
                logger.info('First Phase Triplet Error: ' + str(first_phase_triplet_error))
                logger.info('First Phase Number of Landmarks: ' + str(landmarks.shape))
                logger.info('First Phase Indices Number: ' + str(len(first_phase_indices)))
                logger.info('First Phase Reconstruction Size: ' + str(first_phase_reconstruction.shape))

                embedded_indices = first_phase_indices
                embedded_points = first_phase_reconstruction
                non_embedded_indices = list(set(range(vec_data.shape[0])).difference(set(embedded_indices)))
                non_embedded_points = vec_data[non_embedded_indices, :]

                my_oracle = Oracle(data=vec_data)
                logger.info('Second Phase: ')
                logger.info('Oracle Created...')

                logger.info('Computing LLOE - Phase 2...')
                # second phase for embedding point by point update
                seocnd_phase_embeddings_index, \
                second_phase_embeddings, time_second_phase = second_phase(my_oracle=my_oracle,
                                                                          non_embedded_indices=non_embedded_indices,
                                                                          embedded_indices=embedded_indices,
                                                                          first_phase_embedded_points=embedded_points,
                                                                          dim=emb_dim,
                                                                          lr=phase2_learning_rate, logger=logger)

                # combine the first phase and second phase points and index
                final_embedding = np.zeros((vec_data.shape[0], emb_dim))
                # phase 1 points
                final_embedding[first_phase_indices] = first_phase_reconstruction
                # second phase points
                final_embedding[seocnd_phase_embeddings_index] = second_phase_embeddings

                logger.info('Size of First Phase Indices: ' + str(len(first_phase_indices)))
                logger.info('Size of Second Phase Indices: ' + str(len(seocnd_phase_embeddings_index)))
                logger.info('First Phase Triplet Error: ' + str(first_phase_triplet_error))

                # Evaluation
                logger.info('Evaluation of the Complete Embedding Dataset: ')
                random_trip_indices = gen_triplet_indices(n=vec_data.shape[0], num_trips=number_of_test_triplets)
                test_triplet_data = gen_triplet_data(data=vec_data, random_triplet_indices=random_trip_indices, batch_size=1000)

                test_error, embedding_error_list = triplet_error(final_embedding, test_triplet_data)
                time_taken = time_first_phase + time_second_phase

                logger.info('Time Taken: ' + str(time_taken) + ' seconds.')
                logger.info('Test Error: ' + str(test_error))
                #logger.info('Procrustes Disparity: ' + str(procrustes_error))
                #logger.info('kNN Classification Error on ground-truth: ' + str(knn_error_true_emb))
                #logger.info('kNN Classification Error on embedding: ' + str(knn_error_ord_emb))

                results = {'test_error': test_error, 'last_embedding': final_embedding}

                all_results[emb_dim].update({(phase1_learning_rate, phase2_learning_rate): results})

                if test_error < best_test_error:
                    best_params_test[emb_dim] = {'phase1_learning_rate': phase1_learning_rate, 'phase2_learning_rate': phase2_learning_rate,
                                              'error': test_error}
                    best_test_error = test_error

        result_name = 'lloe_full_hypersearch_' + \
                      'data_' + dataset_name + \
                      '_input_dim_' + str(input_dim) + \
                      '_n_pts_' + str(n_samples) + \
                      '_output_dim_' + str(emb_dim) + \
                      '_bs_' + str(batch_size)
        all_results['labels'] = labels
        joblib.dump(all_results[emb_dim], os.path.join(log_dir, result_name + '.pkl'))


# print all results as well again
    logger.info(10 * '-' + 'ALL RESULTS ' + 10 * '-')
    for emb_dim in dimensions_range:
        results = all_results[emb_dim]
        logger.info('Results for emb dimension ' + str(emb_dim))
        for (phase1_learning_rate, phase2_learning_rate) \
                in product(phase1_learning_rate_range, phase2_learning_rate_range):
            logger.info('phase1_learning_rate ' + str(phase1_learning_rate) + ' phase2_learning_rate ' + str(phase2_learning_rate)
                        + ' -- test error: ' + str(results[(phase1_learning_rate, phase2_learning_rate)]['test_error']))

# print best parameter settings
    for emb_dim in dimensions_range:
        logger.info('Best Parameters for emb dimension ' + str(emb_dim))
        best_on_test = best_params_test[emb_dim]
        logger.info('achieved ' + str(best_on_test['error']) + ' test error with phase1_learning_rate: ' + str(best_on_test['phase1_learning_rate'])
                    + ' phase2_learning_rate: ' + str(best_on_test['phase2_learning_rate']))


if __name__ == "__main__":
    main(parse_args())