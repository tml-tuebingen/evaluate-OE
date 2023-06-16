# coding: utf-8
# !/usr/bin/env python
import torch
import torch.optim
import torch.nn as nn
from torch.nn.parallel import DataParallel
from lib.oenn_utils import data_utils
import math
import time
import os
import sys
import numpy as np
from preprocessing_utils.TripletData import triplet_error_torch
from preprocessing_utils.TripletData import triplet_error


def standard_model(digits, hl_size, dim, layers):
    first_layer = [nn.Linear(digits, hl_size), nn.ReLU()]
    hidden_layers = []
    last_layer = [nn.Linear(hl_size, dim)]
    i = 1
    while i < layers:
        hidden_layers.append(nn.Linear(hl_size, hl_size))
        hidden_layers.append(nn.ReLU())
        i += 1

    emb_net = nn.Sequential(*first_layer, *hidden_layers, *last_layer)
    return emb_net


def create_and_train_triplet_network(dataset_name, logger, number_of_triplets,
                                     ind_loaders, n, dim, layers,
                                     learning_rate=5e-3, epochs=10,
                                     hl_size=100, batch_size=10000, error_change_threshold=-1):
    """
    Description: Constructs the OENN network, defines an optimizer and trains the network on the data w.r.t triplet loss.
    :param experiment_name:
    :param model_name:
    :param dataset_name:
    :param ind_loader_selective:
    :param ind_loader_random:
    :param n: # points
    :param dim: # features/ dimensions
    :param layers: # layers
    :param learning_rate: learning rate of optimizer.
    :param epochs: # epochs
    :param hl_size: # width of the hidden layer
    :param batch_size: # batch size for training
    :param logger: # for logging
    :param number_of_triplets: #TODO
    :return:
    """
    triplet_error_history = []
    loss_history = []
    time_history = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digits = int(math.ceil(math.log2(n)))

    # TODO: add iterator for extracting different criterion based triplets like random and selectived
    all_triplets = ind_loaders['random'].dataset.trips_data
    triplet_num = number_of_triplets

    # select a sample of triplets to be used for computing triplet error
    nmb_train_triplets_for_error = 100000

    bin_array = data_utils.get_binary_array(n, digits)
    data_bin = torch.Tensor(bin_array).to(device)

    emb_net = standard_model(digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    emb_net = emb_net.to(device)
    logger.info('Neural Network Architecture Defined for the experiment: ')
    logger.info(emb_net)
    logger.info('Number of Digits: ' + str(digits))

    if torch.cuda.device_count() > 1:
        emb_net = DataParallel(emb_net)
        logger.info('Detected a Multiple GPU setup.')
        logger.info('Training Neural Network using gpus: ' + str(torch.cuda.device_count()))

    # Optimizer
    optimizer = torch.optim.Adam(emb_net.parameters(), lr=learning_rate, amsgrad=True)
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    criterion = criterion.to(device)

    logger.info('#### Dataset Selection ####')
    logger.info('#### Network and learning parameters ####')
    logger.info('------------------------------------------')
    logger.info('Dataset Name: ' + str(dataset_name))
    logger.info('Number of hidden layers: ' + str(layers))
    logger.info('Hidden layer width: ' + str(hl_size))
    logger.info('Batch size: ' + str(batch_size))
    logger.info('Embedding dimension: ' + str(dim))
    logger.info('Learning rate: ' + str(learning_rate))
    logger.info('Number of epochs: ' + str(epochs))
    logger.info('Number of triplets: ' + str(number_of_triplets))
    logger.info(' #### Training begins ####')
    logger.info('---------------------------')

    logger.info('Computing the initial embedding from the randomly initialized network')
    first_emb = emb_net(data_bin) # First random embedding
    sys.stdout.flush()
    best_x = first_emb
    logger.info('Finished computing the initial embedding from the network: ')
    logger.info('Size of Computed Embedding: ' + str(first_emb.shape))

    epoch_loss = 0
    epoch_length = 0
    logger.info('Computing Initial Loss for the neural network...')
    for each_loader_type in ind_loaders.keys():
        ind_loader = ind_loaders[each_loader_type]
        logger.info('Going through:' + str(each_loader_type) + ' triplets')

        for batch_ind, trips in enumerate(ind_loader):
            sys.stdout.flush()
            trip = trips.squeeze().to(device).float()

            # Forward pass
            embedded_a = emb_net(trip[:, :digits])
            embedded_p = emb_net(trip[:, digits:2 * digits])
            embedded_n = emb_net(trip[:, 2 * digits:])
            # Compute loss
            loss = criterion(embedded_a, embedded_p, embedded_n).to(device)

            if batch_ind % 10 == 0:
                logger.info('Epoch: ' + '-1' + ' Mini batch: ' + str(batch_ind) +
                            '/' + str(len(ind_loader)) + ' Mini batch Loss: ' + str(loss.item()))
                sys.stdout.flush()  # Prints faster to the out file
            epoch_loss += loss.item()
            epoch_length += 1
    epoch_loss_init = epoch_loss/epoch_length
    logger.info('Initial Epoch Finished')
    logger.info('Init Epoch: ' + ' - Average Epoch Loss:  ' + str(epoch_loss_init))
    logger.info('Computing Initial Triplet Error...')
    loss_history.append(epoch_loss_init)

    # keep triplet error computation on CPU
    num_of_triplets_for_te = min(nmb_train_triplets_for_error, triplet_num)
    error_batch_indices = np.random.randint(all_triplets.shape[0], size=num_of_triplets_for_te)
    logger.info('Number of Triplets chosen for Triplet Error Computation: ' + str(num_of_triplets_for_te))
    triplet_error_init = triplet_error(emb=first_emb.detach().cpu().numpy(),
                                       trips=all_triplets[error_batch_indices, :])[0]
    triplet_error_history.append(triplet_error_init)
    logger.info('Initial Triplet Error: ' + str(triplet_error_init))

    sys.stdout.flush()  # prints faster to the out file

    # Training begins
    total_time = 0
    time_to_best = 0
    best_triplet_error = triplet_error_history[0]
    for ep in range(epochs):
        # Epoch is one pass over the dataset
        epoch_loss = 0
        epoch_length = 0
        epoch_time = 0
        logger.info('Digits: ' + str(digits))
        for each_loader_type in ind_loaders.keys():
            ind_loader = ind_loaders[each_loader_type]
            logger.info('Going through:' + str(each_loader_type) + ' triplets')

            for batch_ind, trips in enumerate(ind_loader):
                sys.stdout.flush()
                trip = trips.squeeze().to(device).float()

                # Training time
                begin_train_time = time.time()
                # Forward pass
                embedded_a = emb_net(trip[:, :digits])
                embedded_p = emb_net(trip[:, digits:2 * digits])
                embedded_n = emb_net(trip[:, 2 * digits:])
                # Compute loss
                loss = criterion(embedded_a, embedded_p, embedded_n).to(device)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # End of training
                end_train_time = time.time()
                if batch_ind % 10 == 0:
                    logger.info('Epoch: ' + str(ep) + ' Mini batch: ' + str(batch_ind) +
                                '/' + str(len(ind_loader)) + ' Loss: ' + str(loss.item()))
                    sys.stdout.flush()  # Prints faster to the out file
                epoch_loss += loss.item()
                epoch_time += (end_train_time - begin_train_time)
                epoch_length += 1

        total_time += epoch_time
        epoch_loss = epoch_loss/epoch_length
        time_history.append(total_time)
        loss_history.append(epoch_loss)

        logger.info('Epoch: ' + str(ep) + ' - Average Epoch Loss:  ' + str(epoch_loss) +
                     ',Total Training time ' + str(total_time))
        sys.stdout.flush()  # Prints faster to the out file

        if ep % 50 == 0 or ep == epochs - 1:
            embedding = emb_net(data_bin)  # Feed the binary array of indices to the network and generate embeddings: FP
            if error_change_threshold != -1:
                error_batch_indices = np.random.randint(all_triplets.shape[0], size=num_of_triplets_for_te)
                triplet_error_for_epoch = triplet_error(emb=embedding.detach().cpu().numpy(),
                                                        trips=all_triplets[error_batch_indices, :])[0]
                triplet_error_history.append(triplet_error_for_epoch)
                if triplet_error_for_epoch < best_triplet_error - error_change_threshold:
                    best_x = embedding
                    best_triplet_error = triplet_error_for_epoch
                    time_to_best = total_time
                    logger.info('Found new best in Epoch: ' + str(ep) + ' Loss: ' + str(epoch_loss)
                                + ' Triplet error: ' + str(triplet_error_for_epoch))
                logger.info('Epoch: ' + str(ep) + ' Loss: ' + str(epoch_loss) + ' Triplet error: '
                            + str(triplet_error_for_epoch))
                sys.stdout.flush()
            else:
                best_x = embedding
                time_to_best = total_time
                logger.info('Epoch: ' + str(ep) + ' Loss: ' + str(epoch_loss))
                sys.stdout.flush()

    return best_x.detach().cpu().numpy(), loss_history, triplet_error_history, time_to_best, time_history
