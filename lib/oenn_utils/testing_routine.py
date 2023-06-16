# coding: utf-8
# !/usr/bin/env python
import torch
import torch.optim
import torch.nn as nn
from torch.nn.parallel import DataParallel
from train_utils import data_utils
import math
import time
import os
import sys
import numpy as np
from models.models import standard_model


def define_model(model_name, digits, hl_size, dim, layers):
    # Constructing the Network
    if model_name == 'standard':
        emb_net = standard_model(digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    return emb_net


def create_and_test_triplet_network(batch_triplet_indices_loader, experiment_name, path_to_emb_net, unseen_triplets,
                                    dataset_name, model_name,
                                    logger, test_n, n,
                                    dim, layers, learning_rate=5e-2, epochs=20, hl_size=100):
    """
    Description: Constructs the OENN network, defines an optimizer and trains the network on the data w.r.t triplet loss.
    :param model_name:
    :param dataset_name:
    :param test_n:
    :param path_to_emb_net: Data loader object. Gives triplet indices in batches.
    :param n: # points
    :param dim: # features/ dimensions
    :param layers: # layers
    :param learning_rate: learning rate of optimizer.
    :param epochs: # epochs
    :param hl_size: # width of the hidden layer
    :param unseen_triplets: #TODO
    :param logger: # for logging
    :return:
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    digits = int(math.ceil(math.log2(n)))

    #  Define train model
    emb_net_train = define_model(model_name=model_name, digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    emb_net_train = emb_net_train.to(device)

    for param in emb_net_train.parameters():
        param.requires_grad = False

    if torch.cuda.device_count() > 1:
        emb_net_train = DataParallel(emb_net_train)
        print('multi-gpu')

    checkpoint = torch.load(path_to_emb_net)['model_state_dict']
    key_word = list(checkpoint.keys())[0].split('.')[0]
    if key_word == 'module':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        emb_net_train.load_state_dict(new_state_dict)
    else:
        emb_net_train.load_state_dict(checkpoint)

    emb_net_train.eval()

    #  Define test model
    emb_net_test = define_model(model_name=model_name, digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    emb_net_test = emb_net_test.to(device)

    if torch.cuda.device_count() > 1:
        emb_net_test = DataParallel(emb_net_test)
        print('multi-gpu')

    # Optimizer
    optimizer = torch.optim.Adam(emb_net_test.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    criterion = criterion.to(device)

    logger.info('#### Dataset Selection #### \n')
    logger.info('dataset:', dataset_name)
    logger.info('#### Network and learning parameters #### \n')
    logger.info('------------------------------------------ \n')
    logger.info('Model Name: ' + model_name + '\n')
    logger.info('Number of hidden layers: ' + str(layers) + '\n')
    logger.info('Hidden layer width: ' + str(hl_size) + '\n')
    logger.info('Embedding dimension: ' + str(dim) + '\n')
    logger.info('Learning rate: ' + str(learning_rate) + '\n')
    logger.info('Number of epochs: ' + str(epochs) + '\n')

    logger.info(' #### Training begins #### \n')
    logger.info('---------------------------\n')

    digits = int(math.ceil(math.log2(n)))
    bin_array = data_utils.get_binary_array(n, digits)

    trip_data = torch.tensor(bin_array[unseen_triplets])
    trip = trip_data.squeeze().to(device).float()

    # Training begins
    train_time = 0
    for ep in range(epochs):
        # Epoch is one pass over the dataset
        epoch_loss = 0

        for batch_ind, trips in enumerate(batch_triplet_indices_loader):
            sys.stdout.flush()
            trip = trips.squeeze().to(device).float()

            # Training time
            begin_train_time = time.time()
            # Forward pass
            embedded_a = emb_net_test(trip[:, :digits])
            embedded_p = emb_net_train(trip[:, digits:2 * digits])
            embedded_n = emb_net_train(trip[:, 2 * digits:])
            # Compute loss
            loss = criterion(embedded_a, embedded_p, embedded_n).to(device)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # End of training
            end_train_time = time.time()
            if batch_ind % 50 == 0:
                logger.info('Epoch: ' + str(ep) + ' Mini batch: ' + str(batch_ind) +
                            '/' + str(len(batch_triplet_indices_loader)) + ' Loss: ' + str(loss.item()))
                sys.stdout.flush()  # Prints faster to the out file
            epoch_loss += loss.item()
            train_time = train_time + end_train_time - begin_train_time

        # Log
        logger.info('Epoch ' + str(ep) + ' - Average Epoch Loss:  ' +
                    str(epoch_loss/len(batch_triplet_indices_loader)) + ' Training time ' +
                    str(train_time))
        sys.stdout.flush()  # Prints faster to the out file

        # Saving the results
        logger.info('Saving the models and the results')
        sys.stdout.flush()  # Prints faster to the out file

        os.makedirs('test_checkpoints', mode=0o777, exist_ok=True)
        model_path = 'test_checkpoints/' + \
                     experiment_name + \
                     '.pt'
        torch.save({
            'epochs': ep,
            'model_state_dict': emb_net_test.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss:': epoch_loss,
        }, model_path)

    # Compute the embedding of the data points.
    bin_array_test = data_utils.get_binary_array(test_n, digits)
    test_embeddings = emb_net_test(torch.Tensor(bin_array_test).cuda().float()).cpu().detach().numpy()
    train_embeddings = emb_net_train(torch.Tensor(bin_array).cuda().float()).cpu().detach().numpy()
    unseen_triplet_error, _ = data_utils.triplet_error_unseen(test_embeddings, train_embeddings, unseen_triplets)

    logger.info('Unseen triplet error is ' + str(unseen_triplet_error))
    return unseen_triplet_error
