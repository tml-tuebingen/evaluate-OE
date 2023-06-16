import torch
from torch.autograd import grad
import numpy as np
from preprocessing_utils.TripletData import triplet_error_torch
import time
import sys
from math import ceil

def soe_adam(triplets, n, dim, epochs, batch_size, learning_rate, device, logger, error_change_threshold=-1):
    # create a set of random vectors in the required embedding size
    random_embeddings = torch.rand(size=(n, dim), dtype=torch.float)
    X = torch.Tensor(random_embeddings).to(device)
    X.requires_grad = True
    triplet_num = triplets.shape[0]

    nmb_train_triplets_for_error = 100000

    # number of iterations to get through the dataset
    optimizer = torch.optim.Adam(params=[X], lr=learning_rate, amsgrad=True)
    batches = 1 if batch_size > triplet_num else triplet_num // batch_size
    loss_history = []
    triplet_error_history = []
    time_history = []

    triplets = torch.tensor(triplets).to(device).long()
    logger.info('Number of Batches = ' + str(batches))

    # Compute initial triplet error
    error_batch_indices = np.random.randint(triplet_num, size=min(nmb_train_triplets_for_error, triplet_num))
    triplet_error_history.append(triplet_error_torch(X, triplets[error_batch_indices, :])[0].item())

    # Compute initial loss
    epoch_loss = 0
    for batch_ind in range(batches):
        batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
        batch_xs = X[batch_trips, :]
        batch_loss = soe_loss(batch_xs[:, 0, :].squeeze(),
                            batch_xs[:, 1, :].squeeze(),
                            batch_xs[:, 2, :].squeeze()
                            )
        epoch_loss += batch_loss.item()
    loss_history.append(epoch_loss / triplets.shape[0])

    best_X = X
    total_time = 0
    time_to_best = 0
    best_triplet_error = triplet_error_history[0]
    for it in range(epochs):
        intermediate_time = time.time()
        epoch_loss = 0
        for batch_ind in range(batches):
            # extract the triplet indices
            batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]

            # update the embeddings of the indices only involved in the current batch
            batch_xs = X[batch_trips, :]

            # compute the loss and hinge it
            batch_loss = soe_loss(batch_xs[:, 0, :].squeeze(),
                            batch_xs[:, 1, :].squeeze(),
                            batch_xs[:, 2, :].squeeze()
                            )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        end_time = time.time()
        total_time += (end_time - intermediate_time)
        epoch_loss = epoch_loss / triplets.shape[0]

        # Record loss and time
        time_history.append(total_time)
        loss_history.append(epoch_loss)

        if it % 50 == 0 or it == epochs - 1:
            if error_change_threshold != -1:
                # Compute triplet error
                error_batch_indices = np.random.randint(triplet_num, size=min(nmb_train_triplets_for_error, triplet_num))
                triplet_error = triplet_error_torch(X, triplets[error_batch_indices, :])[0].item()
                triplet_error_history.append(triplet_error)
                if triplet_error < best_triplet_error - error_change_threshold:
                    best_X = X
                    best_triplet_error = triplet_error
                    time_to_best = total_time
                    logger.info('Found new best in Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error))
                logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error))
                sys.stdout.flush()
            else:
                best_X = X
                time_to_best = total_time
                logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss))
                sys.stdout.flush()

    return best_X.cpu().detach().numpy(), loss_history, triplet_error_history, time_to_best, time_history


def triplet_loss_adam(triplets, n, dim, iterations, bs, lr, device, logger):
    # create a set of random vectors in the required embedding size
    random_embeddings = torch.rand(size=(n, dim), dtype=torch.float)
    x = torch.Tensor(random_embeddings).to(device)
    x.requires_grad = True
    triplet_num = triplets.shape[0]

    # number of iterations to get through the dataset
    optimizer = torch.optim.Adam(params=[x], lr=lr, amsgrad=True)
    batches = triplet_num // bs
    epochs = iterations
    loss_history = []

    for it in range(epochs):
        epoch_loss = 0
        # np.random.shuffle(triplets)  # Shuffle the triplets at each epoch
        for batch_ind in range(batches):

            # extract the triplet indices
            batch_trips = triplets[batch_ind * bs: (batch_ind + 1) * bs, ]

            # update the embeddings of the indices only involved in the current batch
            batch_xs = x[batch_trips, :]

            # compute the loss and hinge it
            loss = triplet_loss(batch_xs[:, 0, :].squeeze(),
                                batch_xs[:, 1, :].squeeze(),
                                batch_xs[:, 2, :].squeeze()
                                )
            loss_p = loss[loss > 0]
            batch_loss = torch.sum(loss_p)
            optimizer.zero_grad()

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        trip_error, error_list = triplet_error_torch(x, triplets)
        epoch_loss = epoch_loss / triplets.shape[0]
        if it%100==0:
            logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(trip_error))

        loss_history.append(epoch_loss)
    final_embeddings = x.cpu().detach().numpy()
    return final_embeddings, loss_history, epoch_loss


def soe_sgd(triplets, n, dim, iterations, bs, lr, device, logger):

    # create a set of random vectors in the required embedding size
    x = torch.rand(size=(n, dim), dtype=torch.float).to(device)
    triplet_num = triplets.shape[0]

    # number of iterations to get through the dataset
    batches = triplet_num // bs
    epochs = iterations
    loss_history = []
    for it in range(epochs):
        epoch_loss = 0
        np.random.shuffle(triplets)  # Shuffle the triplets at each epoch
        for batch_ind in range(batches):

            # extract the triplet indices
            batch_trips = triplets[batch_ind * bs: (batch_ind + 1) * bs, ]
            # mark the embedding as required grad as they need to be updated
            x.requires_grad = True

            # update the embeddings of the indices only involved in the current batch
            batch_xs = x[batch_trips, :]

            # compute the loss and hinge it
            batch_loss = soe_loss(batch_xs[:, 0, :].squeeze(),
                            batch_xs[:, 1, :].squeeze(),
                            batch_xs[:, 2, :].squeeze()
                            )
            gradient = grad(batch_loss, x)[0]
            x.requires_grad = False
            x -= lr * gradient
            epoch_loss += batch_loss.item()
        trip_error = triplet_error_torch(x.cpu().detach().numpy(), triplets)
        epoch_loss = epoch_loss/triplets.shape[0]
        logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(trip_error[0]))
        loss_history.append(epoch_loss)
    final_embeddings = x.cpu().detach().numpy()
    return final_embeddings, loss_history, epoch_loss


def soe_loss(x_i, x_j, x_k, delta=1):
    """
    Description: This is the Triplet loss used in ICASPP paper.
    :param x_i:
    :param x_j:
    :param x_k:
    :param delta:
    :return:
    """
    loss = delta + torch.norm(x_i - x_j, p=2, dim=1) - torch.norm(x_i - x_k, p=2, dim=1)
    loss_p = loss[loss > 0]
    return torch.sum(loss_p)


def triplet_loss(x_i, x_j, x_k, delta=1):
    """
    Description: This is the loss used in the ICASPP paper by Bower
    :param x_i:
    :param x_j:
    :param x_k:
    :param delta:
    :return:
    """
    loss = delta + torch.norm(x_i - x_j, p=2, dim=1)**2 - torch.norm(x_i - x_k, p=2, dim=1)**2

    return loss

