import torch
from torch.autograd import grad
import numpy as np
from preprocessing_utils.TripletData import triplet_error_torch
import time
import sys
from math import ceil

def t_ste_adam(triplets, n, emb_dim, epochs, batch_size, learning_rate, device, logger, error_change_threshold=-1):
    random_embeddings = torch.rand(size=(n, emb_dim), dtype=torch.float)
    X = torch.Tensor(random_embeddings).to(device)
    X.requires_grad = True

    nmb_train_triplets_for_error = 100000

    # number of iterations to get through the dataset
    optimizer = torch.optim.Adam(params=[X], lr=learning_rate, amsgrad=True)
    triplet_num = triplets.shape[0]
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
        batch_loss = -1 * get_t_ste_loss(batch_trips, X, emb_dim)
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
            batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets

            batch_loss = -1 * get_t_ste_loss(batch_trips, X, emb_dim)

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


def get_t_ste_loss(triplets, X, emb_dim):
    prob = t_ste_prob(X[triplets, :][:, 0, :].squeeze(),
                    X[triplets, :][:, 1, :].squeeze(),
                    X[triplets, :][:, 2, :].squeeze(),
                      emb_dim - 1)
    return torch.sum(torch.log(prob))


def t_ste_prob(x_i, x_j, x_k, alpha=1):

    nom = (1 + torch.sum((x_i - x_j)**2 / alpha, dim=1))**(-(alpha+1)/2)
    denom = (1 + torch.sum((x_i - x_j)**2 / alpha, dim=1))**(-(alpha+1)/2) + \
            (1 + torch.sum((x_i - x_k)**2 / alpha, dim=1))**(-(alpha+1)/2) + 1e-15
    return nom / denom

#
# def t_ste(triplets, n, dim, epochs, batch_size, learning_rate, device, logger):
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     X = torch.rand(size=(n, dim), dtype=torch.float).to(device)
#     triplet_num = triplets.shape[0]
#     triplets = torch.tensor(triplets).to(device)
#
#     batches = triplet_num // batch_size
#     logger.info('batches = ' + str(batches))
#     for it in range(epochs):
#         # print('epoch', it)
#         epoch_loss = 0
#         perm = np.random.permutation(batches)
#         for batch_ind in perm:
#             batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
#             X.requires_grad = True
#             batch_Xs = X[batch_trips, :]  # items involved in the triplet
#             prob = t_ste_prob(batch_Xs[:, 0, :].squeeze(),
#                               batch_Xs[:, 1, :].squeeze(),
#                               batch_Xs[:, 2, :].squeeze(),
#                               dim - 1)
#
#             batch_prob = torch.sum(torch.log(prob))  # likelihood of batch, without log
#             gradient = grad(batch_prob, X)[0]
#             X.requires_grad = False
#             X += learning_rate * gradient
#             epoch_loss += batch_prob.item()
#         trip_error = triplet_error(X.cpu().detach().numpy(), triplets)
#         logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(trip_error[0]))
#
#     return X.cpu().detach().numpy(), epoch_loss


