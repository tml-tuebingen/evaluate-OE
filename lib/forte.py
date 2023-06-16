import torch
from torch.autograd import grad
import numpy as np
from preprocessing_utils.TripletData import triplet_error_torch
import time
import sys
from math import ceil

def forte_adam(triplets, n, dim, epochs, batch_size, learning_rate, device, logger, error_change_threshold=-1):

    # creation of first psd matrix K. Forte does it differently, but doesnt matter
    X = torch.randn(size=(n, dim), dtype=torch.float)
    K = torch.mm(X, torch.transpose(X, 0, 1)).to(device)*.1

    nmb_train_triplets_for_error = 100000

    optimizer = torch.optim.Adam(params=[K], lr=learning_rate, amsgrad=True)
    triplet_num = triplets.shape[0]
    batches = 1 if batch_size > triplet_num else triplet_num // batch_size
    loss_history = []
    triplet_error_history = []
    time_history = []

    triplets = torch.tensor(triplets).to(device).long()
    logger.info('Number of Batches: ' + str(batches))

    # Compute initial triplet error
    error_batch_indices = np.random.randint(triplet_num, size=min(nmb_train_triplets_for_error, triplet_num))
    triplet_error_history.append(triplet_error_torch(X, triplets[error_batch_indices, :])[0].item())

    # Compute initial loss
    epoch_loss = 0
    for batch_ind in range(batches):
        batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
        batch_loss = get_loss(batch_trips, K)
        epoch_loss += batch_loss.item()
    loss_history.append(epoch_loss / triplets.shape[0])

    best_K = K
    total_time = 0
    time_to_best = 0
    best_triplet_error = triplet_error_history[0]
    for it in range(epochs):
        intermediate_time = time.time()
        epoch_loss = 0
        for batch_ind in range(batches):
            batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets

            K.requires_grad = True
            batch_loss = get_loss(batch_trips, K)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            K.requires_grad = False
            epoch_loss += batch_loss.item()
            # projection back onto semidefinite cone
            K = project_rank(K, dim)

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
                # SVD to get embedding
                U, s, _ = torch.svd(K)
                X = torch.mm(U[:, :dim], torch.diag(torch.sqrt(s[:dim])))
                triplet_error = triplet_error_torch(X, triplets[error_batch_indices, :])[0].item()
                triplet_error_history.append(triplet_error)
                logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error))
                sys.stdout.flush()
                if triplet_error < best_triplet_error - error_change_threshold:
                    best_K = K
                    best_triplet_error = triplet_error
                    time_to_best = total_time
                    logger.info('Found new best in Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error))
            else:
                best_K = K
                time_to_best = total_time
                logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss))
                sys.stdout.flush()

    # SVD to get embedding
    U, s, _ = torch.svd(best_K)
    X = torch.mm(U[:, :dim], torch.diag(torch.sqrt(s[:dim])))
    return X.cpu().detach().numpy(), loss_history, triplet_error_history, time_to_best, time_history

def rank_d_pgd(triplets, n, dim, epochs, batch_size, learning_rate, device, logger, error_change_threshold=-1):

    # creation of first psd matrix K. Forte does it differently, but doesnt matter
    X = torch.randn(size=(n, dim), dtype=torch.float)
    K = torch.mm(X, torch.transpose(X, 0, 1)).to(device)*.1

    nmb_train_triplets_for_error = 100000
    learning_rate = torch.tensor([learning_rate]).to(device)  # learning rate
    rho = torch.tensor([0.5]).to(device)  # backtracking line search parameter
    c1 = torch.tensor([0.0001]).to(device)  # Amarijo stopping condition parameter
    triplet_num = triplets.shape[0]
    batches = 1 if batch_size > triplet_num else triplet_num // batch_size
    triplets = torch.tensor(triplets).to(device).long()
    loss_history = []
    triplet_error_history = []
    time_history = []

    logger.info('Number of Batches: ' + str(batches))

    # Compute initial triplet error
    error_batch_indices = np.random.randint(triplet_num, size=min(nmb_train_triplets_for_error, triplet_num))
    triplet_error_history.append(triplet_error_torch(X, triplets[error_batch_indices, :])[0].item())

    # Compute initial loss
    epoch_loss = 0
    for batch_ind in range(batches):
        batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
        batch_loss = get_loss(batch_trips, K)
        epoch_loss += batch_loss.item()
    loss_history.append(epoch_loss / triplets.shape[0])

    best_K = K
    total_time = 0
    time_to_best = 0
    best_triplet_error = triplet_error_history[0]
    for it in range(epochs):
        intermediate_time = time.time()
        epoch_loss = 0
        for batch_ind in range(batches):
            batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets

            K.requires_grad = True
            old_loss = get_loss(batch_trips, K)
            epoch_loss += old_loss.item()
            gradient = grad(old_loss, K)[0]
            K.requires_grad = False
            new_K = project_rank(K - learning_rate * gradient, dim)
            new_loss = get_loss(batch_trips, new_K)
            diff = new_K - K
            beta = rho
            norm_grad = torch.norm(gradient, dim=0)
            norm_grad_sq = torch.sum(norm_grad)
            inner_t = 0
            while new_loss > old_loss - c1 * learning_rate * norm_grad_sq and inner_t < 10:
                beta = beta*beta
                new_loss = get_loss(batch_trips, K + beta * diff)
                inner_t += 1
            if inner_t > 0:
                learning_rate = torch.max(torch.FloatTensor([0.1]).to(device), learning_rate * rho)

            K = project_rank(K+beta*diff, dim)

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
                # SVD to get embedding
                U, s, _ = torch.svd(K)
                X = torch.mm(U[:, :dim], torch.diag(torch.sqrt(s[:dim])))
                triplet_error = triplet_error_torch(X, triplets[error_batch_indices, :])[0].item()
                triplet_error_history.append(triplet_error)
                if triplet_error < best_triplet_error - error_change_threshold:
                    best_K = K
                    best_triplet_error = triplet_error
                    time_to_best = total_time
                    logger.info('Found new best in Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error))

                logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error))
                sys.stdout.flush()
            else:
                best_K = K
                time_to_best = total_time
                logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss))
                sys.stdout.flush()

    # SVD to get embedding
    U, s, _ = torch.svd(best_K)
    X = torch.mm(U[:, :dim], torch.diag(torch.sqrt(s[:dim])))
    return X.cpu().detach().numpy(), loss_history, triplet_error_history, time_to_best, time_history


# so far, this is my method. Might change it later to the forte version
# which just takes the highest dim eigenvalues.
def project(K, dim):
    D, U = torch.symeig(K, eigenvectors=True)  # will K be surely symmetric?
    pos_ind = (D > 0)
    return torch.mm(torch.mm(U[:, pos_ind], torch.diag(D[pos_ind])), torch.transpose(U[:, pos_ind], 0, 1))


def project_rank(K, dim):
    D, U = torch.symeig(K, eigenvectors=True)  # will K be surely symmetric?
    D = torch.max(D[-dim:], torch.Tensor([0]).to(K.device))
    return torch.mm(torch.mm(U[:, -dim:], torch.diag(D)), torch.transpose(U[:, -dim:], 0, 1))


def get_loss(triplets, K):
    diag = torch.diag(K)[:, None]
    Dist = -2 * K + diag + torch.transpose(diag, 0, 1)
    return torch.sum(forte_loss(Dist[triplets[:, 0], triplets[:, 1]].squeeze(),
                         Dist[triplets[:, 0], triplets[:, 2]].squeeze()))


def forte_loss(d_ij, d_ik):
    loss = torch.log(1+torch.exp(d_ij - d_ik))
    return loss




