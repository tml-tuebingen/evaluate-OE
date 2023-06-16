import torch
from torch.autograd import grad
import numpy as np
from preprocessing_utils.TripletData import triplet_error_torch
import time
import sys
from math import ceil

def ckl_k(triplets, n, dim, epochs, batch_size, learning_rate, device, logger, reg_lbda=0.0, mu=0.1,
          error_change_threshold=-1):
    """
    params: error_change_threshold: The minimum change in triplet error we want. If the change is less, we save stop measuring the time
                                    If False, we return the last embedding, if true we report the best embedding wtr to this criterion.
    """

    reg_lbda = torch.Tensor([reg_lbda]).to(device)
    mu = torch.Tensor([mu]).to(device)
    X = torch.rand(size=(n, dim), dtype=torch.float)
    K = torch.mm(X, torch.transpose(X, 0, 1)).to(device) * .1

    nmb_train_triplets_for_error = 100000

    # number of iterations to get through the dataset
    optimizer = torch.optim.Adam(params=[K], lr=learning_rate, amsgrad=True)
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
        batch_loss = -1*get_ckl_k_loss(batch_trips, K, reg_lbda, mu=mu)
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

            batch_loss = -1*get_ckl_k_loss(batch_trips, K, reg_lbda, mu=mu)

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


def ckl_x(triplets, n, dim, epochs, batch_size, learning_rate, device, logger, mu=0.01,
          error_change_threshold=-1):

    mu = torch.Tensor([mu]).to(device)
    X = torch.rand(size=(n, dim), dtype=torch.float).to(device)*0.1
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
        batch_loss = -1*get_ckl_x_loss(batch_trips, X, mu=mu)
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

            batch_loss = -1*get_ckl_x_loss(batch_trips, X, mu=mu)

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

def ckl_k_line_search(triplets, n, dim, epochs, batch_size, learning_rate, device, logger, reg_lbda=0.0, mu=0.1,
          error_change_threshold=-1):
    """
    params: error_change_threshold: The minimum change in triplet error we want. If the change is less, we save stop measuring the time
                                    If False, we return the last embedding, if true we report the best embedding wtr to this criterion.
    """

    reg_lbda = torch.Tensor([reg_lbda]).to(device)
    mu = torch.Tensor([mu]).to(device)
    X = torch.rand(size=(n, dim), dtype=torch.float)
    K = torch.mm(X, torch.transpose(X, 0, 1)).to(device) * .1

    nmb_train_triplets_for_error = 100000

    # number of iterations to get through the dataset
    learning_rate = torch.tensor([learning_rate]).to(device) # learning rate
    epsilon = torch.tensor([0.001]).to(device)  # stopping criterion
    rho = torch.tensor([0.5]).to(device)  # backtracking line search parameter
    c1 = torch.tensor([0.0001]).to(device)  # Amarijo stopping condition parameter
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
        batch_loss = -1*get_ckl_k_loss(batch_trips, K, reg_lbda, mu=mu)
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

            old_loss = -1*get_ckl_k_loss(batch_trips, K, reg_lbda, mu=mu)
            gradient = grad(old_loss, K)[0]
            K.requires_grad = False
            epoch_loss += old_loss.item()

            #old_loss = -1*get_ckl_k_loss(batch_trips, K, reg_lbda, mu=mu)
            new_K = project_rank(K - learning_rate * gradient, dim)
            new_loss = -1*get_ckl_k_loss(batch_trips, new_K, reg_lbda, mu=mu)
            diff = new_K - K
            beta = rho
            norm_grad = torch.norm(gradient, dim=0)
            norm_grad_sq = torch.sum(norm_grad)
            inner_t = 0
            while new_loss > old_loss - c1 * learning_rate * norm_grad_sq and inner_t < 10:
                beta = beta * beta
                new_loss = -1*get_ckl_k_loss(triplets, K + beta * diff, reg_lbda, mu=mu)
                inner_t += 1
            if inner_t > 0:
                print('using line search')
                learning_rate = torch.max(torch.FloatTensor([.1]).to(device), learning_rate * rho)
                print(learning_rate.cpu())

            K = project_rank(K + beta * diff, dim)

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


def get_ckl_k_loss(triplets, K, lbda, mu=0.1):
    diag = torch.diag(K)[:, None]
    Dist = -2 * K + diag + torch.transpose(diag, 0, 1)
    prob = ckl_prob_dist(Dist[triplets[:, 0], triplets[:, 1]].squeeze(),
                         Dist[triplets[:, 0], triplets[:, 2]].squeeze(), mu=mu)

    return torch.sum(torch.log(prob)) - lbda * torch.trace(K)


def get_ckl_x_loss(triplets, X, mu=0.1):
    prob = ckl_prob(X[triplets, :][:, 0, :].squeeze(),
                    X[triplets, :][:, 1, :].squeeze(),
                    X[triplets, :][:, 2, :].squeeze(), mu=mu)

    return torch.sum(torch.log(prob))


def ckl_prob(x_i, x_j, x_k, mu):
    nom = torch.norm(x_i - x_k, p=2, dim=1)**2 + mu
    denom = torch.norm(x_i - x_j, p=2, dim=1)**2 + torch.norm(x_i - x_k, p=2, dim=1)**2 + 2*mu
    return nom / denom

def ckl_prob_dist(d_ij, d_ik, mu=0.1):
    nom = d_ik + mu
    denom = d_ij + d_ik + 2 * mu
    return nom / denom

# so far, this is my method. Might change it later to the forte version
# which just takes the highest dim eigenvalues.


def project_rank(K, dim):
    D, U = torch.symeig(K, eigenvectors=True)  # will K be surely symmetric?
    D = torch.max(D[-dim:], torch.Tensor([0]).to(K.device))
    return torch.mm(torch.mm(U[:, -dim:], torch.diag(D)), torch.transpose(U[:, -dim:], 0, 1))





