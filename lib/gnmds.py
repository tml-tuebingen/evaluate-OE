import torch
from torch.autograd import grad
import numpy as np
from preprocessing_utils.TripletData import triplet_error_torch
import time
import sys
from math import ceil

def gnmds(triplets, n, dim, epochs, batch_size, learning_rate, device, logger, reg_lbda=0.0,
          error_change_threshold=-1):
    """
    params: error_change_threshold: The minimum change in triplet error we want. If the change is less, we save stop measuring the time
                                    If False, we return the last embedding, if true we report the best embedding wtr to this criterion.
    """

    # reg parameter
    reg_lbda = torch.FloatTensor([reg_lbda]).to(device)
    X = torch.rand(size=(n, dim), dtype=torch.float)
    K = torch.mm(X, torch.transpose(X, 0, 1)).to(device)*.1

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
        batch_loss = get_gnmds_k_loss(batch_trips, K, reg_lbda)
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
            batch_loss = get_gnmds_k_loss(batch_trips, K, reg_lbda)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            K.requires_grad = False
            # projection back onto semidefinite cone
            K = project_rank(K, dim)
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


def project(K, dim):
    D, U = torch.symeig(K, eigenvectors=True)  # will K be surely symmetric?
    pos_ind = (D > 0)
    return torch.mm(torch.mm(U[:, pos_ind], torch.diag(D[pos_ind])), torch.transpose(U[:, pos_ind], 0, 1))


def project_rank(K, dim):
    # by FORTE
    D, U = torch.symeig(K, eigenvectors=True)  # will K be surely symmetric?
    D = torch.max(D[-dim:], torch.Tensor([0]).to(K.device))
    return torch.mm(torch.mm(U[:, -dim:], torch.diag(D)), torch.transpose(U[:, -dim:], 0, 1))


def get_gnmds_k_loss(triplets, K, lbda):

    diag = torch.diag(K)[:, None]
    Dist = -2 * K + diag + torch.transpose(diag, 0, 1)
    loss = gnmds_k_hinge(Dist[triplets[:, 0], triplets[:, 1]].squeeze(),
                   Dist[triplets[:, 0], triplets[:, 2]].squeeze())

    return torch.sum(loss) + lbda * torch.trace(K)


def gnmds_k_hinge(d_ij, d_ik):
    device = d_ij.device
    loss = torch.max(d_ij - d_ik + 1, torch.FloatTensor([0]).to(device))
    return loss


# GNMDS over the embedding X   OLD VERSION
def gnmds_x(triplets, n, dim, iterations, bs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reg parameter
    lbda = torch.FloatTensor([0.05]).to(device)
    lr = torch.tensor([lr]).to(device)
    tol = 1e-7
    X = torch.randn(size=(n, dim), dtype=torch.float).to(device)*.1
    triplets = torch.tensor(triplets).to(device).long()
    triplet_num = triplets.shape[0]
    # lr = 1
    batches = triplet_num // bs
    best_cost = torch.Tensor([float('Inf')]).to(device)
    old_cost = best_cost
    best_X = X
    print('batches = ', batches)
    for it in range(iterations):
        print('epoch', it)

        perm = np.random.permutation(batches)
        for batch_ind in perm:
            batch_trips = triplets[batch_ind * bs: (batch_ind + 1) * bs, ]  # a batch of triplets
            X.requires_grad = True
            batch_loss = get_gnmds_x_loss(batch_trips,X,lbda)
            gradient = grad(batch_loss, X)[0]
            X.requires_grad = False
            X -= lr * gradient

        current_cost = get_gnmds_x_loss(triplets,X, lbda)
        print('overall cost', current_cost.cpu().numpy())

        #update learning rate
        if old_cost > current_cost + tol:
            lr = lr * 1.01
        else:
            lr = lr * .9
        print('learning rate: ', lr.cpu().numpy())

        # save best solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_X = X
        old_cost = current_cost

    return best_X.cpu().detach().numpy()


def get_gnmds_x_loss(triplets, X, lbda):
    batch_Xs = X[triplets, :]  # items involved in the triplet
    loss = gnmds_x_triplet_loss(batch_Xs[:, 0, :].squeeze(),
                        batch_Xs[:, 1, :].squeeze(),
                        batch_Xs[:, 2, :].squeeze())

    return torch.sum(loss) + lbda * torch.sum(X ** 2)  # likelihood of batch, without log


def gnmds_x_triplet_loss(x_i, x_j, x_k):
    device = x_i.device
    loss = torch.max(torch.sum((x_i - x_j)**2, dim=1) + 1 - torch.sum((x_i - x_k)**2, dim=1), torch.FloatTensor([0]).to(device))
    return loss


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    #print(A)
    _, s, V = torch.svd(A)

    H = torch.mm(torch.transpose(V,0,1), torch.mm(torch.diag(s), V))

    A2 = (A + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        print('is pd')
    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = torch.cholesky(B)
        return True
    except Exception as e:
        print(e)
        return False
