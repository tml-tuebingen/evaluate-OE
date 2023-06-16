from sklearn.neighbors import NearestNeighbors
import argparse
import numpy as np
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
import os

def parse_args():
    # If embedding of a model is not already saved. Then use this code.
    parser = argparse.ArgumentParser(description='Input path of the model')
    parser.add_argument('-d', '--dataset_name', dest='dataset', default='mnist', type=str, required=True, help='Name of the dataset')
    parser.add_argument('-s', '--save_dir', dest='save_dir', type=str, required=True, help='Directory to Save Neighbours')

    args = parser.parse_args()
    return args

def main(args):
    # check the saving directory
    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)

    # choose the dataset
    if args.dataset =='mnist':
        mnist = MNIST(root="./downloads", download=True)
        data = mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        print(vec_data.shape)
    else:
        mnist = FashionMNIST(root="./downloads", download=True)
        data = mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        print(vec_data.shape)

    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)

    neighbours_getter = NearestNeighbors(n_neighbors=50, algorithm='ball_tree', n_jobs=8)
    neighbours_getter.fit(vec_data)

    _, indices = neighbours_getter.kneighbors(vec_data)

    save_path = os.path.join(args.save_dir, args.dataset + '_neigbours_indices.npy')
    np.save(save_path, indices)


if __name__ == '__main__':
    main(parse_args())