from torchvision.datasets import MNIST, FashionMNIST, EMNIST, USPS, KMNIST
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups, fetch_covtype, fetch_kddcup99
from sklearn.datasets import make_blobs, make_circles, make_moons
from preprocessing_utils import feature_tranformers
import os
import matplotlib.pyplot as plt

file_path = my_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def select_dataset(dataset_name, input_dim=2, n_samples=10000):
    """
    :params n_samples: number of points returned. If 0, all datapoints will be returned. For artificial data, it will throw an error.
    """
    if dataset_name == 'fmnist':
        f_mnist = FashionMNIST(root="./downloads", download=True)
        data = f_mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = f_mnist.targets.numpy()
    elif dataset_name == 'emnist':
        f_mnist = EMNIST(root="./downloads", download=True, split='byclass')
        data = f_mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = f_mnist.targets.numpy()
    elif dataset_name == 'kmnist':
        f_mnist = KMNIST(root="./downloads", download=True)
        data = f_mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = f_mnist.targets.numpy()
    elif dataset_name == 'usps':
        f_mnist = USPS(root="./downloads", download=True)
        data = f_mnist.data
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = np.float32(f_mnist.targets)
    elif dataset_name == 'news':
        newsgroups_train = fetch_20newsgroups(data_home='./downloads', subset='train',
                                              remove=('headers', 'footers', 'quotes'))
        vectorizer = TfidfVectorizer()
        vec_data = vectorizer.fit_transform(newsgroups_train.data).toarray()
        vec_data = np.float32(vec_data)
        labels = newsgroups_train.target
        labels = np.float32(labels)
    elif dataset_name == 'cover_type':
        file_name = file_path + "/datasets/covtype.data"
        train_data = np.array(pd.read_csv(file_name, sep=','))
        vec_data = np.float32(train_data[:, :-1])
        labels = np.float32(train_data[:, -1])
    elif dataset_name == 'char':
        digits = datasets.load_digits()
        data = digits.images.reshape((len(digits.images), -1))
        vec_data = np.float32(data)
        labels = digits.target

    elif dataset_name == 'charx':
        file_name = file_path + "/datasets/char_x.npy"
        data = np.load(file_name, allow_pickle=True)
        vec_data, labels = data[0], data[1]

    elif dataset_name == 'kdd_cup':
        cover_train = fetch_kddcup99(data_home='./downloads', download_if_missing=True)
        vec_data = cover_train.data
        string_labels = cover_train.target
        vec_data, labels = feature_tranformers.vectorizer_kdd(data=vec_data, labels=string_labels)
    elif dataset_name == 'aggregation':
        file_name = file_path + "/2d_data/Aggregation.csv"
        a = np.array(pd.read_csv(file_name, sep=';', header=None))
        vec_data = a[:, 0:2]
        labels = a[:, 2]
    elif dataset_name == 'compound':
        file_name = file_path + "/2d_data/Compound.txt"
        a = np.array(pd.read_csv(file_name, sep='\t', header=None))
        vec_data = a[:, 0:2]
        labels = a[:, 2]
    elif dataset_name == 'd31':
        file_name = file_path + "/2d_data/D31.txt"
        a = np.array(pd.read_csv(file_name, sep='\t', header=None))
        vec_data = a[:, 0:2]
        labels = a[:, 2]
    elif dataset_name == 'flame':
        file_name = file_path + "/2d_data/flame.txt"
        a = np.array(pd.read_csv(file_name, sep='\t', header=None))
        vec_data = a[:, 0:2]
        labels = a[:, 2]
    elif dataset_name == 'path_based':
        file_name = file_path + "/2d_data/pathbased.txt"
        a = np.array(pd.read_csv(file_name, sep='\t', header=None))
        vec_data = a[:, 0:2]
        labels = a[:, 2]
    elif dataset_name == 'r15':
        file_name = file_path + "/2d_data/R15.txt"
        a = np.array(pd.read_csv(file_name, sep='\t', header=None))
        vec_data = a[:, 0:2]
        labels = a[:, 2]
    elif dataset_name == 'spiral':
        file_name = file_path + "/2d_data/spiral.txt"
        a = np.array(pd.read_csv(file_name, sep='\t', header=None))
        vec_data = a[:, 0:2]
        labels = a[:, 2]
    elif dataset_name == 'birch1':
        file_name = file_path + "/2d_data/birch1.txt"
        a = np.array(pd.read_csv(file_name, delimiter=r"\s+", header=None))
        vec_data = a[:, :]
        labels = np.ones((vec_data.shape[0]))
    elif dataset_name == 'birch2':
        file_name = file_path + "/2d_data/birch2.txt"
        a = np.array(pd.read_csv(file_name, delimiter=r"\s+", header=None))
        vec_data = a[:, :]
        labels = np.ones((vec_data.shape[0]))
    elif dataset_name == 'birch3':
        file_name = file_path + "/2d_data/birch3.txt"
        a = np.array(pd.read_csv(file_name, delimiter=r"\s+", header=None))
        vec_data = a[:, :]
        labels = np.ones((vec_data.shape[0]))
    elif dataset_name == 'worms':
        file_name = file_path + "/2d_data/worms/worms_2d.txt"
        a = np.array(pd.read_csv(file_name, sep=' ', header=None))
        vec_data = a[:, :]
        labels = np.ones((vec_data.shape[0]))
    elif dataset_name == 't48k':
        file_name = file_path + "/2d_data/t4.8k.txt"
        a = np.array(pd.read_csv(file_name, sep=' '))
        vec_data = a[1:, :]
        labels = np.ones((vec_data.shape[0]))
    elif dataset_name == 'moons':
        data, labels = make_moons(n_samples=5000)
        vec_data = np.float32(data)
        labels = np.float32(labels)
    elif dataset_name == 'circles':
        data, labels = make_circles(n_samples=5000)
        vec_data = np.float32(data)
        labels = np.float32(labels)
    elif dataset_name == 'blobs':
        data, labels = make_blobs(n_samples=n_samples, centers=3)
        vec_data = np.float32(data)
        labels = np.float32(labels)
    elif dataset_name == 'gmm':
        mean_1 = np.zeros(input_dim)
        mean_2 = 100 * np.ones(input_dim)
        cov = np.eye(input_dim)
        data_1 = np.random.multivariate_normal(mean_1, cov, int(n_samples / 2))
        labels_1 = np.ones(int(n_samples / 2))
        labels_2 = 2 * np.ones(int(n_samples / 2))
        data_2 = np.random.multivariate_normal(mean_2, cov, int(n_samples / 2))
        vec_data = np.concatenate([data_1, data_2], axis=0)
        labels = np.concatenate([labels_1, labels_2], axis=0)
    elif dataset_name == 'uniform':
        vec_data = np.random.uniform(0, 1, size=(n_samples, input_dim)) * 10
        labels = np.ones(n_samples)
    elif dataset_name == 'mnist_pc':
        d_mnist = MNIST(root="./downloads", download=True)
        mnist = d_mnist.data.numpy()
        data = np.float32(np.reshape(mnist, (mnist.shape[0], -1)))
        pca_data = PCA(n_components=input_dim).fit_transform(data)
        n_indices = np.random.randint(pca_data.shape[0], size=n_samples)
        vec_data = pca_data[n_indices]
        labels = d_mnist.targets.numpy()[n_indices]
    elif dataset_name == 'usps_pc':
        d_mnist = USPS(root="./downloads", download=True)
        mnist = d_mnist.data
        data = np.float32(np.reshape(mnist, (mnist.shape[0], -1)))
        pca_data = PCA(n_components=input_dim).fit_transform(data)
        n_indices = np.random.randint(pca_data.shape[0], size=n_samples)
        vec_data = pca_data[n_indices]
        labels = np.float32(d_mnist.targets)
    elif dataset_name == 'char_pc':
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        data = np.float32(data)
        targets = digits.target
        pca_data = PCA(n_components=input_dim).fit_transform(data)
        n_indices = np.random.randint(pca_data.shape[0], size=n_samples)
        vec_data = pca_data[n_indices]
        labels = targets
    else:
        d_mnist = MNIST(root="./downloads", download=True)
        data = d_mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = d_mnist.targets.numpy()

    # translate all the data to the origin
    vec_data = vec_data - np.mean(vec_data, 0)
    norm1 = np.linalg.norm(vec_data)
    if norm1 == 0:
        raise ValueError("Input matrices must contain >1 unique points")
    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    vec_data /= norm1

    if 0 < n_samples < vec_data.shape[0]:
        rand_indices = np.random.choice(vec_data.shape[0], size=(n_samples,), replace=False)
        return vec_data[rand_indices], labels[rand_indices]
    else:
        rand_indices = np.random.choice(vec_data.shape[0], size=(vec_data.shape[0],), replace=False)
        return vec_data[rand_indices], labels[rand_indices]


def select_test_dataset(dataset_name, testing=False):
    """
    Selects a dataset from the options below
    Parameters
    ----------
    dataset_name: dataset name given as string: 'fmnist'
    testing: testing flag. If testing is True, then the function returns 1000 samples only

    Returns
    -------
    vec_data: the dataset as a numpy array. The dimensions are N X D where N is the number of samples in the data and
    D is the dimensions of the feature vector.
    labels: the labels of the samples. The dimensions are N X 1 where N is the number of samples in the data and
    1 is label of the sample.
    """
    if dataset_name == 'fmnist':
        f_mnist = FashionMNIST(root="./downloads", train=False, download=True)
        data = f_mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = f_mnist.targets.numpy()
    elif dataset_name == 'usps':
        f_mnist = USPS(root="./downloads", train=False, download=True)
        data = f_mnist.data
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = np.float32(f_mnist.targets)
    elif dataset_name == 'char':
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        vec_data = np.float32(data)
        labels = digits.target
    elif dataset_name == 'charx':
        file_name = file_path + "/datasets/char_x.npy"
        data = np.load(file_name, allow_pickle=True)
        vec_data, labels = data[2], data[3]
    else:
        print('The dataset you asked for is not available. Gave you MNIST instead.')
        d_mnist = MNIST(root="./downloads", train=False, download=True)
        data = d_mnist.data.numpy()
        vec_data = np.reshape(data, (data.shape[0], -1))
        vec_data = np.float32(vec_data)
        labels = d_mnist.targets.numpy()

    if testing:
        return vec_data[:1000], labels[:1000]
    else:
        return vec_data, labels


if __name__ == '__main__':

    vec_data, labels = select_dataset('aggregation', n_samples=-1)
    print(vec_data.shape)
    print(vec_data[0])

    # fig, ax= plt.subplots(1,1, figsize=(12,12))
    # ax.scatter(vec_data[:,0], vec_data[:,1])
    # plt.show()
