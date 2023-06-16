import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import data_select_utils

# mnist
data, labels = data_select_utils.select_dataset(dataset_name='mnist')
print('MNIST: ', data.shape, labels.shape)

# fmnist
data, labels = data_select_utils.select_dataset(dataset_name='fmnist')
print('FashionMNIST: ', data.shape, labels.shape)

# gmm
data, labels = data_select_utils.select_dataset(dataset_name='gmm', n_samples=128)
print('Gaussian Mixture Models: ', data.shape, labels.shape)

# emnist
# data, labels = data_select_utils.select_dataset(dataset_name='emnist')
# print('EMNIST: ', data.shape, labels.shape, np.unique(labels))

# kmnist
data, labels = data_select_utils.select_dataset(dataset_name='kmnist')
print('KMNIST: ', data.shape, labels.shape, np.unique(labels))

# usps
# data, labels = data_select_utils.select_dataset(dataset_name='usps')
# print('USPS: ', data.shape, labels.shape)

# cover_type
data, labels = data_select_utils.select_dataset(dataset_name='cover_type')
print('CoverType: ', data.shape, labels.shape)


# charfonts
data, labels = data_select_utils.select_dataset(dataset_name='charfonts')
print('CharFonts: ', data.shape, labels.shape)

# coil20
data, labels = data_select_utils.select_dataset(dataset_name='coil20')
print('Coil-20: ', data.shape, labels.shape)

# # news_groups
# data, labels = data_select_utils.select_dataset(dataset_name='news')
# print('News20: ', data.shape, labels.shape)

# char
data, labels = data_select_utils.select_dataset(dataset_name='char')
print('Char: ', data.shape, labels.shape)

# kdd
data, labels = data_select_utils.select_dataset(dataset_name='kdd_cup')
print('KDD: ', data.shape, labels.shape)

# uniform
data, labels = data_select_utils.select_dataset(dataset_name='blobs')
print('Blobs: ', data.shape, labels.shape)