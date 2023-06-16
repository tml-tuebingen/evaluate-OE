import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import data_select_utils

print('Dataset Name -  Samples - dataset_id')
# birch1
data, labels = data_select_utils.select_dataset(dataset_name='birch1', n_samples=-1)
print('birch1 - ', data.shape[0], ' - ', 'birch1')

# birch3
data, labels = data_select_utils.select_dataset(dataset_name='birch3', n_samples=-1)
print('birch3 - ', data.shape[0], ' - ', 'birch3')

# worms
data, labels = data_select_utils.select_dataset(dataset_name='worms', n_samples=-1)
print('Worms - ', data.shape[0], ' - ', 'worms')

# blobs
data, labels = data_select_utils.select_dataset(dataset_name='blobs')
print('blobs - ', data.shape[0], ' - ', 'blobs')

# compound
data, labels = data_select_utils.select_dataset(dataset_name='compound', n_samples=-1)
print('Compound - ', data.shape[0], ' - ', 'compound')

# pathbased
data, labels = data_select_utils.select_dataset(dataset_name='path-based', n_samples=-1)
print('Path-Based - ', data.shape[0], ' - ', 'path-based')

# spiral
data, labels = data_select_utils.select_dataset(dataset_name='spiral', n_samples=-1)
print('Spiral - ', data.shape[0], ' - ', 'spiral')

# mnist
data, labels = data_select_utils.select_dataset(dataset_name='mnist', n_samples=-1)
print('MNIST - ', data.shape[0], ' - ', 'mnist')

# fmnist
data, labels = data_select_utils.select_dataset(dataset_name='fmnist', n_samples=-1)
print('FashionMNIST - ', data.shape[0], ' - ', 'fmnist')

# gmm
data, labels = data_select_utils.select_dataset(dataset_name='gmm', n_samples=128)
print('Gaussian Mixture Models - ', data.shape[0], ' - ', 'gmm')

# kmnist
data, labels = data_select_utils.select_dataset(dataset_name='kmnist', n_samples=-1)
print('KMNIST - ', data.shape[0], ' - ', 'kmnist')

# usps
data, labels = data_select_utils.select_dataset(dataset_name='usps', n_samples=-1)
print('USPS - ', data.shape[0], ' - ', 'usps')

# cover_type
data, labels = data_select_utils.select_dataset(dataset_name='cover_type', n_samples=-1)
print('CoverType - ', data.shape[0], ' - ', 'cover_type')

# char
data, labels = data_select_utils.select_dataset(dataset_name='char', n_samples=-1)
print('Char - ', data.shape[0], ' - ', 'char')
