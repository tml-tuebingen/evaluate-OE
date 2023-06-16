from preprocessing_utils import data_select_utils

# birch1
data, labels = data_select_utils.select_dataset(dataset_name='birch1')
print('birch1: ', data.shape, labels.shape)

# birch2
data, labels = data_select_utils.select_dataset(dataset_name='birch2')
print('birch2: ', data.shape, labels.shape)

# birch3
data, labels = data_select_utils.select_dataset(dataset_name='birch3')
print('birch3: ', data.shape, labels.shape)


# worms
data, labels = data_select_utils.select_dataset(dataset_name='worms')
print('worms: ', data.shape, labels.shape)

# d31
data, labels = data_select_utils.select_dataset(dataset_name='d31')
print('d31: ', data.shape, labels.shape)

# t48k
data, labels = data_select_utils.select_dataset(dataset_name='t48k')
print('t48k: ', data.shape, labels.shape)
