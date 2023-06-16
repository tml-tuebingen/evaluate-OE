import numpy as np
from preprocessing_utils.oracle import Oracle

def get_mixture_of_gaussians(samples, mean_1=[5, 5], mean_2=[10, 10], cov=[[1, 0], [0, 1]]):
    x1, y1 = np.random.multivariate_normal(mean_1, cov, size=int(samples/2)).T
    x2, y2 = np.random.multivariate_normal(mean_2, cov, size=int(samples/2)).T

    clusters = []
    for each in zip(x1, y1):
        clusters.append(each)

    for each in zip(x2, y2):
        clusters.append(each)
    clusters_data = clusters
    return np.asarray(clusters_data)


def gen_triplet_indices(n, num_trips):
    """
    Description: Generate random triplet indices
    :param n: #points in the data
    :param num_trips: #triplets
    :return: random triplet indices. Shape (#triplets, 3)
    """
    all_triplet_indices = np.random.randint(n, size=(num_trips, 3))

    return all_triplet_indices


data = get_mixture_of_gaussians(samples=10000)
print(data.shape)

# lets make an oracle
my_oracle = Oracle(data=data)
# generate triplet indices

trip_indices = gen_triplet_indices(n=data.shape[0], num_trips=100)

query_result = my_oracle.bulk_oracle(trip_indices, batch_size=20)

print(query_result)
