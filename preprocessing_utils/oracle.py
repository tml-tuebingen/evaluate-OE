import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations


class Oracle:

    def __init__(self, data):
        self.data_store = torch.Tensor(data)

    def query(self, a, b, c):
        # original indices required
        a_b = torch.norm(self.data_store[a, :] - self.data_store[b, :], p=2)
        a_c = torch.norm(self.data_store[a, :] - self.data_store[c, :], p=2)
        return -1 if a_b < a_c else 1

    def bulk_oracle(self, random_triplet_indices, batch_size):
        """
        Description: Generate triplets at once in batches so we won't run into memory issues and training takes less time.
        # TODO: There is redundancy in the code, remove it.
        :param data: 2D numpy array with shape (#points, #dimensions)
        :param random_triplet_indices: A set of triplets with random integers in [#points]. Shape (#triplets, 3)
        :param batch_size: Arbitrary integer, typically chosen as 10000 or 50000
        :return: Query Result: (-1, 0, +1)) . Shape (#triplets, 3)
        """
        num_triplets = random_triplet_indices.shape[0]  # Compute the number of triplets.
        number_of_batches = np.int(np.ceil(num_triplets / batch_size))  # Number of batches
        query_result = torch.zeros(size=(num_triplets,))  # Initializing the triplet set
        for i in range(number_of_batches):
            if i == (number_of_batches - 1):
                indices = random_triplet_indices[(i * batch_size):, :]

                d1 = torch.norm(self.data_store[indices[:, 0], :] - self.data_store[indices[:, 1], :], dim=1, p=2)
                d2 = torch.norm(self.data_store[indices[:, 0], :] - self.data_store[indices[:, 2], :], dim=1, p=2)

                query_result[(i * batch_size):] = torch.Tensor(np.sign(d1 - d2))

            else:
                indices = random_triplet_indices[(i * batch_size):((i + 1) * batch_size), :]

                d1 = torch.norm(self.data_store[indices[:, 0], :] - self.data_store[indices[:, 1], :], dim=1, p=2)
                d2 = torch.norm(self.data_store[indices[:, 0], :] - self.data_store[indices[:, 2], :], dim=1, p=2)

                query_result[(i * batch_size):((i + 1) * batch_size)] = torch.Tensor(np.sign(d1 - d2))
        query_result = query_result.cpu().detach().numpy()
        return query_result

    def bulk_query_pq(self, un_embedded_point_index, inital_anchor_index, other_embeds_index, indicator=0):
        # compute the distance between anchor and the un embedded point
        point_to_init_anchor_distance = torch.norm(self.data_store[un_embedded_point_index] -
                                                   self.data_store[inital_anchor_index])
        compare_dist = point_to_init_anchor_distance
        distance_index = []
        for each_index in other_embeds_index:
            distance_between_anchor_and_each_index = torch.norm(self.data_store[inital_anchor_index] -
                                                                self.data_store[each_index])
            if distance_between_anchor_and_each_index > compare_dist and indicator == 1:
                distance_index.append([distance_between_anchor_and_each_index, each_index])
            if distance_between_anchor_and_each_index < compare_dist and indicator == 0:
                distance_index.append([distance_between_anchor_and_each_index, each_index])
        p = min(distance_index, key=lambda i: i[1])[0]
        q = max(distance_index, key=lambda i: i[1])[0]
        return p if indicator == 1 else q

    @staticmethod
    def gen_landmark_indices(n, k, c):
        """
        Description: Takes n, the total number of triplet and k the number of landmarks. Creates all the triplet
        indices of the form (i, j, k), such that i is a landmark. :param n:
        #TODO :param num_trips: :return:
        """
        pairs_num = int(c * n * np.log2(n))
        # print(pairs_num)
        triplet_indices = torch.zeros(size=(k * pairs_num, 3), dtype=torch.long)
        for pivot in range(k):
            smaller_set = set(range(n)).difference([pivot])
            combs = np.array(list(combinations(smaller_set, 2)))
            combs_rand_ind = np.random.permutation(combs.shape[0])[:pairs_num]
            combs_sel = combs[combs_rand_ind,]
            triplet_indices[pivot * pairs_num:(pivot + 1) * pairs_num, 0] = pivot
            triplet_indices[pivot * pairs_num:(pivot + 1) * pairs_num, 1:3] = torch.Tensor(combs_sel)

        return triplet_indices, pairs_num

    @staticmethod
    def gen_landmark_indices_fast(n, k, c):
        """
                Description: Takes n, the total number of triplet and k the number of landmarks. Creates all the triplet
                indices of the form (i, j, k), such that i is a landmark. :param n:
                #TODO :param num_trips: :return:
                """
        pairs_num = int(c * n * np.log2(n))
        triplet_indices = torch.zeros(size=(k * pairs_num, 3), dtype=torch.long)
        first = torch.arange(0, k).reshape(k, 1).repeat(1, pairs_num).flatten()
        others = torch.randint(0, n, (pairs_num * k, 2))

        triplet_indices[:, 0] = first.squeeze()
        triplet_indices[:, 1:3] = others

        return triplet_indices, pairs_num

    def gen_landmark_triplets(self, n, k, c, device, bs=1000000):
        """
        Description: Takes n, the total number of triplet and k the number of landmarks. Creates all the triplets of
        the form (i, j, k), such that i is a landmark. bs is the batch size. we need it because otherwise the
        distance matrix goes into memory issues :param n: #TODO :param num_trips: :return:
        """
        # landmark_indices, pairs_num = self.gen_landmark_indices(n, k, c)
        landmark_indices, pairs_num = self.gen_landmark_indices_fast(n, k, c)
        triplets_answers = self.bulk_oracle(random_triplet_indices=landmark_indices, batch_size=bs)
        triplets_answers = torch.Tensor(triplets_answers).to(device)
        return landmark_indices.to(device), triplets_answers, pairs_num
