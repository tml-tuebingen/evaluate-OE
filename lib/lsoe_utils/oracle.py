import torch
import torch.nn.functional as F
import numpy as np
import sys

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
        query_result = torch.zeros(size=(num_triplets, ))  # Initializing the triplet set
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

    def bulk_query_pq(self, un_embedded_point_index, anchor_index, already_embedded_index, already_embeddings):
        """
        Description
        :param un_embedded_point_index:
        :param anchor_index:
        :param already_embedded_index:
        :param already_embeddings:
        :param p_or_q: Do we want p or q. Check the Pseudocode in Scaling ordinal embedding for details.
        :return:
        TODO: Make the embeddings a tensor. It is currently just in vector form.
        """
        # print(un_embedded_point_index.shape)
        # print(anchor_index.shape)
        # print(already_embedded_index.shape)
        # print(already_embeddings.shape)
        # sys.stdout.flush()
        # compute the distance between anchor and the un embedded point

        already_embeddings = torch.tensor(already_embeddings)
        compare_dist = torch.norm(self.data_store[un_embedded_point_index] -
                                                   self.data_store[anchor_index])
        m = len(already_embedded_index)

        distance_indices_anchor = torch.norm(self.data_store[anchor_index].repeat(m, 1) -
                                             self.data_store[already_embedded_index], dim=1)
        search_p = []
        search_q = []
        temp_already_embedded_index = torch.tensor(already_embedded_index, dtype = torch.int64)
        embedding_dists = torch.norm(already_embeddings[temp_already_embedded_index == anchor_index].repeat(m, 1) -
                                     already_embeddings, dim=1)

        # print(embedding_dists.shape)
        inds_for_p = torch.where(distance_indices_anchor > compare_dist)[0]
        inds_for_q = torch.where(distance_indices_anchor <= compare_dist)[0]

        if len(inds_for_p) == 0 or len(inds_for_q) == 0:
            return None, None, None, None


        ind_ind_p = torch.argmin(embedding_dists[inds_for_p])
        ind_ind_q = torch.argmin(embedding_dists[inds_for_q])

        # print(inds_for_p)
        # print(ind_ind_p)

        p = already_embedded_index[inds_for_p[ind_ind_p]]
        q = already_embedded_index[inds_for_q[ind_ind_q]]
        dt_to_p = embedding_dists[inds_for_p[ind_ind_p]]
        dt_to_q = embedding_dists[inds_for_q[ind_ind_q]]

        # for each_index in already_embedded_index:
        #     distance_between_anchor_and_each_index = torch.norm(self.data_store[anchor_index] -
        #                                                         self.data_store[each_index])
        #
        #     if distance_between_anchor_and_each_index > compare_dist:
        #         # Distance between anchor and every point where distance is according to the embeddings.
        #         distance_between_anchor_and_each_index_new = torch.norm(already_embeddings[temp_already_embedded_index == anchor_index] -
        #                                                                 already_embeddings[temp_already_embedded_index == each_index])
        #         search_p.append([distance_between_anchor_and_each_index_new, each_index])
        #     if distance_between_anchor_and_each_index < compare_dist:
        #         # Distance between anchor and every point where distance is according to the embeddings.
        #         distance_between_anchor_and_each_index_new = torch.norm(already_embeddings[temp_already_embedded_index == anchor_index] -
        #                                                                 already_embeddings[temp_already_embedded_index == each_index])
        #         search_q.append([distance_between_anchor_and_each_index_new, each_index])
        #
        # if len(search_p) < 1 or len(search_q) < 1:
        #     return None, None, None, None
        # else:
        #     p = min(search_p, key=lambda i: i[0])[1]
        #     q = max(search_q, key=lambda i: i[0])[1]
        #     dt_to_p = min(search_p, key=lambda i: i[0])[0]
        #     dt_to_q = max(search_q, key=lambda i: i[0])[0]
        return p, q, dt_to_p, dt_to_q
