import numpy as np
import random as rd
import scipy.sparse as sp
import time
import collections
import pickle
from utility.helper import *


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        
        self.path_cold = path + '/827/'
        ensureDir(self.path_cold)

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        self.train_users = {}

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.

                        if i not in self.train_users:
                            self.train_users[i] = []
                        self.train_users[i].append(uid)

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items


        self.R = self.R.tocsr()

        self.coo_R = self.R.tocoo()

    def get_adj_mat(self):
        try:
            t1 = time.time()

            norm_adj_mat_54 = sp.load_npz(self.path_cold + '/s_norm_adj_mat_54.npz')
            norm_adj_mat_45 = sp.load_npz(self.path_cold + '/s_norm_adj_mat_45.npz')
            norm_adj_mat_55 = sp.load_npz(self.path_cold + '/s_norm_adj_mat_55.npz')
            # print(norm_adj_mat_54)

            print('already load adj matrix', norm_adj_mat_45.shape, time.time() - t1)

        except Exception:
            norm_adj_mat_54, norm_adj_mat_45, norm_adj_mat_55 = self.create_adj_mat()

            sp.save_npz(self.path_cold + '/s_norm_adj_mat_54.npz', norm_adj_mat_54)
            sp.save_npz(self.path_cold + '/s_norm_adj_mat_45.npz', norm_adj_mat_45)
            sp.save_npz(self.path_cold + '/s_norm_adj_mat_55.npz', norm_adj_mat_55)

        return norm_adj_mat_54, norm_adj_mat_45, norm_adj_mat_55

    def create_adj_mat(self):
        t1 = time.time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time.time() - t1)

        t2 = time.time()

        def normalized_adj_symetric(adj, d1, d2):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, d1).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            d_inv_sqrt_last = np.power(rowsum, d2).flatten()
            d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
            d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

            return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()

        norm_adj_mat_45 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.5, -0.4)
        norm_adj_mat_54 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.5, -0.3)
        norm_adj_mat_55 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.5, -0.5)

        print('already normalize adjacency matrix', time.time() - t2)
        return norm_adj_mat_45.tocsr(), norm_adj_mat_54.tocsr(), norm_adj_mat_55.tocsr()


    def sample(self):
        total_users = self.exist_users
        users = rd.sample(total_users, self.batch_size)

        def sample_pos_items_for_u(u):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]
            return pos_i_id

        def sample_neg_items_for_u(u):
            pos_items = self.train_items[u]
            while True:
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in pos_items:
                    return neg_id

        pos_items, neg_items, pos_users_for_pi, neg_users_for_pi = [], [], [], []
        for u in users:
            pos_i = sample_pos_items_for_u(u)
            neg_i = sample_neg_items_for_u(u)

            pos_items.append(pos_i)
            neg_items.append(neg_i)

        return users, pos_items, neg_items


    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
