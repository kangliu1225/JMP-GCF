# encoding:utf-8
import logging
import os
import time
from utility.helper import *
import tensorflow as tf
import os
import sys
from utility.batch_test import *
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class JMPGCF(object):

    def __init__(self, data_config):

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj_53']
        self.norm_adj_54 = data_config['norm_adj_54']
        self.norm_adj_55 = data_config['norm_adj_55']

        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.n_layers = args.n_layers

        self.decay = args.decay

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # initialization of model parameters
        self.weights = self._init_weights()

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        A_fold_hat = self._split_A_hat(self.norm_adj)
        self.ua_embeddings_1, self.ia_embeddings_1, \
        self.ua_embeddings_2, self.ia_embeddings_2, \
        self.ua_embeddings_3, self.ia_embeddings_3, \
        self.ua_embeddings_4, self.ia_embeddings_4 = self._create_jmpgcf_norm_embed(ego_embeddings, A_fold_hat)

        self.u_g_embeddings_3 = tf.nn.embedding_lookup(self.ua_embeddings_3, self.users)
        self.pos_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.pos_items)
        self.neg_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.neg_items)

        self.u_g_embeddings_4 = tf.nn.embedding_lookup(self.ua_embeddings_4, self.users)
        self.pos_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.pos_items)
        self.neg_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.neg_items)

        self.batch_ratings_1 = tf.matmul(self.u_g_embeddings_3, self.pos_i_g_embeddings_3,
                                         transpose_a=False, transpose_b=True) + \
                               tf.matmul(self.u_g_embeddings_4, self.pos_i_g_embeddings_4,
                                         transpose_a=False, transpose_b=True)

        self.mf_loss_1, self.emb_loss_1, self.margin_loss_1 = self.create_bpr_loss_1()

        self.loss_1 = self.mf_loss_1 + self.emb_loss_1

        self.opt_1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_1)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        A_fold_hat = self._split_A_hat(self.norm_adj_54)
        self.ua_54_embeddings_1, self.ia_54_embeddings_1, \
        self.ua_54_embeddings_2, self.ia_54_embeddings_2, \
        self.ua_54_embeddings_3, self.ia_54_embeddings_3, \
        self.ua_54_embeddings_4, self.ia_54_embeddings_4 = self._create_jmpgcf_norm_embed(ego_embeddings, A_fold_hat)

        self.u_g_54_embeddings_3 = tf.nn.embedding_lookup(self.ua_54_embeddings_3, self.users)
        self.pos_i_g_54_embeddings_3 = tf.nn.embedding_lookup(self.ia_54_embeddings_3, self.pos_items)
        self.neg_i_g_54_embeddings_3 = tf.nn.embedding_lookup(self.ia_54_embeddings_3, self.neg_items)

        self.u_g_54_embeddings_4 = tf.nn.embedding_lookup(self.ua_54_embeddings_4, self.users)
        self.pos_i_g_54_embeddings_4 = tf.nn.embedding_lookup(self.ia_54_embeddings_4, self.pos_items)
        self.neg_i_g_54_embeddings_4 = tf.nn.embedding_lookup(self.ia_54_embeddings_4, self.neg_items)

        self.batch_ratings_2 = tf.matmul(self.u_g_54_embeddings_3, self.pos_i_g_54_embeddings_3,
                                         transpose_a=False, transpose_b=True) + \
                               tf.matmul(self.u_g_54_embeddings_4, self.pos_i_g_54_embeddings_4,
                                         transpose_a=False, transpose_b=True)
        self.batch_ratings_2 = self.batch_ratings_1 + self.batch_ratings_2

        self.mf_loss_2, self.emb_loss_2, self.margin_loss_2 = self.create_bpr_loss_2()

        self.loss_2 = self.mf_loss_2 + self.emb_loss_2 + self.loss_1

        self.opt_2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_2)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        A_fold_hat = self._split_A_hat(self.norm_adj_55)
        self.ua_55_embeddings_1, self.ia_55_embeddings_1, \
        self.ua_55_embeddings_2, self.ia_55_embeddings_2, \
        self.ua_55_embeddings_3, self.ia_55_embeddings_3, \
        self.ua_55_embeddings_4, self.ia_55_embeddings_4 = self._create_jmpgcf_norm_embed(ego_embeddings, A_fold_hat)

        self.u_g_55_embeddings_3 = tf.nn.embedding_lookup(self.ua_55_embeddings_3, self.users)
        self.pos_i_g_55_embeddings_3 = tf.nn.embedding_lookup(self.ia_55_embeddings_3, self.pos_items)
        self.neg_i_g_55_embeddings_3 = tf.nn.embedding_lookup(self.ia_55_embeddings_3, self.neg_items)

        self.u_g_55_embeddings_4 = tf.nn.embedding_lookup(self.ua_55_embeddings_4, self.users)
        self.pos_i_g_55_embeddings_4 = tf.nn.embedding_lookup(self.ia_55_embeddings_4, self.pos_items)
        self.neg_i_g_55_embeddings_4 = tf.nn.embedding_lookup(self.ia_55_embeddings_4, self.neg_items)

        self.batch_ratings_3 = tf.matmul(self.u_g_55_embeddings_4, self.pos_i_g_55_embeddings_4,
                                         transpose_a=False, transpose_b=True) + \
                               tf.matmul(self.u_g_55_embeddings_3, self.pos_i_g_55_embeddings_3,
                                         transpose_a=False, transpose_b=True)
        self.batch_ratings_3 = self.batch_ratings_3 + self.batch_ratings_2

        self.mf_loss_3, self.emb_loss_3, self.margin_loss_3 = self.create_bpr_loss_3()

        self.loss_3 = self.mf_loss_3 + self.emb_loss_3 + self.loss_2

        self.opt_3 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_3)


    def _init_weights(self):

        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_jmpgcf_norm_embed(self, ego_embeddings, A_fold_hat):

        all_embeddings = [ego_embeddings]

        for k in range(0, 5):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        u_g_embeddings_1, i_g_embeddings_1 = tf.split(all_embeddings[1], [self.n_users, self.n_items], 0)
        u_g_embeddings_2, i_g_embeddings_2 = tf.split(all_embeddings[2], [self.n_users, self.n_items], 0)
        u_g_embeddings_3, i_g_embeddings_3 = tf.split(all_embeddings[3], [self.n_users, self.n_items], 0)
        u_g_embeddings_4, i_g_embeddings_4 = tf.split(all_embeddings[4], [self.n_users, self.n_items], 0)
        
        return u_g_embeddings_1, i_g_embeddings_1, u_g_embeddings_2, i_g_embeddings_2, \
               u_g_embeddings_3, i_g_embeddings_3, u_g_embeddings_4, i_g_embeddings_4

    def create_bpr_loss_1(self):

        pos_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_3, self.pos_i_g_embeddings_3), axis=1)
        neg_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_3, self.neg_i_g_embeddings_3), axis=1)

        pos_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.pos_i_g_embeddings_4), axis=1)
        neg_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.neg_i_g_embeddings_4), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_4) + tf.nn.l2_loss(self.pos_i_g_embeddings_4) + \
                      tf.nn.l2_loss(self.neg_i_g_embeddings_4)

        regularizer = regularizer / args.batch_size

        margin_loss = tf.constant(0.0)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores_3 - neg_scores_3))) + \
                  tf.reduce_mean(tf.nn.softplus(-(pos_scores_4 - neg_scores_4)))

        emb_loss = self.decay * regularizer

        return mf_loss, emb_loss, margin_loss

    def create_bpr_loss_2(self):

        pos_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_54_embeddings_3, self.pos_i_g_54_embeddings_3), axis=1)
        neg_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_54_embeddings_3, self.neg_i_g_54_embeddings_3), axis=1)

        pos_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_54_embeddings_4, self.pos_i_g_54_embeddings_4), axis=1)
        neg_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_54_embeddings_4, self.neg_i_g_54_embeddings_4), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_54_embeddings_4) + tf.nn.l2_loss(self.pos_i_g_54_embeddings_4) + \
                      tf.nn.l2_loss(self.neg_i_g_54_embeddings_4)
        

        regularizer = regularizer / args.batch_size

        margin_loss = tf.constant(0.0)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores_3 - neg_scores_3))) + \
                  tf.reduce_mean(tf.nn.softplus(-(pos_scores_4 - neg_scores_4)))

        emb_loss = self.decay * regularizer

        return mf_loss, emb_loss, margin_loss

    def create_bpr_loss_3(self):

        pos_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_55_embeddings_3, self.pos_i_g_55_embeddings_3), axis=1)
        neg_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_55_embeddings_3, self.neg_i_g_55_embeddings_3), axis=1)

        pos_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_55_embeddings_4, self.pos_i_g_55_embeddings_4), axis=1)
        neg_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_55_embeddings_4, self.neg_i_g_55_embeddings_4), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_55_embeddings_4) + tf.nn.l2_loss(self.pos_i_g_55_embeddings_4) + \
                      tf.nn.l2_loss(self.neg_i_g_55_embeddings_4)

        regularizer = regularizer / args.batch_size

        margin_loss = tf.constant(0.0)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores_3 - neg_scores_3))) + \
                  tf.reduce_mean(tf.nn.softplus(-(pos_scores_4 - neg_scores_4)))

        emb_loss = self.decay * regularizer

        return mf_loss, emb_loss, margin_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    ensureDir('Logs/')

    logfile = 'Logs/' + rq + '.txt'

    fh = logging.FileHandler(logfile, mode='w')

    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    logger.addHandler(fh)

    logger.setLevel(logging.DEBUG)

    print('lr-->', args.lr)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    print('layer-->', args.n_layers)
    data_generator.print_statistics()
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['decay'] = args.decay

    logger.info('#' * 40 + ' dataset={} '.format(args.dataset) + '#' * 40)
    logger.info('#' * 40 + ' n_layers={} '.format(args.n_layers) + '#' * 40)
    logger.info('#' * 40 + ' decay={} '.format(args.decay) + '#' * 40)
    logger.info('#' * 40 + ' lr={} '.format(args.lr) + '#' * 40)
    logger.info('-' * 100)

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    norm_54, norm_53, norm_55 = data_generator.get_adj_mat()

    config['norm_adj_53'] = norm_53
    config['norm_adj_54'] = norm_54
    config['norm_adj_55'] = norm_55

    t0 = time.time()

    model = JMPGCF(data_config=config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    max_recall, max_precision, max_ndcg, max_hr = 0., 0., 0., 0.
    max_epoch = 0
    flag = 'normal'
    stage = 0

    metric_dict = {}
    metric_dict['ndcg'] = []
    metric_dict['recall'] = []

    min_loss = 100
    loss_step = 0

    if args.dataset == 'gowalla':
        phase_1, phase_2 = 300, 600
    else:
        phase_1, phase_2 = 300, 1000

    for epoch in range(1000):
        t1 = time.time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        loss_cold, margin_loss_cold, emb_loss_cold = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()

            if epoch <= phase_1:
                stage = 1
                flag = 'First training phase.'
                _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                    [model.opt_1, model.loss_1, model.mf_loss_1, model.emb_loss_1],
                    feed_dict={model.users: users,
                               model.pos_items: pos_items,
                               model.neg_items: neg_items})
                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss
            elif epoch <= phase_2:
                stage = 2
                flag = 'Second training phase.'
                _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                    [model.opt_2, model.loss_2, model.mf_loss_2, model.emb_loss_2],
                    feed_dict={model.users: users,
                               model.pos_items: pos_items,
                               model.neg_items: neg_items})
                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss
            else:
                stage = 3
                flag = 'Third training phase.'
                _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                    [model.opt_3, model.loss_3, model.mf_loss_3, model.emb_loss_3],
                    feed_dict={model.users: users,
                               model.pos_items: pos_items,
                               model.neg_items: neg_items})
                loss += batch_loss
                mf_loss += batch_mf_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        print('[%s] Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
            flag, epoch, time.time() - t1, loss, mf_loss, emb_loss))

        if (epoch + 1) % 20 != 0:
            continue

        t2 = time.time()
        users_to_test = list(data_generator.test_set.keys())

        ret = test(sess, model, users_to_test, stage=stage)

        metric_dict['ndcg'].append(ret['ndcg'][0])
        metric_dict['recall'].append(ret['recall'][0])

        t3 = time.time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        perf_str = '[%s] Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f], ' \
                   'precision=[%.5f], ndcg=[%.5f]' % \
                   (flag, epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                    ret['recall'][0],
                    ret['precision'][0],
                    ret['ndcg'][0])
        logger.info(perf_str)


