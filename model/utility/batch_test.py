from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout
import multiprocessing
import heapq
import numpy as np

cores = multiprocessing.cpu_count() // 2

args = parse_args()

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size


def test(sess, model, users_to_test, train_set_flag=0, stage=1):
    if stage == 1:
        rate = model.batch_ratings_1
    elif stage == 2:
        rate = model.batch_ratings_2
    else:
        rate = model.batch_ratings_3

    top_show = np.sort([20])
    max_top = max(top_show)
    result = {'precision': np.zeros(len([20])), 'recall': np.zeros(len([20])), 'ndcg': np.zeros(len([20]))}

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    all_result = []
    item_batch = range(ITEM_NUM)
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        rate_batch = sess.run(rate, {model.users: user_batch,
                                         model.pos_items: item_batch})

        rate_batch = np.array(rate_batch)  # (B, N)
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                test_items.append(data_generator.test_set[user])  # (B, #test_items)

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for idx, user in enumerate(user_batch):
                train_items_off = data_generator.train_items[user]
                rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                test_items.append(data_generator.train_items[user])

        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)  # (B,k*metric_num), max_top= 20
        count += len(batch_result)
        all_result.append(batch_result)

    assert count == n_test_users
    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)  # mean
    final_result = np.reshape(final_result, newshape=[5, max_top])
    final_result = final_result[:, top_show - 1]
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    return result










