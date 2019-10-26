import numpy as np
import math
import heapq

_model = None
_test_ratings = None
_test_negatives = None
_data_matrix = None


def evaluate_model(model, test_ratings, test_negatives, data_matrix, k):
    global _model
    global _test_ratings
    global _test_negatives
    global _data_matrix

    _model = model
    _test_ratings = test_ratings
    _test_negatives = test_negatives
    _data_matrix = data_matrix

    hits, ndcgs = [], []
    for i in range(len(_test_ratings)):
        (hr, ndcg) = _evaluate_one_rating(i, k=k)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs


def _evaluate_one_rating(idx, k):
    rating = _test_ratings[idx]
    items = _test_negatives[idx]
    user = rating[0]
    gt_item = rating[1]
    items.append(gt_item)

    items_input = []
    users_input = []
    for item in items:
        items_input.append(_data_matrix[:, item])
        users_input.append(_data_matrix[user])
    predictions = _model.predict([np.array(users_input), np.array(items_input)],
                                 batch_size=100 + 1,
                                 verbose=0)

    map_item_score = {}
    for idx, item in enumerate(items):
        map_item_score[item] = predictions[idx]

    items.pop()
    rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
    hr = get_hit_ratio(rank_list, gt_item)
    ndcg = get_ndcg(rank_list, gt_item)
    return hr, ndcg


def get_hit_ratio(rank_list, gt_item):
    if gt_item in rank_list:
        return 1
    return 0


def get_ndcg(rank_list, gt_item):
    for idx, item in enumerate(rank_list):
        if item == gt_item:
            return math.log(2) / math.log(idx + 2)
    return 0
