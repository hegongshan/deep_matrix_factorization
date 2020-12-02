import logging
import os
from argparse import ArgumentParser
from time import time

import numpy as np
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda, Flatten
from keras.models import Model

from dataset import DataSet
from evaluate import evaluate_model


def parse_args():
    parser = ArgumentParser(description='Run DMF.')
    parser.add_argument('--path', nargs='?', default='data',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset, either ml-1m or ml-100k.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--user_layers', nargs='?', default='[512,64]',
                        help="Size of each layer for user.")
    parser.add_argument('--item_layers', nargs='?', default='[1024,64]',
                        help="Size of each layer for item.")
    parser.add_argument('--num_neg', type=int, default=7,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--topN', type=int, default=10,
                        help='Size of recommendation list.')
    return parser.parse_args()


class DMF(object):

    def __init__(self,
                 num_users,
                 num_items,
                 user_layers,
                 item_layers,
                 lr,
                 train_matrix):
        self.num_users = num_users
        self.num_items = num_items
        self.user_layers = user_layers
        self.item_layers = item_layers

        self.lr = lr

        self.user_rating = K.constant(train_matrix)
        self.item_rating = K.constant(train_matrix.T)

    @staticmethod
    def init_normal(shape, dtype=None):
        return K.random_normal(shape=shape, stddev=0.01, dtype=dtype)

    @staticmethod
    def cosine_similarity(inputs, epsilon=1.0e-6, delta=1e-12):
        x, y = inputs[0], inputs[1]
        numerator = K.sum(x * y, axis=1, keepdims=True)
        denominator = K.sqrt(K.sum(x * x, axis=1, keepdims=True) * K.sum(y * y, axis=1, keepdims=True))
        cosine_similarity = numerator / K.maximum(denominator, delta)
        return K.maximum(cosine_similarity, epsilon)

    def get_model(self):
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        user_rating_input = Lambda(lambda x: K.gather(self.user_rating, x))(user_input)
        user_rating_vector = Flatten()(user_rating_input)

        item_rating_input = Lambda(lambda x: K.gather(self.item_rating, x))(item_input)
        item_rating_vector = Flatten()(item_rating_input)

        user_vector = None
        item_vector = None
        for i in range(len(self.user_layers)):
            layer = Dense(self.user_layers[i],
                          activation='relu',
                          kernel_initializer=self.init_normal,
                          bias_initializer=self.init_normal,
                          name='user_layer%d' % (i + 1))
            if i == 0:
                user_vector = layer(user_rating_vector)
            else:
                user_vector = layer(user_vector)

        for i in range(len(self.item_layers)):
            layer = Dense(self.item_layers[i],
                          activation='relu',
                          kernel_initializer=self.init_normal,
                          bias_initializer=self.init_normal,
                          name='item_layer%d' % (i + 1))
            if i == 0:
                item_vector = layer(item_rating_vector)
            else:
                item_vector = layer(item_vector)

        y_predict = Lambda(function=self.cosine_similarity, name='predict')([user_vector, item_vector])
        model = Model(inputs=[user_input, item_input],
                      outputs=y_predict)
        model.compile(optimizer=optimizers.Adam(lr=self.lr),
                      loss='binary_crossentropy')
        return model


# def generate_user_item_input(users, items, ratings, data_matrix, batch_size):
#     batch = math.ceil(len(items) / batch_size)
#     for batch_id in range(batch):
#         user_input, item_input = [], []
#         max_idx = min(len(items), (batch_id + 1) * batch_size)
#         for idx in range(batch_id * batch_size, max_idx):
#             u = users[idx]
#             i = items[idx]
#             item_input.append(data_matrix[:, i])
#             user_input.append(data_matrix[u])
#         target_ratings = ratings[batch_id * batch_size:max_idx]
#         yield [np.array(user_input), np.array(item_input)], target_ratings

def log_config(log_name: str) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.handlers = []
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        filename=log_name,
                        level=logging.INFO)


def output_result(content: str) -> None:
    print(content)
    logging.info(content)


if __name__ == '__main__':

    args = parse_args()
    path = args.path
    epochs = args.epochs
    dmf_user_layers = eval(args.user_layers)
    dmf_item_layers = eval(args.item_layers)
    batch_size = args.batch_size
    data_set = args.dataset
    lr = args.lr
    topN = args.topN
    num_train_negatives = args.num_neg

    print(args)
    log_config('dmf_%s.log' % data_set)

    if not os.path.exists('model'):
        os.mkdir('model')
    model_out_file = 'model/%s_u%s_i%s_%d_%d.h5' % (data_set, str(dmf_user_layers),
                                                    str(dmf_item_layers), batch_size, time())

    # load data set
    dataset = DataSet(path, data_set)

    # initialize DMF
    dmf = DMF(num_users=dataset.num_users,
              num_items=dataset.num_items,
              user_layers=dmf_user_layers,
              item_layers=dmf_item_layers,
              lr=lr,
              train_matrix=dataset.data_matrix)
    model = dmf.get_model()
    model.summary()

    (hits, ndcgs) = evaluate_model(model, dataset.test_ratings, dataset.test_negatives, dataset.data_matrix, topN)
    hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), -1
    output_result('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg = hr, ndcg
    best_iter = 0

    for epoch in range(epochs):
        start = time()

        # Generate training instances
        user_input, item_input, ratings = dataset.get_train_instances(num_train_negatives)

        # history = model.fit_generator(generate_user_item_input(users, items, ratings, dataset.data_matrix, batch_size),
        #                               steps_per_epoch=math.ceil(len(users) / batch_size),
        #                               epochs=1)

        history = model.fit(x=[np.array(user_input), np.array(item_input)],
                            y=np.array(ratings),
                            batch_size=batch_size,
                            epochs=1,
                            shuffle=True)

        end = time()
        (hits, ndcgs) = evaluate_model(model, dataset.test_ratings, dataset.test_negatives, dataset.data_matrix, topN)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), history.history['loss'][0]

        output_result('Epoch %d: HR = %.4f, NDCG = %.4f, loss = %.4f'
                      % (epoch + 1, hr, ndcg, loss))

        if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch + 1
            model.save_weights(model_out_file, overwrite=True)

    output_result('Best epoch %d: HR = %.4f, NDCG = %.4f' % (best_iter, best_hr, best_ndcg))
