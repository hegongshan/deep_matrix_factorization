import logging
import math
import os
from argparse import ArgumentParser
from time import time

import numpy as np
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda
from keras.models import Model

from dataset import DataSet
from evaluate import evaluate_model


class DMF(object):

    def __init__(self,
                 num_users,
                 num_items,
                 user_layers,
                 item_layers,
                 lr):
        self.max_rate = dataset.max_rate

        self.num_users = num_users
        self.num_items = num_items
        self.user_layers = user_layers
        self.item_layers = item_layers

        self.lr = lr

    @staticmethod
    def init_normal(shape, dtype=None):
        return K.random_normal(shape=shape, stddev=0.01, dtype=dtype)

    def normalized_crossentropy(self, y_true, y_pred):
        reg_rate = y_true / self.max_rate
        loss = reg_rate * K.log(y_pred) + (1 - reg_rate) * K.log(1 - y_pred)
        return -K.sum(loss)

    def cosine_similarity_relu(self, inputs):
        x, y = inputs[0], inputs[1]
        vec = K.batch_dot(x, y) / (K.sqrt(K.batch_dot(x, x) * K.batch_dot(y, y)))
        return K.maximum(vec, 1.0e-6)

    def get_model(self):
        user_input = Input(shape=(self.num_items,), dtype='float32', name='user_input')
        item_input = Input(shape=(self.num_users,), dtype='float32', name='item_input')

        user_vector = None
        item_vector = None
        for i in range(len(self.user_layers)):
            layer = Dense(self.user_layers[i],
                          activation='relu',
                          kernel_initializer=self.init_normal,
                          bias_initializer=self.init_normal,
                          name='user_layer%d' % (i + 1))
            if i == 0:
                user_vector = layer(user_input)
            else:
                user_vector = layer(user_vector)

        for i in range(len(self.item_layers)):
            layer = Dense(self.item_layers[i],
                          activation='relu',
                          kernel_initializer=self.init_normal,
                          bias_initializer=self.init_normal,
                          name='item_layer%d' % (i + 1))
            if i == 0:
                item_vector = layer(item_input)
            else:
                item_vector = layer(item_vector)

        # cosine similarity
        # y = Dot(axes=1, normalize=True)([user_vector, item_vector])
        # y_predict = Activation(activation=lambda x: K.maximum(x, self.mu))(y)

        y_predict = Lambda(function=self.cosine_similarity_relu, name='predict')([user_vector, item_vector])
        model = Model(inputs=[user_input, item_input], outputs=y_predict)
        model.compile(optimizer=optimizers.Adam(lr=self.lr), loss=self.normalized_crossentropy)
        return model


def generate_user_item_input(users, items, ratings, data_matrix, batch_size):
    batch = math.ceil(len(items) / batch_size)
    for batch_id in range(batch):
        user_input, item_input = [], []
        max_idx = min(len(items), (batch_id + 1) * batch_size)
        for idx in range(batch_id * batch_size, max_idx):
            u = users[idx]
            i = items[idx]
            item_input.append(data_matrix[:, i])
            user_input.append(data_matrix[u])
        target_ratings = ratings[batch_id * batch_size:max_idx]
        yield [np.array(user_input), np.array(item_input)], target_ratings


def parse_args():
    parser = ArgumentParser(description='Run DMF.')
    parser.add_argument('--path', nargs='?', default='data',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
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

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO,
                        filename='dmf_%s.log' % data_set)

    if not os.path.exists('model'):
        os.mkdir('model')
    model_out_file = 'model/%s_u%s_i%s_%d_%d.h5' % (data_set, str(dmf_user_layers),
                                                    str(dmf_item_layers), batch_size, time())

    # load data set
    dataset = DataSet('%s/%s/' % (path, data_set))

    # initialize DMF
    dmf = DMF(num_users=dataset.num_users,
              num_items=dataset.num_items,
              user_layers=dmf_user_layers,
              item_layers=dmf_item_layers,
              lr=lr)
    model = dmf.get_model()
    model.summary()

    (hits, ndcgs) = evaluate_model(model, dataset.test_ratings, dataset.test_negatives, dataset.data_matrix, topN)
    hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), -1
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    logging.info('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg = hr, ndcg
    best_iter = 0

    for epoch in range(epochs):
        start = time()

        # Generate training instances
        users, items, ratings = dataset.get_train_instances(num_train_negatives)

        print('start training...')

        # ml-1m's train set contains 994169 records, 994169 * 8 = 7953352
        # It's too large and will cause the following problem:
        # Process finished with exit code 137 (interrupted by signal 9: SIGKILL)

        # user_input, item_input = [], []
        # for idx in range(len(items)):
        #     u = users[idx]
        #     i = items[idx]
        #     item_input.append(dataset.data_matrix[:, i])
        #     user_input.append(dataset.data_matrix[u])

        # history = model.fit(x=[np.array(user_input), np.array(item_input)],
        #                     y=np.array(ratings),
        #                     batch_size=batch_size,
        #                     epochs=1,
        #                     shuffle=True)

        history = model.fit_generator(generate_user_item_input(users, items, ratings, dataset.data_matrix, batch_size),
                                      steps_per_epoch=math.ceil(len(users) / batch_size),
                                      epochs=1)

        end = time()
        print('Epoch %d Finished. [%.1f s]' % (epoch + 1, end - start))
        (hits, ndcgs) = evaluate_model(model, dataset.test_ratings, dataset.test_negatives, dataset.data_matrix, topN)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), history.history['loss'][0]

        print('HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
              % (hr, ndcg, loss, time() - end))
        logging.info('Epoch %d: HR = %.4f, NDCG = %.4f, loss = %.4f'
                     % (epoch + 1, hr, ndcg, loss))

        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch + 1
            model.save_weights(model_out_file, overwrite=True)

    print('Finished.\n Best epoch %d: HR = %.4f, NDCG = %.4f' % (best_iter, best_hr, best_ndcg))
    logging.info('Best epoch %d: HR = %.4f, NDCG = %.4f' % (best_iter, best_hr, best_ndcg))
