from keras.layers import Input, Dense, Dot, Activation
from keras.models import Model
from keras import initializers, optimizers, activations
import keras.backend as K
import numpy as np
from time import time
from dataset import DataSet
from evaluate import evaluate_model


class DMF(object):

    def __init__(self, lr=0.0001, maxRate=5):
        self.lr = lr
        self.maxRate = maxRate

    def normalized_crossentropy(self, y_true, y_pred):
        regRate = y_true / self.maxRate
        loss = regRate * K.log(y_pred) + (1 - regRate) * K.log(1 - y_pred)
        return -loss

    def cosine_similarity(self, x, y):
        return K.sum(x * y) / (K.sqrt(K.sum(x ** 2)) * K.sqrt(K.sum(y ** 2)))

    def relu(self, x):
        return K.relu(x, threshold=1e-6, max_value=x)

    def get_model(self, num_users, num_items):
        user_input = Input(shape=(num_items,), dtype='float32', name='user_input')
        item_input = Input(shape=(num_users,), dtype='float32', name='item_input')

        p_layer1 = Dense(128, activation='relu',
                         kernel_initializer=initializers.random_normal(stddev=0.01),
                         bias_initializer=initializers.random_normal(stddev=0.01))(user_input)
        q_layer1 = Dense(128, activation='relu',
                         kernel_initializer=initializers.random_normal(stddev=0.01),
                         bias_initializer=initializers.random_normal(stddev=0.01))(item_input)

        p_layer2 = Dense(64, activation='relu',
                         kernel_initializer=initializers.random_normal(stddev=0.01),
                         bias_initializer=initializers.random_normal(stddev=0.01))(p_layer1)
        q_layer2 = Dense(64, activation='relu',
                         kernel_initializer=initializers.random_normal(stddev=0.01),
                         bias_initializer=initializers.random_normal(stddev=0.01))(q_layer1)

        # cosine similarity
        y = Dot(axes=1, normalize=True)([p_layer2, q_layer2])
        y_predict = Activation(activation=self.relu)(y)

        model = Model(inputs=[user_input, item_input], outputs=y_predict)
        model.compile(optimizer=optimizers.Adam(lr=self.lr), loss=self.normalized_crossentropy)
        return model


if __name__ == '__main__':
    dataset = DataSet('data/ml-1m/')
    user_input, item_input, ratings = dataset.train_data
    test_ratings, test_negatives = dataset.test_data

    dmf = DMF()
    model = dmf.get_model(num_users=dataset.data_shape[0], num_items=dataset.data_shape[1])
    model.summary()

    epochs = 10
    top_k = 10
    model_out_file = 'model/ml-1m-[128,64]-256.h5'

    (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, dataset.data_matrix, top_k)
    hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), -1
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg = hr, ndcg

    history = model.fit(x=[np.array(user_input), np.array(item_input)],
                        y=np.array(ratings),
                        batch_size=8,
                        epochs=1)
    print(history.history)
    exit(-1)

    for epoch in range(epochs):
        start = time()
        history = model.fit(x=[np.array(user_input), np.array(item_input)],
                            y=np.array(ratings),
                            batch_size=1,
                            epochs=1,
                            shuffle=True)
        end = time()
        print('Epoch %d [%.1f s]' % (epoch + 1, end - start))
        (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, top_k)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), history['loss'][0]

        print('HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
              % (hr, ndcg, loss, time() - end))

        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            model.save_weights(model_out_file, overwrite=True)

    print('Finished.\n Best epoch %d: HR = %.4f, NDCG = %.4f' % (best_iter, best_hr, best_ndcg))
