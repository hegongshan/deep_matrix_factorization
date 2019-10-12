import scipy.sparse as sp
import numpy as np


class DataSet(object):

    def __init__(self, path: str):
        np.random.seed(34)
        self.path = path
        self.data_list, self.data_shape = self.load_rating_file_as_list(path + 'ratings.dat')
        self.data_matrix = self.get_data_matrix()
        self.train, self.test = self.get_train_test()
        self.train_data = self.get_train_instances()
        self.test_data = self.get_test_instances()

    def load_rating_file_as_list(self, filename: str):
        data = []
        num_user, num_item = 0, 0
        with open(filename, 'r') as file:
            for line in file:
                if line is not None and line != '':
                    arr = line.strip().split('::')
                    u, i, rating, timestamp = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3])
                    data.append([u, i, rating, timestamp])
                    if u > num_user:
                        num_user = u
                    if i > num_item:
                        num_item = i
        return data, (num_user, num_item)

    def get_data_matrix(self):
        mat = np.zeros(self.data_shape)
        for i in range(len(self.data_list)):
            line = self.data_list[i]
            user, item, rating = line[0], line[1], line[2]
            mat[user - 1, item - 1] = rating
        return mat

    # def load_rating_file_as_matrix(self, filename: str) -> sp.dok_matrix:
    #
    #     num_users, num_items = 0, 0
    #     with open(filename, 'r') as file:
    #         for line in file.readlines():
    #             if line is not None and line != '':
    #                 arr = line.split('::')
    #                 u, i = int(arr[0]), int(arr[1])
    #                 num_users = max(u, num_users)
    #                 num_items = max(i, num_items)
    #     print('number of users: %d, number of items: %d' % (num_users, num_items))
    #     mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    #
    #     with open(filename, 'r') as file:
    #         for line in file.readlines():
    #             if line is not None and line != '':
    #                 arr = line.split('::')
    #                 u, i, rating = int(arr[0]), int(arr[1]), float(arr[2])
    #                 if rating > 0:
    #                     mat[u, i] = rating
    #     print('shape of matrix Y: (%d,%d)' % mat.shape)
    #     return mat

    def get_train_test(self):
        data = self.data_list
        data = sorted(data, key=lambda x: (x[0], x[3]))

        train = []
        test = []
        for i in range(len(data) - 1):
            user, item, rating = data[i][0], data[i][1], data[i][2]
            if data[i][0] != data[i + 1][0]:
                test.append([user, item, rating])
            else:
                train.append([user, item, rating])

        test.append([data[-1][0], data[-1][1], data[-1][2]])
        return train, test

    def get_train_instances(self, num_negatives=7):
        user_input = []
        item_input = []
        ratings = []
        for i in self.train:
            u = i[0]
            user_input.append(self.data_matrix[u - 1])
            item_input.append(self.data_matrix[:, i[1] - 1])
            ratings.append(i[2])

            # negative samples
            num_items = self.data_matrix.shape[1]
            for t in range(num_negatives):
                while True:
                    j = np.random.randint(num_items)
                    if self.data_matrix[u - 1, j] == 0:
                        user_input.append(self.data_matrix[u - 1])
                        item_input.append(self.data_matrix[:, j])
                        ratings.append(0)
                        break
        return user_input, item_input, ratings

    def get_test_instances(self, num_negatives=100):
        test_ratings = []
        test_negatives = []
        for i in self.test:
            u = i[0]
            test_ratings.append([u, i[1], i[2]])
            # negative samples
            num_items = self.data_matrix.shape[1]
            negative = []
            for t in range(num_negatives):
                while True:
                    j = np.random.randint(1, num_items + 1)
                    if self.data_matrix[u - 1, j - 1] == 0:
                        negative.append(j)
                        break
            test_negatives.append(negative)
        return test_ratings, test_negatives


if __name__ == '__main__':
    dataset = DataSet('data/ml-1m/')
    print(len(dataset.data_matrix[:, 0]))
