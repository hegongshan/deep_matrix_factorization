import scipy.sparse as sp
import numpy as np
import os


class DataSet(object):

    def __init__(self, path, data_set='ml-100k'):
        path_sep = os.path.sep
        filename = path + path_sep + data_set + path_sep
        if data_set == 'ml-100k':
            filename += 'u.data'
            data_separator = '\t'
        elif data_set == 'ml-1m':
            filename += 'ratings.dat'
            data_separator = '::'
        else:
            raise FileNotFoundError('Directory %s not found!' % path + path_sep + data_set)

        self.data_list, self.num_users, self.num_items, self.max_rate = \
            self.load_rating_file_as_list(filename, separator=data_separator)
        self.train, self.test = self.get_train_test()
        self.data_matrix = self.get_data_matrix()
        # self.train_data = self.get_train_instances()
        self.test_ratings, self.test_negatives = self.get_test_instances()

    @staticmethod
    def load_rating_file_as_list(filename, separator='\t'):
        print('loading rating file: %s...' % filename)
        data = []
        num_users, num_items, max_rate = 0, 0, 0
        with open(filename, 'r') as file:
            for line in file:
                if line is not None and line != '':
                    arr = line.strip().split(separator)
                    u, i, rating, timestamp = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3])
                    data.append([u, i, rating, timestamp])
                    if u > num_users:
                        num_users = u
                    if i > num_items:
                        num_items = i
                    if rating > max_rate:
                        max_rate = rating
        print('number of users: %d, number of items: %d' % (num_users, num_items))
        return data, num_users, num_items, max_rate

    def get_data_matrix(self):
        mat = np.zeros((self.num_users, self.num_items))
        for line in self.train:
            user, item, rating = line[0], line[1], line[2]
            mat[user, item] = rating
        return mat

    def get_train_test(self):
        print('splitting train and test data...')
        data = self.data_list
        data = sorted(data, key=lambda x: (x[0], x[3]))

        train = []
        test = []
        for i in range(len(data) - 1):
            user, item, rating = data[i][0], data[i][1], data[i][2]
            if data[i][0] != data[i + 1][0]:
                test.append((user - 1, item - 1, rating))
            else:
                train.append((user - 1, item - 1, rating))

        test.append((data[-1][0] - 1, data[-1][1] - 1, data[-1][2]))
        return train, test

    def get_train_instances(self, num_negatives):
        print('getting train instances...')
        user_input = []
        item_input = []
        ratings = []
        for i in self.train:
            u = i[0]
            user_input.append(u)
            item_input.append(i[1])
            ratings.append(i[2])

            # negative samples
            item_list = []
            for t in range(num_negatives):
                while True:
                    j = np.random.randint(self.num_items)
                    if self.data_matrix[u, j] == 0 and j not in item_list:
                        user_input.append(u)
                        item_input.append(j)
                        ratings.append(0)

                        item_list.append(j)
                        break
        return user_input, item_input, ratings

    def get_test_instances(self, num_negatives=100):
        print('getting test instances...')
        np.random.seed(34)
        test_ratings = []
        test_negatives = []
        for i in self.test:
            u = i[0]
            test_ratings.append([u, i[1], i[2]])
            # negative samples
            negative = []
            for t in range(num_negatives):
                while True:
                    j = np.random.randint(self.num_items)
                    if self.data_matrix[u, j] == 0 and j not in negative:
                        negative.append(j)
                        break
            test_negatives.append(negative)
        return test_ratings, test_negatives


if __name__ == '__main__':
    dataset = DataSet('data/ml-1m/')
    print(len(dataset.train))
    print(len(dataset.test))
    print(len(dataset.get_train_instances(7)[2]))
    print(len(dataset.test_ratings))
