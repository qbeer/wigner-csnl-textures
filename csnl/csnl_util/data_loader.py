import pickle
from sklearn.utils import shuffle
from sklearn import model_selection
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd


class DataLoader:
    def __init__(self, file_path, binarize, zca_whiten=False):
        self.binarize = binarize
        self.zca_whiten = zca_whiten
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def train_test_split(self):
        try:
            X_train = self.data['train_images']
            X_test = self.data['test_images']

            X_train = np.clip(X_train, 0, 1.0)
            X_test = np.clip(X_test, 0, 1.0)

            mean, std = np.mean(X_train), np.std(X_train)
            print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
            print('Min: %.3f, Max: %.3f' % (np.min(X_train), np.max(X_train)))
            print('Training size : %d \t Test size : %d' %
                  (X_train.shape[0], X_test.shape[0]))

            channel = int(np.prod(X_train.shape) /
                          (X_train.shape[0] * 28 * 28))

            # not possible to not split them
            X_train = shuffle(X_train.reshape(
                X_train.shape[0], 28, 28, channel), random_state=42)
            X_test = shuffle(X_test.reshape(
                X_test.shape[0], 28, 28, 1), random_state=137)

            print("Shapes : ", X_train.shape, "\t", X_test.shape)
        except IndexError:
            X = self.data
            X = np.clip(X, 0, 1.0)

            mean, std = np.mean(X), np.std(X)
            print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
            print('Min: %.3f, Max: %.3f' % (np.min(X), np.max(X)))

            channel = int(np.prod(X.shape) / (X.shape[0] * 28 * 28))

            X_train, X_test = model_selection.train_test_split(
                X.reshape(X.shape[0], 28, 28, channel), test_size=0.05, random_state=137)
        finally:
            if self.binarize:
                X_train[X_train >= .5] = 1.
                X_train[X_train < .5] = 0.
                X_test[X_test >= .5] = 1.
                X_test[X_test < .5] = 0.
            if self.zca_whiten:
                X_train = self._whiten(X_train)
                X_test = self._whiten(X_test)
            return (X_train, X_test)

    def _whiten(self, x):
        flat_x = np.reshape(
            x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = svd(sigma)
        s_inv = 1. / np.sqrt(s[np.newaxis] + 1e-12)
        principal_components = (u * s_inv).dot(u.T)
        flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
        whitex = np.dot(flatx, principal_components)
        x = np.reshape(whitex, x.shape)
        return x
