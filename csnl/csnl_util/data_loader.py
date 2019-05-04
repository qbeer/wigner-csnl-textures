import pickle
from sklearn import model_selection
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, file_path, binarize):
        self.binarize = binarize
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

            channel = int(np.prod(X_train.shape) / (X_train.shape[0] * 28 * 28))

            X_train, _ = model_selection.train_test_split(
                X_train.reshape(X_train.shape[0], 28, 28, channel), test_size=0, random_state=45)
            X_test, _ = model_selection.train_test_split(
                X_test.reshape(X_test.shape[0], 28, 28, channel), test_size=0, random_state=42)
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
            return (X_train, X_test)
