import pickle
from sklearn.utils import shuffle
from sklearn import model_selection
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd


class DataLoader:
    def __init__(self,
                 file_path,
                 binarize,
                 zca_whiten=False,
                 contrast_normalize=False):
        self.binarize = binarize
        self.zca_whiten = zca_whiten
        self.contrast_normalize = contrast_normalize
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def get_data_and_labels(self):
        X_train = self.data['train_images']
        X_test = self.data['test_images']
        y_train = self.data['train_labels']
        y_test = self.data['test_labels']

        X_train = np.clip(X_train, 0, 1.0)
        X_test = np.clip(X_test, 0, 1.0)

        print('Training size : %d \t Test size : %d' %
              (X_train.shape[0], X_test.shape[0]))

        channel = int(np.prod(X_train.shape) / (X_train.shape[0] * 28 * 28))

        # not possible to not split them
        X_train = shuffle(X_train.reshape(X_train.shape[0], 28, 28, channel),
                          random_state=42)
        y_train = shuffle(y_train, random_state=42)
        X_test = shuffle(X_test.reshape(X_test.shape[0], 28, 28, 1),
                         random_state=137)
        y_test = shuffle(y_test, random_state=42)

        if self.binarize:
            X_train[X_train >= .5] = 1.
            X_train[X_train < .5] = 0.
            X_test[X_test >= .5] = 1.
            X_test[X_test < .5] = 0.
        if self.zca_whiten:
            # add +1 and divide by 2 to shift
            # them back to the interval [-1, 1]
            X_train = self._whiten(X_train)
            X_test = self._whiten(X_test)
        if self.contrast_normalize:
            X_train = np.array([
                self._contrast_normalization_train(x, y_train[ind])
                for ind, x in enumerate(X_train)
            ])
            X_test = np.array([
                self._contrast_normalization_test(x, y_test[ind])
                for ind, x in enumerate(X_test)
            ])

        print("Shapes : ", X_train.shape, "\t", X_test.shape)
        print("Label shaped : ", y_train.shape, "\t", y_test.shape)

        mean, std = np.mean(X_train), np.std(X_train)
        print("Train set : ")
        print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
        print('Min: %.3f, Max: %.3f' % (np.min(X_train), np.max(X_train)))

        mean, std = np.mean(X_test), np.std(X_test)
        print("Test set : ")
        print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
        print('Min: %.3f, Max: %.3f' % (np.min(X_test), np.max(X_test)))

        return (X_train, y_train, X_test, y_test)

    def train_test_split(self):
        try:
            X_train, y_train, X_test, y_test = self.get_data_and_labels()
            return (X_train, X_test)
        except KeyError:
            print('not getting here')
            X_train = self.data['train_images']
            X_test = self.data['test_images']

            X_train = np.clip(X_train, 0, 1.0)
            X_test = np.clip(X_test, 0, 1.0)

            print('Training size : %d \t Test size : %d' %
                  (X_train.shape[0], X_test.shape[0]))

            channel = int(
                np.prod(X_train.shape) / (X_train.shape[0] * 28 * 28))

            # not possible to not split them
            X_train = shuffle(X_train.reshape(X_train.shape[0], 28, 28,
                                              channel),
                              random_state=42)
            X_test = shuffle(X_test.reshape(X_test.shape[0], 28, 28, 1),
                             random_state=137)

            print("Shapes : ", X_train.shape, "\t", X_test.shape)
        except IndexError:
            X = self.data
            X = np.clip(X, 0, 1.0)

            channel = int(np.prod(X.shape) / (X.shape[0] * 28 * 28))

            X_train, X_test = model_selection.train_test_split(
                X.reshape(X.shape[0], 28, 28, channel),
                test_size=0.05,
                random_state=137)
        finally:
            if self.binarize:
                X_train[X_train >= .5] = 1.
                X_train[X_train < .5] = 0.
                X_test[X_test >= .5] = 1.
                X_test[X_test < .5] = 0.
            if self.zca_whiten:
                # add +1 and divide by 2 to shift
                # them back to the interval [-1, 1]
                X_train = self._whiten(X_train)
                X_test = self._whiten(X_test)

            mean, std = np.mean(X_train), np.std(X_train)
            print("Train set : ")
            print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
            print('Min: %.3f, Max: %.3f' % (np.min(X_train), np.max(X_train)))

            return (X_train, X_test)

    def _whiten(self, x):
        flat_x = np.reshape(x,
                            (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = svd(sigma)
        s_inv = 1. / np.sqrt(s[np.newaxis] + 1e-12)
        principal_components = (u * s_inv).dot(u.T)
        flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
        whitex = np.dot(flatx, principal_components)
        x = np.reshape(whitex, x.shape)
        return x

    def _contrast_normalization_train(self,
                                      x,
                                      label,
                                      s=3.33,
                                      lmbd=10,
                                      epsilon=1e-2):
        means = {
            0: 0.49618801083712977,
            1: 0.5000000000000002,
            2: 0.5000043854028291,
            3: 0.49999037217872805,
            4: 0.4999999999999999,
            5: 0.49999999999999944,
            6: 0.5000000000000006
        }
        stds = {
            0: 0.2795232304659204,
            1: 0.12972179491334127,
            2: 0.1878776468561605,
            3: 0.1455744995271547,
            4: 0.042093943681260626,
            5: 0.08038296207324777,
            6: 0.11335077635372165
        }
        x_label = means[label]
        x_updated = x - x_label
        x_updated /= stds[label]
        x_updated += 0.5  # set new mean
        mini, maxi = -4.753, 7.455  # from train min and max values
        diff = maxi - mini
        x_updated -= mini
        x_updated /= diff
        x_updated = np.clip(x_updated, 0, 1)
        return x_updated

    def _contrast_normalization_test(self,
                                     x,
                                     label,
                                     s=3.33,
                                     lmbd=10,
                                     epsilon=1e-2):
        means = {
            0: 0.49938174332232904,
            1: 0.4994474854248672,
            2: 0.49953722540249085,
            3: 0.4993921865195907,
            4: 0.4994409264556163,
            5: 0.49944513245349165,
            6: 0.4995206918013805
        }
        stds = {
            0: 0.16108819138477978,
            1: 0.15795423837534453,
            2: 0.1530487919614309,
            3: 0.1603416970580439,
            4: 0.15674000765449378,
            5: 0.15705830699788273,
            6: 0.15216802897686568
        }
        x_label = means[label]
        x_updated = x - x_label
        x_updated /= stds[label]
        x_updated += 0.5  # set new mean
        mini, maxi = -2.783, 3.789  # from train mean and max values
        diff = maxi - mini
        x_updated -= mini
        x_updated /= diff
        x_updated *= (0.081913499344692 / 0.15216068167985394)
        x_updated += (0.4302916120576671 - 0.26892201834862384)
        x_updated = np.clip(x_updated, 0, 1)
        return x_updated
