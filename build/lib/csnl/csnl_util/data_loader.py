import pickle
from sklearn import model_selection
import numpy as np
import os


class DataLoader:
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def train_test_split(self):
        try:
            X_train = self.data['train_images']
            X_test = self.data['test_images']
            X_train, _ = model_selection.train_test_split(
                X_train.reshape(X_train.shape[0], 28, 28, 1), test_size=0, random_state=45)
            X_test, _ = model_selection.train_test_split(
                X_test.reshape(X_test.shape[0], 28, 28, 1), test_size=0, random_state=42)
        except IndexError:
            X = self.data
            X_train, X_test = model_selection.train_test_split(
                X.reshape(X.shape[0], 28, 28, 1), test_size=0.33, random_state=137)
        finally:
            return (X_train, X_test)
