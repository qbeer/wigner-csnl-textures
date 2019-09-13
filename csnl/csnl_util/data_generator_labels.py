from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .data_loader import DataLoader
import os


class DataGeneratorWithLabels:
    def __init__(self,
                 image_shape,
                 batch_size,
                 file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl',
                 binarize=False,
                 whiten=False,
                 contrast_normalize=False):
        self.DATA_LOADER = DataLoader(file_path, binarize, whiten,
                                      contrast_normalize)
        self.IMAGE_SHAPE = image_shape
        self.BATCH_SIZE = batch_size
        self.DATA_GENERATOR = ImageDataGenerator()
        self.TRAIN, self.TRAIN_LABELS, self.TEST, self.TEST_LABELS = self.DATA_LOADER.get_data_and_labels(
        )
        print("Train SHAPE : ", self.TRAIN.shape)
        means = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: []
        }
        stds = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: []
        }
        for ind, label in enumerate(self.TRAIN_LABELS):
            means[label].append(self.TRAIN[ind])
        for K in means.keys():
            stds[K] = np.std(means[K])
            means[K] = np.mean(means[K])
        print("MEAN : ", means)
        print("STD : ", stds)

        print("Test SHAPE : ", self.TEST.shape)
        means = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: []
        }
        stds = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: []
        }
        for ind, label in enumerate(self.TEST_LABELS):
            means[label].append(self.TEST[ind])
        for K in means.keys():
            stds[K] = np.std(means[K])
            means[K] = np.mean(means[K])
        print("MEAN : ", means)
        print("STD : ", stds)

    def flow(self):
        return self.DATA_GENERATOR.flow(self.TRAIN,
                                        self.TRAIN_LABELS,
                                        batch_size=self.BATCH_SIZE)
