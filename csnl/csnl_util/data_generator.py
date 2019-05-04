from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .data_loader import DataLoader


class DataGenerator:
    def __init__(self, image_shape, batch_size, file_path='/data/scramtex_700_28px.pkl', binarize=False):
        self.DATA_LOADER = DataLoader(file_path, binarize)
        self.IMAGE_SHAPE = image_shape
        self.BATCH_SIZE = batch_size

        self.DATA_GENERATOR = ImageDataGenerator()

        self.TRAIN, self.TEST = self.DATA_LOADER.train_test_split()

    def flow(self):
        return self.DATA_GENERATOR.flow(self.TRAIN, self.TRAIN, batch_size=self.BATCH_SIZE)

    def flattened_flow(self):
        def train_generator(_it):
            image_dim = np.prod(self.IMAGE_SHAPE)
            while True:
                batch_x, batch_y = next(_it)
                yield batch_x.reshape(batch_x.shape[0], image_dim), batch_y.reshape(batch_y.shape[0], image_dim)
        return train_generator(self.flow())

    def validation_data(self):
        return self.TEST, self.TEST

    def flattened_validation_data(self):
        image_dim = np.prod(self.IMAGE_SHAPE)
        reshaped_test = self.TEST.reshape(self.TEST.shape[0], image_dim)
        return reshaped_test, reshaped_test
