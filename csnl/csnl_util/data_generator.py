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

    def contrast_flow(self):
        def train_generator(_it):
            while True:
                batch_x, batch_y = next(_it)
                yield (self._contrast(batch_x), batch_y)
        print("WAS RUN!")
        return train_generator(self.flow())

    def flattened_flow(self):
        image_dim = np.prod(self.IMAGE_SHAPE)

        def train_generator(_it):
            while True:
                batch_x, batch_y = next(_it)
                yield batch_x.reshape(self.BATCH_SIZE, image_dim), batch_y.reshape(self.BATCH_SIZE, image_dim)
        return train_generator(self.flow())

    def flattened_contrast_flow(self):
        image_dim = np.prod(self.IMAGE_SHAPE)

        def train_generator(_it):
            while True:
                batch_x, batch_y = next(_it)
                batch_x, batch_y = self._contrast(
                    batch_x), self._contrast(batch_y)
                yield batch_x.reshape(self.BATCH_SIZE, image_dim), batch_y.reshape(self.BATCH_SIZE, image_dim)

        return train_generator(self.flow())

    def _contrast(self, images):
        contrasted_images = np.zeros(shape=images.shape)
        for ind in range(images.shape[0]):

            contrasted_images[ind] = np.clip(
                np.random.rand() * 2. * (images[ind] - 0.5) + 0.5, 0, 1)
        return contrasted_images

    def validation_data(self):
        return self.TEST, self.TEST

    def flattened_validation_data(self):
        image_dim = np.prod(self.IMAGE_SHAPE)
        reshaped_test = self.TEST.reshape(self.TEST.shape[0], image_dim)
        return reshaped_test, reshaped_test
