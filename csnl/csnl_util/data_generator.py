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
                contrast = np.random.rand(batch_x.shape[0]) * 2.
                _batch_x, _batch_y = np.split(np.array(
                    [[cont*batch_x[ind], cont*batch_y[ind]] for ind, cont in enumerate(contrast)]), 2, axis=1)
                _batch_x, _batch_y = _batch_x.reshape(_batch_x.shape[0], *_batch_x.shape[2:]), \
                    _batch_y.reshape(_batch_y.shape[0], *_batch_y.shape[2:])
                _batch_x, _batch_y = np.clip(
                    _batch_x, 0, 1), np.clip(_batch_y, 0, 1)
                yield _batch_x, _batch_y
        return train_generator(self.flow())

    def flattened_flow(self):
        return self._flat_train_generator(self.flow())

    def flattened_contrast_flow(self):
        return self._flat_train_generator(self.contrast_flow())

    def _flat_train_generator(self, _it):
        image_dim = np.prod(self.IMAGE_SHAPE)
        while True:
            batch_x, batch_y = next(_it)
            yield batch_x.reshape(self.BATCH_SIZE, image_dim), batch_y.reshape(self.BATCH_SIZE, image_dim)

    def _contrast(self, img, alpha, brightness=0):
        return np.clip(alpha*(img - 0.5) + 0.5 + brightness, 0, 1)

    def validation_data(self):
        return self.TEST, self.TEST

    def flattened_validation_data(self):
        image_dim = np.prod(self.IMAGE_SHAPE)
        reshaped_test = self.TEST.reshape(self.TEST.shape[0], image_dim)
        return reshaped_test, reshaped_test
