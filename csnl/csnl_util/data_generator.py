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

    def flow(self, latent_dim):
        def train_generator(_it):
            while True:
                batch_x, batch_y = next(_it)
                val = np.ones(shape=(self.BATCH_SIZE, latent_dim))
                yield batch_x, {"target" : batch_y, "latent" : val}
        return train_generator(self.DATA_GENERATOR.flow(self.TRAIN, self.TRAIN, batch_size=self.BATCH_SIZE))

    def contrast_flow(self, latent_dim):
        def train_generator(_it):
            while True:
                batch_x, target_dict = next(_it)
                batch_y = target_dict["target"]
                val = target_dict["latent"]
                contrast = np.random.rand()
                _batch_x, _batch_y = contrast*batch_x, contrast*batch_y
                _batch_x, _batch_y = np.clip(_batch_x, 0, 1), np.clip(_batch_y, 0, 1)
                yield _batch_x, {"target" : _batch_y, "latent" : val}
        return train_generator(self.flow(latent_dim))

    def flattened_flow(self, latent_dim):
        def train_generator(_it):
            image_dim = np.prod(self.IMAGE_SHAPE)
            while True:
                batch_x, (batch_y, val) = next(_it)
                yield batch_x.reshape(batch_x.shape[0], image_dim), batch_y.reshape(batch_y.shape[0], image_dim), val
        return train_generator(self.flow(latent_dim))

    def validation_data(self, latent_dim):
        return self.TEST, {"target" : self.TEST, "latent" : np.ones(shape=(self.TEST.shape[0], latent_dim))}

    def flattened_validation_data(self, latent_dim):
        image_dim = np.prod(self.IMAGE_SHAPE)
        reshaped_test = self.TEST.reshape(self.TEST.shape[0], image_dim)
        return reshaped_test, reshaped_test, np.ones(shape=(self.BATCH_SIZE, latent_dim))
