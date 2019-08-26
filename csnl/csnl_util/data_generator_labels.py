from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .data_loader import DataLoader


class DataGeneratorWithLabels:
    def __init__(self,
                 image_shape,
                 batch_size,
                 file_path='/data/scramtex_700_28px.pkl',
                 binarize=False,
                 whiten=False):
        self.DATA_LOADER = DataLoader(file_path, binarize, whiten)
        self.IMAGE_SHAPE = image_shape
        self.BATCH_SIZE = batch_size
        self.DATA_GENERATOR = ImageDataGenerator()
        self.TRAIN, self.TRAIN_LABEL, _, _ = self.DATA_LOADER.get_data_and_labels(
        )
        print("Train SHAPE : ", self.TRAIN.shape)

    def flow(self):
        return self.DATA_GENERATOR.flow(self.TRAIN,
                                        self.TRAIN_LABEL,
                                        batch_size=self.BATCH_SIZE)
