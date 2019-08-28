from csnl import DataGenerator

datagen = DataGenerator(image_shape=(28, 28, 1),
                        batch_size=70,
                        file_path=os.getcwd() +
                        '/csnl/data/scramtex_700_28px.pkl',
                        whiten=False, contrast_normalize=True)
