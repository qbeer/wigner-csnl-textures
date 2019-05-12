import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model, data_generator, loss_fn="normal", lr=1e-3, decay=5e-5, observation_noise=1e-3, beta=1e-12):
        assert loss_fn in str(["normal", "normalDiag", "binary", None]),\
            "Loss function should be in [\'normal\', \'normalDiag\', \'None\' (bernoulli), \'binary\']"
        args = loss_fn, lr, decay, observation_noise, beta
        self.model, self.generator = model.get_compiled_model(*args)
        self.latent_dim = model.latent_dim
        self.data_generator = data_generator
        self.model.summary()

    def fit(self, EPOCHS, STEPS, contrast=False):
        try:
            self.history = self.model.fit_generator(
                self.data_generator.flow() if not contrast else self.data_generator.contrast_flow(),
                steps_per_epoch=STEPS,
                verbose=1, epochs=EPOCHS,
                validation_data=self.data_generator.validation_data())
        except ValueError:
            self.history = self.model.fit_generator(
                self.data_generator.flattened_flow(), steps_per_epoch=STEPS,
                verbose=1, epochs=EPOCHS,
                validation_data=self.data_generator.flattened_validation_data())
        finally:
            plt.title("Model loss")
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.show()