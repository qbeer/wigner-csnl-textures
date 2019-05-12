import matplotlib.pyplot as plt
from keras.models import load_model


class ModelTrainer:
    def __init__(self, model, data_generator, loss_fn="normal", lr=1e-3, decay=5e-5, observation_noise=1e-3, beta=1e-12):
        assert loss_fn in str(["normal", "normalDiag", "binary", None]),\
            "Loss function should be in [\'normal\', \'normalDiag\', \'None\' (bernoulli), \'binary\']"
        args = loss_fn, lr, decay, observation_noise, beta
        self.model, self.generator = model.get_compiled_model(*args)
        self.latent_dim = model.latent_dim
        self.data_generator = data_generator
        self.saved = False
        self.model.summary()

    def fit(self, EPOCHS, STEPS, contrast=False):
        try:
            self.history = self.model.fit_generator(
                self.data_generator.flow(self.latent_dim) if not contrast else self.data_generator.contrast_flow(self.latent_dim),
                steps_per_epoch=STEPS,
                verbose=1, epochs=EPOCHS,
                validation_data=self.data_generator.validation_data(self.latent_dim))
        except ValueError:
            pass
            self.history = self.model.fit_generator(
                self.data_generator.flattened_flow(), steps_per_epoch=STEPS,
                verbose=1, epochs=EPOCHS,
                validation_data=self.data_generator.flattened_validation_data())
        finally:
            self._save_model()
            plt.title("Model loss")
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.show()

    def _save_model(self):
        print("Saving the trained inference and generator models...\t", end='')
        self.model.save("model.h5")
        self.generator.save("generator-model.h5")
        self.saved = True
        print("OK!")

    def load_models(self):
        if not self.saved:
            print("The models have not been trained or saved yet.")
            return
        print("Loading models...\t", end='')
        self.model = load_model("model.h5")
        self.generator = load_model("generator-model.h5")
        print("OK!")