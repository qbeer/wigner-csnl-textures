import matplotlib.pyplot as plt
from keras.models import load_model
import os
from ..visualize import GifCallBack
from ..csnl_util import IncrementalBeta


class ModelTrainer:
    def __init__(self, model, data_generator, loss_fn="normal", lr=1e-7, decay=5e-5, observation_noise=1e-3, beta=1e-12):
        assert loss_fn in str(["normal", "normalDiag", "binary", "bernoulli"]),\
            "Loss function should be in [\'normal\', \'normalDiag\', \'bernoulli\', \'binary\']"
        args = loss_fn, lr, decay, observation_noise, beta
        self._model = model
        self.model, self.generator, self.latent_model = self._model.get_compiled_model(
            *args)
        self.latent_dim = model.latent_dim
        self.data_generator = data_generator
        self.saved = False
        self.gifCallBack = GifCallBack(
            self.data_generator, self.generator, self.latent_dim)
        self.betaCallback = IncrementalBeta(beta)
        self.model.summary()

    def fit(self, EPOCHS, STEPS, contrast=False):
        self.gifCallBack._make_on_train_start()
        try:
            self.history = self.model.fit_generator(
                self.data_generator.flow() if not contrast else self.data_generator.contrast_flow(),
                steps_per_epoch=STEPS,
                verbose=1, epochs=EPOCHS,
                validation_data=self.data_generator.validation_data(), callbacks=[self.gifCallBack, self.betaCallback])
        except ValueError:
            try:
                self.history = self.model.fit_generator(
                    self.data_generator.flattened_flow() if not contrast else self.data_generator.flattened_contrast_flow(), steps_per_epoch=STEPS,
                    verbose=1, epochs=EPOCHS,
                    validation_data=self.data_generator.flattened_validation_data(), callbacks=[self.gifCallBack, self.betaCallback])
            except ValueError:
                self.gifCallBack._remove_on_error()
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
        print("Saving the trained inference, generator and latent models...\t", end='')
        self.model.save(os.getcwd() + "/model.h5")
        self.generator.save(os.getcwd() + "/generator_model.h5")
        """
            This must be acquired here before we are after training at this point!
        """
        self.latent_model.save(os.getcwd() + "/latent_model.h5")
        self.saved = True
        print("OK!")
