import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model, data_generator, loss_fn):
        assert loss_fn in str(["normal", "normalDiag", "bernouilli", "binary"]),\
            "Loss function should be in [\'normal\', \'normalDiag\', \'bernoulli\', \'binary\']"
        self.model, self.generator = model.get_compiled_model(loss_fn=loss_fn)
        self.data_generator = data_generator
        print(self.model.summary())

    def fit(self, EPOCHS, STEPS):
        try:
            self.history = self.model.fit_generator(
                self.data_generator.flow(), steps_per_epoch=STEPS,
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