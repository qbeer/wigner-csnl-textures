import matplotlib.pyplot as plt


class VAEPlotter:
    def __init__(self, model, generator_model, datagen):
        self.model = model
        self.generator_model = generator_model
        self.datagen = datagen

    def grid_plot(self):
        train_images, _ = next(self.datagen.flow())
        test_images, _ = self.datagen.validation_data()
        batch_size = train_images.shape[0]

        reco = self.model.predict(train_images.reshape(
            batch_size, 28, 28, 1), batch_size=batch_size)

        fig, axes = plt.subplots(
            4, 4, sharex=True, sharey=True, figsize=(7, 10))

        for ind, ax in enumerate(axes.flatten()):
            if ind % 2 == 0:
                ax.set_title('Reconstructed')
                ax.imshow(reco[ind].reshape(28, 28), interpolation=None)
            else:
                ax.set_title(' <- Original')
                ax.imshow(
                    train_images[ind - 1].reshape(28, 28), interpolation=None)

        fig.suptitle("Train reconstructions\n\n")
        fig.tight_layout()
        plt.show()

        reco = self.model.predict(test_images[:batch_size].reshape(
            batch_size, 28, 28, 1), batch_size=batch_size)

        fig, axes = plt.subplots(
            4, 4, sharex=True, sharey=True, figsize=(7, 10))

        for ind, ax in enumerate(axes.flatten()):
            if ind % 2 == 0:
                ax.set_title('Reconstructed')
                ax.imshow(reco[ind].reshape(28, 28), interpolation=None)
            else:
                ax.set_title(' <- Original')
                ax.imshow(
                    test_images[ind - 1].reshape(28, 28), interpolation=None)

        fig.suptitle("Test reconstructions\n\n")
        fig.tight_layout()
        plt.show()
