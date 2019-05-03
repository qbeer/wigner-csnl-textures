import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class VAEPlotter:
    def __init__(self, fitted_model, datagen):
        self.model = fitted_model.model
        self.generator_model = fitted_model.generator
        self._latent_dim = fitted_model.latent_dim
        self.datagen = datagen
        self.train_images, _ = next(self.datagen.flow())
        self.test_images, _ = self.datagen.validation_data()
        self.batch_size = self.train_images.shape[0]
        self.input_shape = self.model.layers[0].output_shape
        self.image_shape = self.train_images.shape[1:]
        if self.image_shape[2] == 1:
            self.image_shape = self.image_shape[:-1]

    def grid_plot(self):
        reco_train = self.model.predict(self.train_images.reshape(
            self.batch_size, *self.input_shape[1:]), batch_size=self.batch_size)

        reco_test = self.model.predict(self.test_images[:self.batch_size].reshape(
            self.batch_size, *self.input_shape[1:]), batch_size=self.batch_size)

        reco = [reco_train, reco_test]
        images = [self.train_images, self.test_images]

        fig = plt.figure(figsize=(14, 7))
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

        for outer_ind in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(4, 4,
                                                     subplot_spec=outer[outer_ind], wspace=0.095, hspace=0.095)
            for inner_ind in range(16):
                ax = plt.Subplot(fig, inner[inner_ind])
                if inner_ind % 2 == 0:
                    if inner_ind < 4: ax.set_title('Reconstructed')
                    ax.imshow(reco[outer_ind][inner_ind].reshape(
                        *self.image_shape), interpolation=None)
                else:
                    if inner_ind < 4: ax.set_title(' <- Original')
                    ax.imshow(images[outer_ind][inner_ind -
                                                1].reshape(*self.image_shape), interpolation=None)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

        fig.suptitle("Train and test reconstructions\n\n")
        plt.show()

    def generate_samples(self, subdivision=10):
        starts = np.linspace(-1, 1, self._latent_dim)
        stops = np.linspace(-1, 1, self._latent_dim)
        latent_inputs = self._ndim_grid(starts, stops, subdivision)

        assert len(latent_inputs) > 63, "Latent inputs should be at least 64."
        recos = self.generator_model.predict(latent_inputs[:self.batch_size].reshape(self.batch_size, self._latent_dim))

        # Output 64 images
        fig, axes = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(8, 8))

        for ind, ax in enumerate(axes.flatten()):
            ax.imshow(recos[ind].reshape(*self.image_shape))
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("Generated samples on %d - dimensional grid" % self._latent_dim)
        plt.show()

    def _ndim_grid(self, starts, stops, subdivision):
        # Set number of dimensions
        ndims = len(starts)

        # List of ranges across all dimensions
        L = [np.linspace(starts[i], stops[i], subdivision) for i in range(ndims)]

        # Finally use meshgrid to form all combinations corresponding to all 
        # dimensions and stack them as M x ndims array
        return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T