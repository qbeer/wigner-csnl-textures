import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class VAEPlotter:
    def __init__(self, fitted_model, datagen, grid_size=14):
        self.model = fitted_model.model
        self.generator_model = fitted_model.generator
        self._latent_dim = fitted_model.latent_dim
        self.datagen = datagen
        self.grid_size = grid_size
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
                    if inner_ind < 4:
                        ax.set_title('Reconstructed')
                    ax.imshow(reco[outer_ind][inner_ind].reshape(
                        *self.image_shape),
                        interpolation='none', vmin=0, vmax=1)
                else:
                    if inner_ind < 4:
                        ax.set_title(' <- Original')
                    ax.imshow(images[outer_ind][inner_ind -
                                                1].reshape(*self.image_shape),
                              interpolation='none', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

        fig.suptitle("Train and test reconstructions\n\n")
        plt.show()

    def generate_samples(self, vmax=1):
        latent_inputs = np.random.normal(size=(self.grid_size*self.grid_size, self._latent_dim))

        self._plot_samples(latent_inputs, vmax)

    def visualize_latent(self, axis=0, sweep_from=-1, sweep_to=1, vmax=1):
        sweep = np.linspace(sweep_from, sweep_to, self.grid_size*self.grid_size)
        latent_inputs = np.random.normal(size=(1, self._latent_dim))
        latent_inputs = np.array(latent_inputs.tolist() * (self.grid_size*self.grid_size))
        latent_inputs = latent_inputs.reshape(self.grid_size*self.grid_size, self._latent_dim)
        latent_inputs[:, axis] = sweep

        self._plot_samples(latent_inputs, vmax)

    def _plot_samples(self, latent_inputs, vmax):
        recos = self.generator_model.predict(
            latent_inputs[:self.grid_size*self.grid_size].reshape(self.grid_size*self.grid_size, self._latent_dim))
        # Output 196 images
        fig, axes = plt.subplots(
            self.grid_size, self.grid_size, sharex=True, sharey=True, figsize=(11, 11))

        if np.prod(recos[0].shape) / (28*28) != 1:
            recos = recos.reshape(
                self.grid_size*self.grid_size, 28, 28, int(np.prod(recos[0].shape) / (28*28)))
        else:
            recos = recos.reshape(self.grid_size*self.grid_size, 28, 28)

        for ind, ax in enumerate(axes.flatten()):
            ax.imshow(recos[ind], interpolation='none', vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("Generated samples on %d - dimensional grid" %
                     self._latent_dim)
        plt.show()
