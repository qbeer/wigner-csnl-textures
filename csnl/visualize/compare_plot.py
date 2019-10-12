import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from keras.utils import to_categorical
from scipy.stats import pearsonr
import os


class VAEPlotter:
    def __init__(self, fitted_model, datagen, label_datagen, grid_size=14):
        self.model = fitted_model.model
        self.generator_model = fitted_model.generator
        self.latent_model = fitted_model.latent_model
        self._latent_dim = fitted_model.latent_dim
        self.datagen = datagen
        self.label_datagen = label_datagen
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
            self.batch_size, *self.input_shape[1:]),
                                        batch_size=self.batch_size)

        reco_test = self.model.predict(
            self.test_images[:self.batch_size].reshape(self.batch_size,
                                                       *self.input_shape[1:]),
            batch_size=self.batch_size)

        reco = [reco_train, reco_test]
        images = [self.train_images, self.test_images]

        fig = plt.figure(figsize=(14, 7))
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

        for outer_ind in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(
                4,
                4,
                subplot_spec=outer[outer_ind],
                wspace=0.095,
                hspace=0.095)
            for inner_ind in range(16):
                ax = plt.Subplot(fig, inner[inner_ind])
                if inner_ind % 2 == 0:
                    if inner_ind < 4:
                        ax.set_title('Reconstructed')
                    ax.imshow(reco[outer_ind][inner_ind].reshape(
                        *self.image_shape),
                              interpolation='none',
                              vmin=0,
                              vmax=1)
                else:
                    if inner_ind < 4:
                        ax.set_title(' <- Original')
                    ax.imshow(images[outer_ind][inner_ind - 1].reshape(
                        *self.image_shape),
                              interpolation='none',
                              vmin=0,
                              vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

        fig.suptitle("Train and test reconstructions\n\n")
        plt.savefig(os.getcwd() + "/results/reconstrunction_samples.png",
                    dpi=50)
        plt.show()

    def generate_samples(self, vmax=1):
        latent_inputs = np.random.normal(size=(self.grid_size * self.grid_size,
                                               self._latent_dim))

        self._plot_samples(latent_inputs, vmax)

    def visualize_latent(self, axis=0, sweep_from=-1, sweep_to=1, vmax=1):
        sweep = np.linspace(sweep_from, sweep_to,
                            self.grid_size * self.grid_size)
        latent_inputs = np.random.normal(size=(1, self._latent_dim))
        latent_inputs = np.array(latent_inputs.tolist() *
                                 (self.grid_size * self.grid_size))
        latent_inputs = latent_inputs.reshape(self.grid_size * self.grid_size,
                                              self._latent_dim)
        latent_inputs[:, axis] = sweep

        self._plot_samples(latent_inputs, vmax)

    def _plot_samples(self, latent_inputs, vmax):
        recos = self.generator_model.predict(
            latent_inputs[:self.grid_size * self.grid_size].reshape(
                self.grid_size * self.grid_size, self._latent_dim))
        # Output 196 images
        fig, axes = plt.subplots(self.grid_size,
                                 self.grid_size,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(11, 11))

        if np.prod(recos[0].shape) / (28 * 28) != 1:
            recos = recos.reshape(self.grid_size * self.grid_size, 28, 28,
                                  int(np.prod(recos[0].shape) / (28 * 28)))
        else:
            recos = recos.reshape(self.grid_size * self.grid_size, 28, 28)

        for ind, ax in enumerate(axes.flatten()):
            ax.imshow(recos[ind], interpolation='none', vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("Generated samples on %d - dimensional grid" %
                     self._latent_dim)
        plt.savefig(os.getcwd() + "/results/generated_samples.png", dpi=50)
        plt.show()

    def plot_td_bu_values(self, latent_dim1):
        images, _ = next(self.datagen.flow())
        for IMAGE_INDEX in range(images.shape[0]):
            _images = np.copy(images)
            for ind in range(
                    images.shape[0]):  # predict same element of the batch
                _images[ind] = images[IMAGE_INDEX]

            try:
                recos, z1, z1_mean, z1_sigma, z2, z_mean_BU, z_log_var_BU, z_mean_TD, z_log_var_TD = self.latent_model.predict(
                    _images, batch_size=self.batch_size)
            except ValueError:
                recos, z1, z1_mean, z1_sigma, z2, z_mean_BU, z_log_var_BU, z_mean_TD, z_log_var_TD = self.latent_model.predict(
                    _images.reshape(self.batch_size, 28 * 28),
                    batch_size=self.batch_size)

            self._plot_td_bu_comparisons(z_mean_BU, z_log_var_BU, z_mean_TD,
                                         z_log_var_TD, _images, z1_mean,
                                         z1_sigma, recos, IMAGE_INDEX)

    def _plot_td_bu_comparisons(self, z_mean_BU, z_log_var_BU, z_mean_TD,
                                z_log_var_TD, _images, z1_mean, z1_sigma,
                                recos, IMAGE_INDEX):
        mean_and_vars = [[z_mean_BU, z_log_var_BU], [z_mean_TD, z_log_var_TD]]
        names = ['Bottom up values', 'Top down values']
        for img_ind in range(_images.shape[0]):
            fig, axes = plt.subplots(3,
                                     2,
                                     sharex=False,
                                     sharey=False,
                                     figsize=(7, 9))
            for ind in range(0, 2):
                mean, log_var = mean_and_vars[ind]
                axes[ind, 0].hist(z1_mean[img_ind],
                                  bins=10,
                                  alpha=0.2,
                                  label="z1")
                axes[ind, 1].hist(z1_sigma[img_ind],
                                  bins=10,
                                  alpha=0.2,
                                  label="z1")
                axes[ind, 0].hist(mean[img_ind],
                                  bins=10,
                                  alpha=0.2,
                                  label="other")
                axes[ind, 1].hist(np.exp(log_var[img_ind]),
                                  bins=10,
                                  alpha=0.2,
                                  label="other")
                axes[ind, 0].set_title(names[ind] + " - mean")
                axes[ind, 1].set_title(names[ind] + " - sigma")
                axes[ind, 0].legend(loc='upper right')
                axes[ind, 1].legend(loc='upper right')
            axes[2, 0].imshow(_images[img_ind].reshape(28, 28))
            axes[2, 0].set_title('Original image')
            axes[2, 1].imshow(recos[img_ind].reshape(28, 28))
            axes[2, 1].set_title('Reconstructed image')
            fig.tight_layout()
            plt.savefig(os.getcwd() + "/results/%d_TD_BU_COMPS_%d.png" %
                        (IMAGE_INDEX + 1, img_ind + 1),
                        dpi=50)
            plt.close(fig)

    def plot_label_correlations(self):
        images, labels = next(self.label_datagen.flow())
        try:
            recos, z1, z1_mean, z1_sigma, z2, _, _, _, _ = self.latent_model.predict(
                images, batch_size=self.batch_size)
        except ValueError:
            recos, z1, z1_mean, z1_sigma, z2, _, _, _, _ = self.latent_model.predict(
                images.reshape(self.batch_size, 28 * 28),
                batch_size=self.batch_size)

        if np.prod(images[0].shape) / (28 * 28) != 1:
            images = images.reshape(self.batch_size, 28, 28,
                                    int(np.prod(images[0].shape) / (28 * 28)))
        else:
            images = images.reshape(self.batch_size, 28, 28)

        labels = to_categorical(labels)

        for cat in range(labels[0].shape[0]):
            fig = plt.figure(figsize=(10, 8))
            correlations = []

            for i in range(self._latent_dim):
                correlations.append(pearsonr(z2[:, i], labels[:, cat]))

            correlations = np.array(correlations)

            plt.title("#%d one-hot category correlation with z2" % (cat + 1))
            plt.scatter(range(correlations.shape[0]),
                        correlations[:, 0],
                        c=[
                            'b' if x >= 0.4 else 'r'
                            for x in np.abs(correlations[:, 0])
                        ])

            plt.hlines(y=.4, xmin=0, xmax=self._latent_dim)
            plt.hlines(y=-.4, xmin=0, xmax=self._latent_dim)
            plt.xlim(0, self._latent_dim)
            plt.xticks(np.arange(1, self._latent_dim, 2))
            plt.xlabel('Latent parameters')
            plt.ylabel('Pearson-correlation')
            plt.savefig(os.getcwd() + "/results/cat-%d-to-z2-corr.png" %
                        (cat + 1),
                        dpi=50)
            plt.show()

    def plot_contrast_correlations(self, latent_dim2=None):
        def contrast_flow(_flow):
            def train_generator(_it):
                while True:
                    batch_x, batch_y = next(_it)
                    yield _contrast(batch_x)

            return train_generator(_flow)

        def conv_contrast_flow(_flow):
            def train_generator(_it):
                while True:
                    batch_x, batch_y = next(_it)
                    batch_x, batch_y = _contrast(batch_x)
                    yield batch_x.reshape(self.batch_size, 28, 28,
                                          1), batch_y.reshape(
                                              self.batch_size, 28, 28, 1)

            return train_generator(_flow)

        random_contrasts = []

        def _contrast(images):
            contrasted_images = np.zeros(shape=images.shape)
            for ind in range(images.shape[0]):
                contrast = np.random.rand() * 2.
                random_contrasts.append(contrast)
                contrasted_images[ind] = np.clip(
                    contrast * (images[ind] - 0.5) + 0.5, 0, 1)
            return contrasted_images.reshape(self.batch_size,
                                             784), contrasted_images.reshape(
                                                 self.batch_size, 784)

        try:
            reco, z1, z1_mean, z1_sigma, z2, _, _, _, _ = self.latent_model.predict_generator(
                contrast_flow(self.datagen.flow()), steps=50, verbose=1)
        except ValueError:
            reco, z1, _, _, z2, _, _, _, _ = self.latent_model.predict_generator(
                conv_contrast_flow(self.datagen.flow()), steps=50, verbose=1)

        random_contrasts = np.array(random_contrasts).flatten()[:reco.shape[0]]
        reco = reco.reshape(reco.shape[0], 28, 28)

        self._latent_correlation(z2, random_contrasts, "z2")
        self._stats(z2, self._latent_dim, "z2")
        if latent_dim2 != None:
            self._latent_correlation(z1, random_contrasts, "z1", latent_dim2)
            self._stats(z1, latent_dim2, "z1")

    def _latent_correlation(self,
                            z,
                            random_contrasts,
                            latent_name,
                            latent_dim=None):
        fig = plt.figure(figsize=(10, 8))

        correlations = []

        latent_dim = self._latent_dim if latent_dim == None else latent_dim

        for i in range(latent_dim):
            correlations.append(pearsonr(z[:, i], random_contrasts))

        correlations = np.array(correlations)

        plt.title("Contrast correlation with %s" % latent_name)
        plt.scatter(
            range(correlations.shape[0]),
            correlations[:, 0],
            c=['b' if x >= 0.4 else 'r' for x in np.abs(correlations[:, 0])])
        plt.hlines(y=.4, xmin=0, xmax=latent_dim)
        plt.hlines(y=-.4, xmin=0, xmax=latent_dim)
        plt.xlim(0, latent_dim)
        plt.xticks(np.arange(1, latent_dim, 2))
        plt.xlabel('Latent parameters')
        plt.ylabel('Pearson-correlation')
        plt.savefig(os.getcwd() +
                    "/results/contrast-to-%s-corr.png" % latent_name,
                    dpi=50)
        plt.show()

    def _stats(self, z, latent_dim, name):
        fig = plt.figure(figsize=(20, 10))

        plt.errorbar(x=range(latent_dim),
                     y=np.mean(z, axis=0).flatten(),
                     yerr=np.std(z, axis=0).flatten(),
                     ecolor='r',
                     fmt='o')
        plt.xticks(np.linspace(0, latent_dim, 17))
        plt.title('Mean and standard deviation of %s' % name)
        plt.savefig(os.getcwd() + "/results/mean-and-std-of-%s.png" % name,
                    dpi=50)
        plt.show()
