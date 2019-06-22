from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio


class GifCallBack(Callback):
    def __init__(self, datagen, generator, latent_dim, grid_size=4):
        self.datagen = datagen
        self.generator = generator
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.latent_inputs = np.random.normal(
            size=(grid_size*grid_size, latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        self._make_gif(epoch)

    def on_train_end(self, logs=None):
        images = []
        for filename in os.listdir(os.getcwd() + '/tmp'):
            images.append(imageio.imread(os.getcwd() + '/tmp/' + filename))
            os.remove(os.getcwd() + '/tmp/' + filename)
        imageio.mimsave(os.getcwd() + '/latent.gif', images)
        os.rmdir(os.getcwd() + '/tmp')

    def _make_gif(self, epoch):
        recos = self._plot_samples(self.latent_inputs)

        fig, axes = plt.subplots(
            self.grid_size, self.grid_size, sharex=True, sharey=True, figsize=(11, 11))

        for ind, ax in enumerate(axes.flatten()):
            ax.imshow(recos[ind], interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.savefig(os.getcwd() + "/tmp/" + str(epoch) + ".png", dpi=100)
        plt.close(fig)

    def _plot_samples(self, latent_inputs):
        recos = self.generator.predict(
            latent_inputs[:self.grid_size*self.grid_size].reshape(self.grid_size**2, self.latent_dim))

        if np.prod(recos[0].shape) / (28*28) != 1:
            recos = recos.reshape(
                self.grid_size**2, 28, 28, int(np.prod(recos[0].shape) / (28*28)))
        else:
            recos = recos.reshape(self.grid_size**2, 28, 28)

        return recos

    def _make_on_train_start(self):
        os.mkdir(os.getcwd() + '/tmp')

    def _remove_on_error(self):
        for filename in os.listdir(os.getcwd() + '/tmp'):
            os.remove(os.getcwd() + '/tmp/' + filename)
        os.rmdir(os.getcwd() + '/tmp')
