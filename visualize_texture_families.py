from csnl import DataGeneratorWithLabels
import os

data_gen = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                   batch_size=100,
                                   file_path=os.getcwd() +
                                   '/csnl/data/textures_42000_28px.pkl',
                                   contrast_normalize=False)

default, labels = next(data_gen.flow())
default_contrast, labels_contrast = next(data_gen.contrast_flow())

data_gen = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                   batch_size=100,
                                   file_path=os.getcwd() +
                                   '/csnl/data/textures_42000_28px.pkl',
                                   contrast_normalize=True)

normalized, norm_labels = next(data_gen.flow())
normalized_contrast, norm_labels_contrast = next(data_gen.contrast_flow())

import matplotlib.pyplot as plt
import numpy as np


def plot_uniqe(index_arr, image_arr, plot_name="default.png"):
    _, indices = np.unique(index_arr, return_index=True)
    fig, axes = plt.subplots(1,
                             indices.shape[0],
                             figsize=(7, 2),
                             sharex=True,
                             sharey=True)
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(image_arr[indices[ind]].reshape(28, 28),
                  vmin=0,
                  vmax=1,
                  interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Label : %d " % (index_arr[indices[ind]] + 1))
    fig.tight_layout()
    plt.savefig(plot_name)
    plt.close()


plot_uniqe(labels, default)
plot_uniqe(labels_contrast, default_contrast, plot_name="default_contrast.png")
plot_uniqe(norm_labels, normalized, plot_name="normalized.png")
plot_uniqe(norm_labels_contrast,
           normalized_contrast,
           plot_name="normalized_contrast.png")
