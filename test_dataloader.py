from csnl import DataGenerator, DataGeneratorWithLabels
import os

datagen = DataGenerator(image_shape=(28, 28, 1),
                              batch_size=100,
                              file_path=os.getcwd() +
                              '/csnl/data/textures_42000_28px.pkl',
                              whiten=False,
                              contrast_normalize=True)

print('\n\n\n')

label_datagen = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                        batch_size=100,
                        file_path=os.getcwd() +
                        '/csnl/data/textures_42000_28px.pkl',
                        whiten=False,
                        contrast_normalize=True)

import matplotlib.pyplot as plt

images, labels = next(datagen.flow())

fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(10, 10))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(images[ind].reshape(28, 28), vmin=0, vmax=1)

fig.tight_layout()
plt.savefig(os.getcwd() + "/results/contrast_normalized.png", dpi=200)
plt.show()