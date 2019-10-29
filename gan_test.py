# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

NOISE_DIM = 16
N_EXAMPLES = 16

generator_noise_vector = tfd.Normal(loc=[0.] * NOISE_DIM,
                                    scale=[1.] * NOISE_DIM).sample(
                                        [N_EXAMPLES])

generator = tf.keras.models.Sequential(layers=[
    tf.keras.layers.Dense(7 * 7 * 32, input_shape=(NOISE_DIM, )),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Reshape((7, 7, 32)),
    tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(1,
                                                         1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2,
                                                         2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2,
                                                         2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2DTranspose(
        1, (2, 2), strides=(1, 1), padding='same', activation='sigmoid')
])

discriminator = tf.keras.models.Sequential(layers=[
    tf.keras.layers.Conv2D(
        64, (2, 2), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])


def generator_loss(generated):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated), generated)


def discriminator_loss(real, generated):
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real), logits=real)
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated), logits=generated)
    total_loss = real_loss + generated_loss
    return total_loss


def train_step(images, batch_size, noise_dim=NOISE_DIM):
    noise_vec = tfd.Normal(loc=[0.] * noise_dim,
                           scale=[1.] * noise_dim).sample([batch_size])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noise_vec)

        real_out = discriminator(images)
        gen_out = discriminator(generated)

        gen_loss = generator_loss(gen_out)
        disc_loss = discriminator_loss(real_out, gen_out)

    grads_of_gen = gen_tape.gradient(gen_loss, generator.variables)
    grads_of_disc = disc_tape.gradient(disc_loss, discriminator.variables)

    tf.train.AdamOptimizer(1e-4).apply_gradients(
        zip(grads_of_gen, generator.variables))
    tf.train.AdamOptimizer(1e-4).apply_gradients(
        zip(grads_of_disc, discriminator.variables))


def generate_and_save_images(current_epoch):
    generated = generator(generator_noise_vector)
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(10, 10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(generated[ind].numpy().reshape(28, 28), vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()
    plt.savefig('gan_images/generated_after_epoch_%d.png' % current_epoch)


def train(dataset, batch_size, epochs=50, steps=100):
    for epoch in range(epochs):
        for _ in range(steps):
            train_step(next(dataset), batch_size)

        generate_and_save_images(epoch + 1)


import os
from csnl import DataGenerator

data_gen_labels = DataGenerator(image_shape=(28, 28, 1),
                                batch_size=100,
                                file_path=os.getcwd() +
                                '/csnl/data/textures_42000_28px.pkl',
                                contrast_normalize=False)

data_iterator = data_gen_labels.flow()

train(data_iterator, batch_size=100, epochs=50, steps=100)