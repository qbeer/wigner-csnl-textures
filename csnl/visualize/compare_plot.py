import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class VAEPlotter:
    def __init__(self, model, generator_model, datagen):
        self.model = model
        self.generator_model = generator_model
        self.datagen = datagen

    def grid_plot(self):
        input_shape = self.model.layers[0].output_shape

        train_images, _ = next(self.datagen.flow())
        test_images, _ = self.datagen.validation_data()

        batch_size = train_images.shape[0]

        reco_train = self.model.predict(train_images.reshape(
            *input_shape), batch_size=batch_size)

        reco_test = self.model.predict(test_images[:batch_size].reshape(
            *input_shape), batch_size=batch_size)

        reco = [reco_train, reco_test]
        images = [train_images, test_images]

        fig = plt.figure(figsize=(14, 7))
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

        for outer_ind in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(4, 4,
                                                     subplot_spec=outer[outer_ind], wspace=0.095, hspace=0.095)
            for inner_ind in range(16):
                ax = plt.Subplot(fig, inner[inner_ind])
                if inner_ind % 2 == 0:
                    ax.set_title('Reconstructed')
                    ax.imshow(reco[outer_ind][inner_ind].reshape(
                        28, 28), interpolation=None)
                else:
                    ax.set_title(' <- Original')
                    ax.imshow(images[outer_ind][inner_ind -
                                                1].reshape(28, 28), interpolation=None)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

        fig.suptitle("Train and test reconstructions\n\n")
        plt.show()
