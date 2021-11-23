# Tensorflow/Keras related imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from keras_lr_finder import LRFinder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dropout, AveragePooling2D

'''
Here is the implementation of VGG16, VGG19 - with Keras and
Resnet50 and Densenet161 with Pytorch. Layers of base model were set to non trainable and then set as trainable to compare the performance of the models.
There are some visualization methods as well. Samples visualization was performed only with keras, to omit repetitions, yet augmentation visualization was performed using both keras and torch libraries.
Optimal learning rate for all models was implemented to determine visually and to be set on user choice.(Leslie N. Smith method - https://arxiv.org/abs/1506.01186)
All visualizations were stored to the corresponding folder(*project_folder/results/**library)
*location of all of the files for the project
**library which you are using - keras/torch
'''


class ImageClassificationTF:
    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size

    def visualization(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path + '/train',
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=self.image_size,
            batch_size=32,
        )
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
                plt.suptitle('Dataset visualization with labels')

        plt.savefig('results/keras/dataset_visualization.jpg')

        train_ds = train_ds.prefetch(buffer_size=32)
        return train_ds

    def test_train_val(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            height_shift_range=0.1,
            validation_split=0.2,
            rotation_range=30,
            brightness_range=(0.5, 1.1),
            fill_mode='nearest')

        self.train_generator = train_datagen.flow_from_directory(
            self.path + '/train',
            target_size=self.image_size,
            batch_size=24,
            class_mode='binary',
            subset='training')

        self.validation_generator = train_datagen.flow_from_directory(
            self.path + '/train',
            target_size=self.image_size,
            batch_size=16,
            class_mode='binary',
            subset='validation')

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_pictures = test_datagen.flow_from_directory(self.path + '/test/', target_size=self.image_size,
                                                              batch_size=32,
                                                              class_mode='binary')
        return self.train_generator, self.validation_generator, self.test_pictures

    def augumented_visualization(self):
        imgs, labels = next(self.train_generator)
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(imgs, axes):
            ax.imshow(img)
            ax.axis('off')
            plt.suptitle('Examples of data augmentation')
        plt.tight_layout()
        plt.savefig('results/keras/augmentation_examples.png')
        plt.show()

    def customized_model(self, image_size, num_classes):
        inputs = keras.Input(shape=image_size + (3,))
        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside residual
        for self.size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(self.size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(self.size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
            # Project residual
            residual = layers.Conv2D(self.size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        customized_model = keras.Model(inputs, outputs)

        return customized_model

    def pretrained_model(self, model, unfreeze):
        if model == 'resnet':
            baseModel = ResNet50(include_top=False, input_shape=self.image_size + (3,), weights='imagenet')
            for layer in baseModel.layers[:]:
                if unfreeze == 1:
                    layer.trainable = True
                elif unfreeze == 0:
                    layer.trainable = False
            headModel = baseModel.output
            headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
            headModel = Flatten(name="flatten")(headModel)
            headModel = Dense(256, activation="relu")(headModel)
            headModel = Dropout(0.5)(headModel)
            headModel = Dense(1, activation="sigmoid")(headModel)
            customized_model = Model(inputs=baseModel.input, outputs=headModel)

        elif model == 'VGG19':
            model = VGG19(include_top=False, input_shape=self.image_size + (3,), weights='imagenet')
            customized_model = Sequential()

        # Add layers of base model
            for layer in model.layers[:]:
                if unfreeze == 1:
                    layer.trainable = True
                    customized_model.add(layer)
                elif unfreeze == 0:
                    layer.trainable = False
                    customized_model.add(layer)

                else:
                    raise ValueError('Please choose either 0 - frozen layers or 1 - trainable layers')

            customized_model.add(layers.Flatten())
            customized_model.add(tf.keras.layers.Dropout(0.5))

        # Add last layer with sigmoid activation function since we have 2 classes
            customized_model.add(Dense(units=1, activation='sigmoid'))

        return customized_model

    def training_process(self, model, epochs, train_ds, val_ds, test_set):
        model.summary()  # model layers+info

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(),  # most popular optimizer at the moment
            loss="binary_crossentropy",  # loss for classification problem
            metrics=["accuracy"],# required metric
        )

        lr_finder = LRFinder(model)

        # Train a model with batches
        # with learning rate growing exponentially from 0.0001 to 1
        lr_finder.find_generator(train_ds, start_lr=0.000001, end_lr=1, epochs=50)
        # Plot the loss, ignore 20 batches in the beginning and 5 in the end
        lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
        plt.savefig('results/keras/plot_loss.png')
        plt.show()

        # Plot rate of change of the loss
        # Ignore 20 batches in the beginning and 5 in the end
        # Smooth the curve using simple moving average of 20 batches
        # Limit the range for y axis to (-0.02, 0.01)
        lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
        plt.savefig('results/keras/plot_loss_change.png')
        plt.show()

        # Now we are changing learning rate per visualization and recompile the model - fastest decrease on the graph
        learning_rate = float(input('Please type in learning rate per visualization '))
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),  # most popular optimizer at the moment
            loss="binary_crossentropy",  # loss for classification problem
            metrics=["accuracy"],  # required metric
        )
        model.summary()

        # Fit model with data
        history = model.fit_generator(train_ds, epochs=epochs,
                                      validation_data=val_ds)
        # Model results visualization

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('results/keras/accuracy.png')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('results/keras/loss.png')
        plt.show()
        score = model.evaluate(test_set, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

