# import all libraries
from IPython.display import display, Image
import numpy as np
from keras.models import Sequential  # initialize NN as a Sequence
from keras.layers import Convolution2D  # to deal with images
from keras.layers import MaxPool2D  # this will add pooling layers
from keras.layers import Flatten  # this will flatten pooling layers
from keras.layers import Dense  # create NN
from keras.preprocessing.image import ImageDataGenerator


class CNN:

    def __init__(self):
        pass

    classifier = Sequential()

    # create convolutional layer
    def addConvolutionalLayer(self):
        self.classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
        # feature detectors , filter 3 by 3 ,
        # input shape - format of pictures that are inputed , depends if there are coloured images or black and white
        # number of channels - 3
        # 64 , 64 - parameters of pictures
        # activation function - rectify

    def maxPooling(self):
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))
        # we take a size 2,2 of stride

    def flattening(self):
        self.classifier.add(Flatten())

    def fullConnection(self):
        self.classifier.add(Dense(output_dim=128, activation='relu'))
        self.classifier.add(
            Dense(output_dim=1, activation='sigmoid'))  # sigmoid function becouse there is a binary outcome
        # output_dim - how many nodes will be

    def compile(self):
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics={"named_accuracy" : 'accuracy'})

        # stochastic discent - adam
        # loss - binary cross entropy , becouse it is a classification problem
        # metrics - accuracy

        # using image augmentation to prevent overfitting

    def trainTest(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            'data/training_set',
            target_size=(64, 64),
            batch_size=32,  # number of images that will go threw and weights will be updated
            class_mode='binary')

        test_set = test_datagen.flow_from_directory(
            'data/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

        self.classifier.fit_generator(
            train_generator,
            steps_per_epoch=200,
            epochs=5,
            validation_data=test_set,
            validation_steps=2000)
