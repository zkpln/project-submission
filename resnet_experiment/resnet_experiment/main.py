#!/usr/bin/env python3


import argparse
import sys

import tensorflow as tf
from tensorflow import keras


def main(argv=sys.argv[1:]):
    argument_parser = argparse.ArgumentParser()

    args = argument_parser.parse_args(argv)

    # Right now, this is just follows the sample code, but will
    # eventually be experiementation for our project

    # Most of this is from the CSE 4/574 NN_Example.py example code and
    # TensorFlow's documentation
    # https://piazza.com/class_profile/get_resource/k5obys2ajh66dx/k83psp6rubdj3
    # https://www.tensorflow.org/tutorials/keras/classification

    ## load the data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    ## preprocess data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    ## define the model
    model = keras.Sequential([
        ## input layer
        keras.layers.Flatten(input_shape=(28, 28)),

        ## middle layer
        keras.layers.Dense(128, activation=tf.nn.relu),
        # # TensorFlow's documentation:
        # keras.layers.Dense(128, activation='relu'),

        ## output/categories layer
        # CSE 4/574 NN_Example.py
        keras.layers.Dense(10, activation=tf.nn.softmax),
        # # TensorFlow's documentation:
        # keras.layers.Dense(10),
    ])
    model.compile(
        optimizer='adam',
        # CSE 4/574 NN_Example.py:
        loss='sparse_categorical_crossentropy',
        # # TensorFlow's documentation:
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    ## Train the model

    # CSE 4/574 NN_Example.py
    model.fit(train_images, train_labels, epochs=5)
    # # TensorFlow's documentation:
    # model.fit(train_images, train_labels, epochs=10)

    ## evaluate the model

    # CSE 4/574 NN_Example.py
    acc = model.evaluate(test_images, test_labels)
    print(acc)

    # TensorFlow's documentation:
    test_loss, test_accuracy = model.evaluate(
        test_images,
        test_labels,
        verbose=2,
    )

    # print("Hello, World!")


if __name__ == '__main__':
    main()

