import os
import pathlib

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from traffic_sign_classifier_methods import preprocess_image_data, oversample_dataset, plot_dataset_data, \
    plot_number_of_samples_per_class, save_to_pickle_file, load_from_pickle_file, image_transformation_sample


def flattening_layer(input, output_size):
    return tf.reshape(input, [-1, output_size])


def fully_connected(input, input_size, output_size, mu, sigma):
    fc1_W = tf.Variable(tf.truncated_normal(shape=(input_size, output_size), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(output_size))
    return tf.matmul(input, fc1_W) + fc1_b


def LeNet(x, class_n=10, dropout_keep=1.):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    l1_out_depth = 6
    x = tf.nn.conv2d(x,
                     filter=tf.Variable(tf.truncated_normal((5, 5, 1, l1_out_depth))),
                     strides=[1, 1, 1, 1],
                     padding='VALID') + tf.Variable(tf.zeros(l1_out_depth))

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    l2_out_depth = 16
    x = tf.nn.conv2d(x,
                     filter=tf.Variable(tf.truncated_normal((5, 5, l1_out_depth, l2_out_depth))),
                     strides=[1, 1, 1, 1],
                     padding='VALID') + tf.Variable(tf.zeros(l2_out_depth))

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    x = flattening_layer(x, 400)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    x = fully_connected(x, 400, 120, mu, sigma)
    x = tf.nn.dropout(x, keep_prob=dropout_keep)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    x = fully_connected(x, 120, 84, mu, sigma)
    x = tf.nn.dropout(x, keep_prob=dropout_keep)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = fully_connected(x, 84, class_n, mu, sigma)

    return logits


def load_data_from_dir(data_path, img_height, img_width, batch_size):
    data_gen = ImageDataGenerator(rescale=1. / 255.)

    train_data = data_gen.flow_from_directory(data_path,
                                              shuffle=True,
                                              target_size=(img_height, img_width),
                                              color_mode='rgb',
                                              class_mode='categorical',
                                              batch_size=batch_size)

    class_names = train_data.class_indices
    class_count = len(train_data.class_indices)
    print(f"{class_count} classes:")
    print(class_names)

    train_x = []
    train_y = []
    for _ in range(train_data.samples):
        image, labels = next(train_data)
        train_x.append(image[0])
        label = np.argmax(labels[0])
        train_y.append(label)

    train_y = np.array(train_y)
    train_x = np.array(train_x)

    plot_dataset_data(train_x, train_y)
    plot_number_of_samples_per_class(train_y, "data/metrics/label_hist.png")

    X_train, y_train = oversample_dataset(train_x, train_y, 100)

    plot_dataset_data(train_x, train_y, filename="dataset_images_02.png")
    plot_number_of_samples_per_class(y_train, "data/metrics/label_hist_oversamp.png")

    image_transformation_sample(X_train, y_train, save_file="data/metrics/label_hist_oversamp.png", transformed=False)

    X_train = preprocess_image_data(X_train)

    save_to_pickle_file("data/train.p", [X_train, y_train])


def main_loop(load_preset):
    main_dir = os.getcwd()
    data_path = os.path.join(main_dir, "data", "traffic_lights")

    data_dir = pathlib.Path(data_path)
    image_count = len(list(data_dir.glob('*/*.png')))
    print(image_count)

    batch_size = 1
    img_height = 180
    img_width = 180
    channel = 3
    class_number = 4
    epochs = 10

    if not load_preset:
        load_data_from_dir(data_path, img_height, img_width, batch_size)
    X_train, y_train = load_from_pickle_file("data/train.p")

    # define model
    lenet_5_model = keras.models.Sequential([
        keras.layers.Conv2D(6, kernel_size=5, strides=1, activation='tanh',
                            input_shape=(img_height, img_width, channel),
                            padding='same'),  # C1
        keras.layers.AveragePooling2D(),  # S2
        keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),  # C3
        keras.layers.AveragePooling2D(),  # S4
        keras.layers.Flatten(),  # Flatten
        keras.layers.Dense(120, activation='tanh'),  # C5
        keras.layers.Dense(84, activation='tanh'),  # F6
        keras.layers.Dense(class_number, activation='softmax')  # Output layer
    ])
    # fit model
    lenet_5_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    lenet_5_model.fit(X_train, y_train, epochs=epochs)

    lenet_5_model.save("tl_classifier_leenet5.h5")


def visualize_data(train_ds):
    class_names = train_ds.class_indices
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def test_model(path):
    model = keras.models.load_model(path)
    X_train, y_train = load_from_pickle_file("data/train.p")
    image = np.asarray(X_train[0])
    image = np.expand_dims(image, axis=0)
    prediction = np.argmax(model.predict(image))
    print(f"Predicted {prediction} - label {y_train[0]}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_model("tl_classifier_leenet5.h5")

    print(f"Tensoflow version: {tf.__version__}")
    assert float(tf.__version__.rsplit(".", 1)[0]) < 2, "Tensorflow must be version 1.x to run this notebook"

    test_run = False
    preprocess_data = False

    main_loop(load_preset=False)
