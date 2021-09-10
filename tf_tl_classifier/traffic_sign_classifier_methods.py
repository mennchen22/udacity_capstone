import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Save to pickel file for later testing

def save_to_pickle_file(save_file: str, data):
    print("[Saving] To file {}".format(save_file))
    if os.path.exists(save_file):
        print("[Warning] Overwrite old pickle file!")
        os.remove(save_file)
    with open(save_file, 'ab') as f:
        pickle.dump(data, f)


def load_from_pickle_file(load_file: str):
    print(f"[Loading] Try to load {load_file}")
    if os.path.exists(load_file):
        with open(load_file, 'rb') as file:
            print("[Loading] Found file storage. Load existing one")
            return pickle.loads(file.read())
    print("[Loading] Failed!")


def rotate_image(image, angle_deg):
    row, col = image.shape[:2]
    image_center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    rot_img = cv2.warpAffine(image, rot_mat, (col, row))
    return rot_img


def translate_image(image, x, y):
    row, col = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, translation_matrix, (col, row))


def data_oversampling_pipe(image, random_factor=False):
    """
    change the image for better generalization
    Add rotation
    Add translation
    :param random_factor: apply a random rotation and translation
    :param image:
    :return: transformed image
    """
    if random_factor:
        flip = 1 if random.randint(0, 1) == 1 else -1
        r = random.random() * flip
    else:
        r = 1
    # rotation
    img = rotate_image(image, 10 * r)

    # Translation
    img = translate_image(img, 3 * r, 6 * r)

    # we dont flip because the signs also flip make it harder to learn letters and numbers on signs
    # flip image horizontal
    # if random_factor and (flip == -1):
    #     img = cv2.flip(image, flipCode=0)

    return img


def data_correction_pipe(image, img_size=(32, 32), noise_ratio=0.1):
    # resize image
    resized = cv2.resize(image, img_size)
    normalized = resized / 255.
    # convert to gray
    # gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    #
    # # noisy = random_noise(gray, mode="s&p", amount=noise_ratio)
    #
    # result = np.stack((gray,) * 1, axis=-1)

    return normalized


def oversample_dataset(X_input, y_input, min_number_records=3000):
    """
    :param min_number_records: Number of samples for each subclass needed
    :param X_input: feature images
    :param y_input: labels
    :return: Updates features , labels
    """

    unique_labels = np.sort(np.unique(y_input))
    for class_label in unique_labels:
        label_subset = np.where(y_input == class_label)[0]

        missing_samples = min_number_records - len(label_subset)
        if missing_samples > 0:
            print(f"[Class {class_label}] Need additional {missing_samples} images")
            print("Create sample patch from original data")
            while True:
                print(f"Process images. {missing_samples} samples remain")
                for data in X_input[label_subset]:
                    new_data = data_oversampling_pipe(data, random_factor=True)
                    X_input = np.concatenate((X_input, [new_data]))
                    y_input = np.concatenate((y_input, [class_label]))
                    missing_samples -= 1
                    if missing_samples <= 0:
                        break
                if missing_samples <= 0:
                    break
    return X_input, y_input


def preprocess_image_data(X_input):
    """
    Process pipe (resize -> normaize [0,1] --> noise --> gray --> shift to zero centered
    :param X_input:
    :return:
    """
    # X_output = list()
    # print(f"Preprocess {len(X_input)} images")
    # for pos, data in enumerate(X_input):
    #     result = data_correction_pipe(data)
    #     X_output.append(result)
    result = np.asarray(X_input)
    return result - 0.5


# Plot number of images per class as bar chart
def plot_number_of_samples_per_class(y_input, save_file):
    unique_elements, counts_elements = np.unique(y_input, return_counts=True)
    plt.bar(np.arange(len(unique_elements)), counts_elements, align='center')
    plt.ylabel('Training samples')
    plt.xlabel('Classes')
    plt.xlim([-1, len(unique_elements)])
    plt.savefig(save_file)
    plt.show()
    plt.close()


def plot_dataset_data(X_train, y_train, filename="dataset_images.png"):
    # show a random pictore for each lable class

    unique_labels = np.sort(pd.unique(y_train))
    n_img_col = 4
    # plotting with subplots
    f, axs = plt.subplots(nrows=max([2, len(unique_labels) // n_img_col + 1]), ncols=n_img_col, figsize=(50, 50))
    col_counter = 0
    row_counter = 0
    for unique in unique_labels:
        label_subset = np.where(y_train == unique)[0]
        rand_label_array_ind = random.randint(0, len(label_subset) - 1)
        index = label_subset[rand_label_array_ind]
        # print(f"UL : {unique} Rand POs: {rand_label_array_ind} Index : {index}")
        image = X_train[index].squeeze()

        axs[row_counter, col_counter].imshow(image)
        axs[row_counter, col_counter].set_title("Label: " + str(unique) + f"\n Samples: {len(label_subset)}",
                                                fontsize=42)

        col_counter += 1
        if col_counter >= n_img_col:
            col_counter = 0
            row_counter += 1

    # fill missing images
    while col_counter < n_img_col:
        axs[row_counter, col_counter].imshow(np.zeros([3, 3, 3]))
        col_counter += 1

    plt.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig("data/metrics/"+filename)
    plt.show()
    plt.close()


def image_transformation_sample(X_input, y_input, save_file, transformed=False, gray=False, gray_scale=(0, 255),
                                signnames_file="./data/classnames.csv"):
    # load sign names
    sign_names = pd.read_csv(signnames_file).iloc[:, 1].to_dict()
    unique_labels = pd.read_csv(signnames_file).iloc[:, 0].astype(int)
    n_img_col = 10
    # plotting with subplots
    f, axs = plt.subplots(max([2,len(unique_labels) // n_img_col + 1]), n_img_col, figsize=(160, 160))
    col_counter = 0
    row_counter = 0
    for unique in unique_labels:
        print(f"Take image for {unique} - {sign_names[unique]}", end="")
        label_subset = np.where(y_input.astype(int) == unique)[0]
        if len(label_subset) > 0:
            rand_label_array_ind = random.randint(0, len(label_subset) - 1)
            index = label_subset[rand_label_array_ind]
            print(f"UL : {unique} Rand POs: {rand_label_array_ind} Index : {index}")
            image = X_input[index].squeeze()
            if transformed:
                image = data_oversampling_pipe(image, random_factor=True)
            print("")
        else:
            print(" > No data for")
            image = np.zeros([3, 3, 3])
        if gray:
            axs[row_counter, col_counter].imshow(image, cmap='gray', vmin=gray_scale[0], vmax=gray_scale[1])
        else:
            axs[row_counter, col_counter].imshow(image)
        axs[row_counter, col_counter].set_title(f"Label {unique}\n{sign_names[unique]}\n Samples:{len(label_subset)}",
                                                fontsize=50)

        col_counter += 1
        if col_counter >= n_img_col:
            col_counter = 0
            row_counter += 1

    # fill missing images
    while col_counter < n_img_col:
        axs[row_counter, col_counter].imshow(np.zeros([3, 3, 3]))
        col_counter += 1

    plt.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(save_file)
    plt.show()
