import os
import cv2
import time
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def check_and_import(img_dir, img_name, lbl_dir, lbl_name):
    # Imports and normalizes images from two directories
    # img_path handles the images, lbl_path handles the labels
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, lbl_name)

    # image and label filenames should be identical
    assert os.path.splitext(img_name)[0] == os.path.splitext(lbl_name)[0]

    try:
        img = np.asarray(cv2.imread(img_path, 0))
        lbl = np.asarray(cv2.imread(lbl_path, 0))

        assert img.shape == lbl.shape, "images and labels should have the same shape."

        s_max = np.amax(lbl)
        lbl = np.where(lbl > s_max/2, 1, 0)     # binarize labels if needed
    except:
        pass
    else:
        return img, lbl


def import_and_save(data_dir, mode):
    """
    the file structure of the data should be as follows:
    - train (naming optional)
        - images
            - ONLY images
        - labels
            - ONLY binarized labels (with same filenames, minus extensions)
    - test (naming optional)
        - same as above
    - validation (naming optional)
        - same as above
    """

    img_npy_path = os.path.join(data_dir, mode + "_data.npy")
    lbl_npy_path = os.path.join(data_dir, mode + "_labels.npy")

    if os.path.exists(img_npy_path) and os.path.exists(lbl_npy_path):
        print(mode + ': NPY files found - importing...')
        img_load = np.load(img_npy_path)
        lbl_load = np.load(lbl_npy_path)
    else:
        print(mode + ': NPY files not found - loading images...')

        # Imports all images and labels from the training data dir
        # and saves them into npy files, while also returning them to main
        print('Getting file names...')
        load_start_time = time.time()
        img_dir = os.path.join(data_dir, mode, 'images')
        lbl_dir = os.path.join(data_dir, mode, 'labels')

        # Getting the total number of images to initialize arrays
        img_names = os.listdir(img_dir)
        lbl_names = os.listdir(lbl_dir)

        assert len(img_names) == len(lbl_names), "number of elements in the image and label folders should be identical"
        arr_size = len(img_names)

        # Each image should be the same size
        temp_img = cv2.imread(os.path.join(img_dir, img_names[0]))
        height = temp_img.shape[0]
        width = temp_img.shape[1]

        # allocating the image and label arrays to save load time
        img_load = np.zeros([arr_size, height, width, 1], dtype=np.float32)
        lbl_load = np.zeros([arr_size, height, width], dtype=np.float32)

        print(img_load.shape)

        idx = 0
        for i in range(len(img_names)):

            # Importing the images and normalizing them
            img_in, lbl_in = check_and_import(img_dir, img_names[i], lbl_dir, lbl_names[i])
            if img_in is not None:
                img_load[idx, :, :, 0] = img_in
                lbl_load[idx, :, :] = lbl_in
                idx += 1

            current_time = time.time() - load_start_time
            minutes = current_time / 60
            hours, minutes = divmod(minutes, 60)
            print('Progress: ' + str(i) + ' out of ' + str(len(img_names)) + ' completed. Time: %dh %dm (%ds)' % (
                hours, minutes, current_time), end='\r')

        # saving the npy files for faster load times later
        img_load = img_load / np.amax(img_load)
        np.save(img_npy_path, img_load)
        np.save(lbl_npy_path, lbl_load)

    return img_load/np.amax(img_load), lbl_load


def normalize(matrix):
    a = np.amin(matrix)
    b = np.ptp(matrix)
    matrix[matrix < 0] = 0
    temp_mat = matrix / np.amax(matrix)
    return temp_mat


def augment(images, labels, n_augs):
    aug_start_time = time.time()

    # Define our augmentation pipeline
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.1]),  # drop 5% or 10% of all pixels
        iaa.LinearContrast((0.5, 3)),
        iaa.GammaContrast((0.4, 1.75)),
        iaa.Affine(rotate=(-15, 15)),  # rotate by -15 to 15 degrees (affects segmaps)
        iaa.Fliplr()
    ], random_order=True)

    height = images.shape[1]
    width = images.shape[2]

    # initialize arrays to save time
    img_aug = np.zeros([images.shape[0] * (n_augs + 1), height, width, 1], dtype=np.float32)
    lbl_aug = np.zeros([labels.shape[0] * (n_augs + 1), height, width], dtype=np.float32)
    aidx = 0

    print('Augmenting ' + str(n_augs) + ' times per image (' + str(len(img_aug)) + ' total)')

    for aa in range(images.shape[0]):
        img_temp = np.squeeze(images[aa, :, :])
        lbl_temp = np.squeeze(labels[aa, :, :])
        lbl_temp = lbl_temp.astype(np.int8)

        # defining the segmap object so augmentations are identical
        segmap = SegmentationMapsOnImage(lbl_temp, shape=img_temp.shape)

        img_aug[aidx, :, :, 0] = img_temp
        lbl_aug[aidx, :, :] = lbl_temp
        aidx += 1

        # Applying the augmentations
        for aaa in range(n_augs):
            img_aug_i, lbl_aug_i = seq(image=img_temp, segmentation_maps=segmap)
            lbl_aug_i = lbl_aug_i.get_arr()

            img_aug_i = normalize(img_aug_i)

            img_aug[aidx, :, :, 0] = img_aug_i
            lbl_aug[aidx, :, :] = lbl_aug_i
            aidx += 1

    print('\n')
    return img_aug, lbl_aug
