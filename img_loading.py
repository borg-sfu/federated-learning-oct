''' functions used for loading and saving images into .npy files in addition to performing upsampling of the lower prevalent class through image augmentation '''

# import libraries
import os
import cv2
import time
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imageio
import pandas as pd
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
import copy


def save_folds(img, lbl, num_folds, save_outputs, image_size, class_list, kfold_indices, INPUT_SHAPE, AUG_NUM, loc_stamp):
    ''' 
    merging all classes of the same fold... saving both image and label data for each fold as .npy files
    if processed and saved already, this function will not do anything
    '''
    # first, convert to numpy arrays
    img = np.array(img)
    lbl = np.array(lbl)

    # loop through the number of folds and save .npy files for the different folds
    for fold in range(num_folds):
        temp_arr_idx = []
        # fold specific data
        img_npy_path_aug = os.path.join(save_outputs, str(image_size) + "_aug_fold_" + str(fold) + "_" + loc_stamp + "_data.npy")
        lbl_npy_path_aug = os.path.join(save_outputs, str(image_size) + "_aug_fold_" + str(fold) + "_" + loc_stamp + "_labels.npy")
        img_npy_path = os.path.join(save_outputs, str(image_size) + "_fold_" + str(fold) + "_" + loc_stamp + "_data.npy")
        lbl_npy_path = os.path.join(save_outputs, str(image_size) + "_fold_" + str(fold) + "_" + loc_stamp + "_labels.npy")

        # if the .npy files exist, ignore, else, for each fold, combine all classes together and index the img and lbl data and save
        if os.path.exists(img_npy_path_aug) and os.path.exists(lbl_npy_path_aug) and os.path.exists(img_npy_path) and os.path.exists(lbl_npy_path):
            print('NPY files found - moving on to the next fold or training')
        else:
            for curr_class in range(len(class_list)):
                temp_arr_idx = np.concatenate((temp_arr_idx, kfold_indices[curr_class][fold]))

            temp_arr_idx = np.array(temp_arr_idx.astype(int))

            # index the data and lbl
            img_fold = img[temp_arr_idx]
            lbl_fold = lbl[temp_arr_idx]
            print(lbl[temp_arr_idx])
            print(temp_arr_idx)

            # saving as npy files for EACH fold
            print('Saving...')
            np.save(img_npy_path, img_fold)
            np.save(lbl_npy_path, lbl_fold)

            # augment and save
            img_aug, lbl_aug = augment(img_fold, lbl_fold, AUG_NUM, save_outputs, INPUT_SHAPE)
            print('Saving augmented images...')
            np.save(img_npy_path_aug, img_aug)
            np.save(lbl_npy_path_aug, lbl_aug)


def save_folds_stratified(img, lbl, num_folds, save_outputs, image_size, class_list, kfold_indices, INPUT_SHAPE, AUG_NUM, loc_stamp):
    ''' 
    merging all classes of the same fold... saving both image and label data for each fold as .npy files
    if processed and saved already, this function will not do anything
    '''
    # first, convert to numpy arrays
    img = np.array(img)
    lbl = np.array(lbl)

    # loop through the number of folds and save .npy files for the different folds
    for fold in range(num_folds):
        temp_arr_idx = []
        # fold specific data
        img_npy_path = os.path.join(save_outputs, str(image_size) + "_fold_stratified_" + str(fold) + "_" + loc_stamp + "_data.npy")
        lbl_npy_path = os.path.join(save_outputs, str(image_size) + "_fold_stratified_" + str(fold) + "_" + loc_stamp + "_labels.npy")

        # if the .npy files exist, ignore, else, for each fold, combine all classes together and index the img and lbl data and save
        if os.path.exists(img_npy_path) and os.path.exists(lbl_npy_path):
            print('NPY files found - moving on to the next fold or training')
        else:
            for curr_class in range(len(class_list)):
                temp_arr_idx = np.concatenate((temp_arr_idx, kfold_indices[curr_class][fold]))

            temp_arr_idx = np.array(temp_arr_idx.astype(int))

            # index the data and lbl
            img_fold = img[temp_arr_idx]
            lbl_fold = lbl[temp_arr_idx]
            print(lbl[temp_arr_idx])
            print(temp_arr_idx)

            # saving as npy files for EACH fold
            print('Saving...')
            np.save(img_npy_path, img_fold)
            np.save(lbl_npy_path, lbl_fold)


def load_data_from_folds(base_dir, outer_fold, inner_fold, image_size, INPUT_SHAPE, num_folds, loc_stamp, purpose):
    '''
    call load data from folds function. 

    Inputs:
        - base_dir (path) directory containing .npy files
        - outer_fold (int) the current outer fold in nested cross validation
        - inner_fold (int) the current inner fold in nested cross validation
        - image_size (int) image size specified in config file
        - INPUT_SHAPE (tuple) shape of image used for augmentation
        - num_folds (int) the number of folds used for nested cross validation as defined in the client configuration file
    Outputs:
        - training img and labels (augmented to balance, normalized)
        - validation img and labels (unaugmented, normalized)
        - testing img and labels (unaugmented, normalized)
        - stratified test set (unaugmented, normalized)
    '''
    if purpose == 'test':
    # test data (no augmentation)
        print('TESTING FROM FOLD: '+str((outer_fold)%num_folds))
        img_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold)%num_folds) + "_" + loc_stamp + "_data.npy")
        lbl_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold)%num_folds) + "_" + loc_stamp + "_labels.npy")

        # load from saved path and normalize (original data)
        img_test = np.load(img_npy_path)
        lbl_test = np.load(lbl_npy_path)
        img_test = normalize(img_test)

        # load from saved path and normalize (further stratified data)
        img_npy_path = os.path.join(base_dir, str(image_size) + "_fold_stratified_" + str((outer_fold)%num_folds) + "_" + loc_stamp + "_data.npy")
        lbl_npy_path = os.path.join(base_dir, str(image_size) + "_fold_stratified_" + str((outer_fold)%num_folds) + "_" + loc_stamp + "_labels.npy")
        img_strat_test = np.load(img_npy_path)
        lbl_strat_test = np.load(lbl_npy_path)
        img_strat_test = normalize(img_strat_test)
        
        print('VALIDATION FROM FOLD: '+str((outer_fold+inner_fold+1)%num_folds))
        # val data (no augmentation -> keep the same ratio as what is seen in the test set)
        img_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold+inner_fold+1)%num_folds) + "_" + loc_stamp + "_data.npy")
        lbl_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold+inner_fold+1)%num_folds) + "_" + loc_stamp + "_labels.npy")
        img_val = np.load(img_npy_path)
        lbl_val = np.load(lbl_npy_path)
        lbl_val = normalize(lbl_val)

        return img_test, lbl_test, img_strat_test, lbl_strat_test, img_val, lbl_val

    elif purpose == 'train_val':
        lbl_train = None
        i = inner_fold
        # this loop goes through numfolds-1
        for ii in range(num_folds-1):
            # loads the fold as validation if ii equals to i (in short, the inner fold is set to the validation)
            if ii == inner_fold:
                print('VALIDATION FROM FOLD: '+str((outer_fold+ii+1)%num_folds))
                # val data (no augmentation -> keep the same ratio as what is seen in the test set)
                img_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold+ii+1)%num_folds) + "_" + loc_stamp + "_data.npy")
                lbl_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold+ii+1)%num_folds) + "_" + loc_stamp + "_labels.npy")
                img_val = np.load(img_npy_path)
                lbl_val = np.load(lbl_npy_path)
            # otherwise, the remaining folds will be used for training
            else:
                print('TRAINING FROM FOLD: '+str((outer_fold+ii+1)%num_folds))
                img_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold+ii+1)%num_folds) + "_" + loc_stamp + "_data.npy")
                lbl_npy_path = os.path.join(base_dir, str(image_size) + "_fold_" + str((outer_fold+ii+1)%num_folds) + "_" + loc_stamp + "_labels.npy")
                # if list is empty
                if lbl_train is None:
                    img_train = np.load(img_npy_path)
                    lbl_train = np.load(lbl_npy_path)
                # if list is not empty, concatenate the loaded
                else:
                    img_train = np.concatenate((img_train, np.load(img_npy_path)))
                    lbl_train = np.concatenate((lbl_train, np.load(lbl_npy_path)))

        # augment to upsample the lower prevalent classes in the training data
        img_train, lbl_train = augment(img_train, lbl_train, 1, base_dir, INPUT_SHAPE)

        # normalize all once again
        img_train = normalize(img_train)
        img_val = normalize(img_val)

        # print data shape for verification
        print('Train shape (upsampled through augmentation):')
        print(img_train.shape)
        print(lbl_train.shape)

        print('Validation shape (no augmentation):')
        print(img_val.shape)
        print(lbl_val.shape)

        return img_val, lbl_val, img_train, lbl_train
    else:
        # should always be either 'test' or 'train_val'
        raise Exception("invalid string for 'purpose' in load_data_from_folds within img_loading.py")


def evaluate_stratified_volume(labels, images, image_breakdown, original_stratification, save_dir, verbose=0):
    ''' 
    Plots the data distribution
    Inputs:
        - label (np.array volume)
        - images (np.array volume)
        - image_breakdown (list) required for labelling/naming of the plots
        - original_stratification (list) required for labelling/naming of the plots
        - save_dir (path) path to save the histograms
    '''

    # use label to create a balanced and shuffled split
    unique_set = list(set(labels))

    # initialize lists
    ioc_arr = []
    distribution_of_classes = []
    class_idx_in_list = []

    # loop through each unique value and find the indicies 
    for i, unique_val in enumerate(unique_set):

        # ioc are the indices of occurences
        ioc = np.where(labels == unique_val)[0]
        distribution_of_classes.append(len(ioc))
        ioc_arr.append(ioc)

    # for each unique class, generate an intensity plot
    for i, curr_idx_arr in enumerate(ioc_arr):
        img_temp = copy.deepcopy(images)

        if verbose == 1: print('For severity:', unique_set[i], original_stratification[i])
        if verbose == 1: print('Count: ', str(len(curr_idx_arr)))
        if verbose == 1: print('image shape:', img_temp.shape)

        img_temp = np.squeeze(img_temp[curr_idx_arr])

        # https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
        colour = ('r','g','b')

        # generates the histogram
        img_temp = normalize(img_temp)*256
        img_temp = np.array(img_temp).astype(np.uint8)
        for channel_breakdown_idx, curr_channel in enumerate(image_breakdown):
            plt.hist(img_temp[:,:,:,channel_breakdown_idx].ravel(),256,[0,256], color=colour[channel_breakdown_idx], label=curr_channel, log=False, histtype='step')
            plt.xlim([0,256])

        # define title and axes
        plt.title('Histogram for ' + original_stratification[i] + ' (n='+str(len(curr_idx_arr))+')')
        plt.xlabel('Intensity - normalized [0, 256]')
        plt.ylabel('Counts (log)')
        plt.legend(loc="upper right")

        # save histogram
        plt.savefig(os.path.join(save_dir,'Intensity_hist_'+original_stratification[i]+'.png'))
        plt.close('all')


def stratified_shuffle(labels, n_splits, random_state=7):
    '''
    Stratify shuffle the labels. Similar to the one found: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    Main difference is to provide flexibility to define validation split as well

    Returns the indices of the folds as a list of length (n_splits)

    Input:
        - labels (numpy array) the label inputs
        - n_splits (int) number of folds to split the data into
        - random_state (int) randomize the shuffle and specify for repeatability

    Output:
        - class_idx_in_list (list) class x fold shape list with structure [class][fold] containing the indices for each class & fold combination
    '''

    # use label to create a balanced and shuffled split
    unique_set = list(set(labels))

    # initialize lists
    ioc_arr = []
    distribution_of_classes = []
    class_idx_in_list = []

    # loop through each unique value and find the indicies 
    for i, unique_val in enumerate(unique_set):

        # ioc are the indices of occurences
        ioc = np.where(labels == unique_val)[0]
        distribution_of_classes.append(len(ioc))
        ioc_arr.append(ioc)

    # for each unique class, shuffle and split into five folds
    for i, curr_idx_arr in enumerate(ioc_arr):
        fold_idx_in_class = []

        # shuffle with random_state defined to have it be reproducible
        curr_idx_arr = shuffle(curr_idx_arr, random_state=random_state)

        # the number of elements in each fold
        fold_num_elements = curr_idx_arr.size // n_splits

        # loop through the number of folds and split the data accordingly
        for split_ctr, split_iter in enumerate(range(n_splits)):
            # distribute into a list the current split of data for the specific fold
            fold_idx_in_class.append(curr_idx_arr[split_iter::n_splits])

        # combine each class that each been split into k-folds
        class_idx_in_list.append(fold_idx_in_class)
    return class_idx_in_list


def process_MIP(stack, axis=2):
    ''' Takes any volume and returns the max value for each - performs maximum intensity projection (MIP)'''
    return np.max(stack, axis=axis)


def load_and_process(SVC_angio_curr, DVC_angio_curr, SVC_struct_curr, DVC_struct_curr, INPUT_SHAPE, label_class):
    ''' loads images, merges, processes the MIP, combines the image into SVC-angio, DVC-angio, and MIP of SVC/DVC-struct
    
    Inputs:
        - SVC_angio_curr (path) - path to the SVC OCT-A image
        - DVC_angio_curr (path) - path to the DVC OCT-A image
        - SVC_struct_curr (path) - path to the SVC Structural image
        - DVC_struct_curr (path) - path to the DVC Structural image

    Outputs:
        - image_data (np.array) - numpy array containing the 3 channel image (SVC OCT-A, DVC OCT-A, MIP of SVC and DVC Structural enface)
    '''

    # load the images
    SVC_angio_data = imageio.imread(SVC_angio_curr)
    DVC_angio_data = imageio.imread(DVC_angio_curr)
    SVC_struct_data = imageio.imread(SVC_struct_curr)
    DVC_struct_data = imageio.imread(DVC_struct_curr)

    if len(SVC_angio_data.shape) > 2:
        # only one channel since this is not channel-specific data
        SVC_angio_data = SVC_angio_data[:,:,0]
        DVC_angio_data = DVC_angio_data[:,:,0]
        SVC_struct_data = SVC_struct_data[:,:,0]
        DVC_struct_data = DVC_struct_data[:,:,0]

    # merge to process the MIP of these images
    MIP = cv2.merge((SVC_struct_data, DVC_struct_data))

    # process and resize MIP of structural images to same as angios
    MIP_struct_data = process_MIP(MIP, 2)
    MIP_struct_data = cv2.resize(MIP_struct_data, dsize=(INPUT_SHAPE[0], INPUT_SHAPE[1]), interpolation=cv2.INTER_CUBIC)

    # resize the image to the input shape (essentially interpolating the subsampled slow axis back to the original dimensions)
    SVC_angio_data = cv2.resize(SVC_angio_data, dsize=(INPUT_SHAPE[0], INPUT_SHAPE[1]), interpolation=cv2.INTER_CUBIC)
    DVC_angio_data = cv2.resize(DVC_angio_data, dsize=(INPUT_SHAPE[0], INPUT_SHAPE[1]), interpolation=cv2.INTER_CUBIC)

    # subsample in the slow axis (channel-wise normalization [0, 1])
    image_data = cv2.merge((normalize(SVC_angio_data), normalize(DVC_angio_data), normalize(MIP_struct_data)))

    # normalize the image and check the dimensions
    if image_data.ndim == 2:
        raise Exception("The input needs to be 3-dimensional")

    return image_data


def import_and_save(data_dir, INPUT_SHAPE, class_list, df, original_stratification, image_breakdown, evaluate_volume_bool, loc_stamp):
    ''' This function imports the data from an excel spreadsheet and saves as .npy for easy loading
    Inputs:
        - data_dir (path) points to the directory to save the .npy files
        - INPUT_SHAPE (3D tuple) shape that you would like the image to be resized to
        - class_list (list) list of potential classes
        - original_stratification (str list) obtained from the client config file defining the split of classes
        - image_breakdown (str list) obtained from the client config file defining the breakdown of images (only used for the evaluation and creation of histograms)
        - evaluate_volume_bool (bool) user-specified input boolean for whether the data should be evaluated and histograms generated to visualize domain differences
        - loc_stamp (str) institution ID

    '''

    '''
    import from excel file - the format of the excel file should be as follows (an example of our use case):
    - Label (first column)
        - column of label (eg. dog or cat)
    - SVC_angio (second column)
        - column of path to the corresponding SVC OCT-A
    - DVC_angio (third column)
        - column of path to the corresponding DVC OCT-A
    - SVC_struct (fourth column)
        - column of path to the corresponding SVC structural OCT
    - DVC_struct (fifth column)
        - column of path to the corresponding DVC structural OCT
    '''

    # define variables
    lbl_load = []
    img_load = []
    image_size = INPUT_SHAPE[0]

    # define the image and label save paths 
    img_npy_path = os.path.join(data_dir, str(image_size) + "_" + loc_stamp + "_data.npy")
    lbl_npy_path = os.path.join(data_dir, str(image_size) + "_" + loc_stamp + "_labels.npy")

    # if the .npy file exists, just load
    if os.path.exists(img_npy_path) and os.path.exists(lbl_npy_path):
        print('NPY files found - importing...')
        temp_load = np.load(img_npy_path)
        lbl_load = np.load(lbl_npy_path)

        # resize images to have consistency across institutions
        for ii in tqdm(range(len(temp_load))):
            img_load.append(cv2.resize(temp_load[ii,:,:,:], dsize=(INPUT_SHAPE[0], INPUT_SHAPE[1]), interpolation=cv2.INTER_CUBIC))

        # convert to numpy arrays
        img_load = np.asarray(img_load)
        lbl_load = np.asarray(lbl_load)

    # if the .npy file does not exist, we load either from excel file or using identifiers in the file name
    else:
        print('NPY files not found - loading images...')

        # Imports all images and labels from the training data dir and saves them into npy files, while also returning them to main
        load_start_time = time.time()

        # Raise exception if excel file/dataframe does not exist
        if df.empty:
            raise Exception("Dataframe is empty - please check that the excel path exists (variable in config file labeled: excel_file)")
        else:
            # Loaded columns are 'Label', 'Split', and various paths
            print('Loading from excel file...')
            
            # Indicies contain those in the specific split
            lbl_data = df["Label"]
            SVC_angio = df["SVC_angio"]
            DVC_angio = df["DVC_angio"]
            SVC_struct = df["SVC_struct"]
            DVC_struct = df["DVC_struct"]

            # Turn string encoded classes into integers
            for idx, lbl_data_curr in enumerate(tqdm(lbl_data)):
                current_label = [original_stratification.index(ii) for ii in original_stratification if str(lbl_data_curr) in ii]
                lbl_load.append(current_label[0])
                # Importing the images and normalizing them/resizing them through a function
                image_data = load_and_process(SVC_angio[idx], DVC_angio[idx], SVC_struct[idx], DVC_struct[idx], INPUT_SHAPE, current_label[0])
                img_load.append(image_data)
                   
        # Convert to numpy array
        lbl_load = np.array(lbl_load)
        img_load = np.array(img_load)

        # Saving as npy files for faster load times later
        # raise Exception("ERROR: testing script is saving data")
        print('Saving...')
        np.save(img_npy_path, img_load)
        np.save(lbl_npy_path, lbl_load)
        
    # if true, we want to evaluate the volume and generate histograms
    if evaluate_volume_bool: evaluate_stratified_volume(lbl_load, img_load, image_breakdown, original_stratification, data_dir, verbose=1)

    return img_load, lbl_load


def normalize(matrix):
    '''
    Normalize the matrix.. it will subtract by the minimum value and normalize to the max
    Normalizes to [0, 1]
    '''
    a = np.amin(matrix)
    b = np.ptp(matrix)
    matrix = matrix - a
    temp_mat = matrix / np.amax(matrix)
    return temp_mat


def augment(images, labels, n_augs, data_dir, INPUT_SHAPE):
    '''
    This function will augment the images - change 'seq' below to adjust what parameters to augment. It has the ability to upsample the lower prevalent classes to match the highest prevalent class
    Input:
        - images (numpy array)
        - labels (numpy array)
        - n_augs (int) the augmentation factor for the most prevalent class where all other classes are augmented to balance the dataset
        - data_dir (path) path to the folder to save processed data as .npy files
        - INPUT_SHAPE (tuple) eg. (512, 512, 3)
    Output: 
        - img_aug (numpy array) augmented images
        - lbl_aug (numpy array) respective labels
    '''
    img_aug = []
    lbl_aug = []
    aug_start_time = time.time()

    # Define our augmentation pipeline
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.1]),  # drop 5% or 10% of all pixels
        iaa.LinearContrast((0.9, 1.1)),
        iaa.Affine(rotate=(-10, 10)),  # Minimum rotations expected
        iaa.Fliplr(),
        iaa.Flipud()
    ], random_order=True)

    height = images.shape[1]
    width = images.shape[2]
    
    # Get distribution of dataset
    # First, get the unique values in the label
    unique_set = list(set(labels))
    ioc_arr = []
    distribution_of_classes = []
    # Loop through each unique value and find the indicies 
    for i, unique_val in enumerate(unique_set):
        # ioc are the indices of occurences
        ioc = np.where(labels == unique_val)[0]
        distribution_of_classes.append(len(ioc))
        ioc_arr.append(ioc)

    # Most prevalent class
    pv_class = max(distribution_of_classes)
    print('Augmenting all classes to contain', str(n_augs), '*', str(pv_class), 'images')

    final_num_of_images = int(n_augs)*int(pv_class)
    required_augmentations = -1 * np.array(distribution_of_classes) + final_num_of_images

    # Loop through each unique class
    for i, curr_unique_class in enumerate(unique_set):
        # Cycle through the occuring indices of this class
        itered_ctr = 0
        for aa in itertools.cycle(ioc_arr[i]):
            if itered_ctr == required_augmentations[i]:
                # If it has augmented the required amount, we break the for loop
                print('Class', str(curr_unique_class), 'augmented by', str(required_augmentations[i]), 'to contain', str(final_num_of_images), 'images')
                break
            img_temp = np.squeeze(images[aa, :, :, :])
            lbl_temp = labels[aa]
            img_temp = img_temp*255
            img_temp = img_temp.astype(np.uint8)
            img_aug_i = seq(image=img_temp)
            img_aug_i = img_aug_i.astype(np.float64)
            img_aug.append(img_aug_i)
            lbl_aug.append(lbl_temp)

            # Increment the counter
            itered_ctr = itered_ctr + 1
        
    # Normalize after we have the required number of augmentations
    img_aug = normalize(img_aug)
    img_aug = np.array(img_aug)
    lbl_aug = np.array(lbl_aug)

    # Stack with original images (maintains the right order)
    img_aug = np.concatenate((images, img_aug), axis=0)
    lbl_aug = np.concatenate((labels, lbl_aug), axis=0)

    return img_aug, lbl_aug
