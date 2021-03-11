''' Supports binary classification '''

# import libraries
import time
import cv2
import argparse
import os
import configparser
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import owncloud
import pandas as pd
import csv
import gc
import numpy as np

# import from local files
from misc_functions import str2bool
from img_loading import *
from owncloud_ops import get_latest, redundant_put
from train_ops import evaluate_classification, testFold
from keras_model_lib import keras_models, lock_weights

'''
Explanation of arguments:
-f: the local config ini that stores the locations of the training data, saved models, and connection details
-w: maximum time this whole script will run (in hours)
-p: time the script will wait in between checking for available files
-ev: evaluate volume with output histogram of the pixel values for all three input channels. Used to visualize the differences in domains.
-ID: institution-specific ID to save files in the case where one institution would like to split their data into two 'virtual silos'
'''

# acquire input variables from the user
parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('-f', '--file', type=str, default="client_DR.ini")
parser.add_argument('-w', '--walltime_h', type=int, default=24)
parser.add_argument('-p', '--polltime_s', type=int, default=15)
parser.add_argument('-ev', '--evaluate_volume', type=str, default="False")
parser.add_argument('-ID', '--institution_ID', type=str, default="SFU")

args = parser.parse_args()

# convert from string to boolean for whether or not to evaluate the volume and generate histogram
# default is set to False
evaluate_volume_bool = str2bool(args.evaluate_volume)

# these lines are required for the code to run successfully on TF 2.1
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# read the config file
base_config = configparser.ConfigParser()
base_config.read(args.file)
model_dir = os.path.normpath(base_config['training']['model_dir'])
base_dir = os.path.normpath(base_config['training']['data_dir'])
weight_config = base_config['training']['weights']
excel_file = base_config['training']['excel_file']
model_type = base_config['training']['model_type']
num_folds = int(base_config['training']['num_folds'])
num_layers = int(base_config['training']['num_layers'])
BATCH_SIZE = int(base_config['training']['batch_size'])

# convert the integer 'image_size' into a square input shape - may need to change this if a non-square image is prefered for the specific use case
image_size = int(base_config['training']['image_size'])
INPUT_SHAPE = (image_size, image_size, 3)

# convert the string for classes into a list 
class_list = base_config['training']['classes']

# split the classes separated by a semicolon (;)
class_list = class_list.split(";")

# convert the original stratification (to have a balanced distribution throughout all further stratified sub-classes)
original_stratification = base_config['training']['original_stratification']
original_stratification = original_stratification.split(";")

# where Normal;Mild_Mod;NprolifDR;Severe;PDR would yield:
# class_definitions[0] = Normal;Mild
# class_definitions[1] = Mod;NprolifDR;Severe;PDR
class_definitions = base_config['training']['class_definitions']
class_definitions = class_definitions.split("_") 

# obtain the image definitions (the specific use-case is for SVC, DVC OCT-A and a MIP of the SVC and DVC Structural)
image_breakdown = base_config['training']['image_breakdown']
image_breakdown = image_breakdown.split(";")

# (Defined in .ini) All classes are augmented to this factor multiplied by the most prevalent class (balanced to highest class)
AUG_NUM = base_config['training']['aug_factor']

csv_dir = os.path.join(model_dir, 'logs')

c_server = owncloud.Client.from_public_link(base_config['cloud']['link'], folder_password=base_config['cloud']['pw'])

# change this comparison to your use case - to check and ensure that the specified ID is proper for consistent loading from the same file names
institution_ID = args.institution_ID
if institution_ID != 'SFU' and institution_ID != 'OHSU' and institution_ID != 'UW' and institution_ID != 'CUHK':
    raise Exception("Please specify institution ID: OHSU, UW, or SFU with flag -ID")

loc_stamp = institution_ID

if __name__ == '__main__':

    os.makedirs(csv_dir, exist_ok=True)

    start_time = time.time()
    current_fmod_h5 = 0
    current_fmod_ini = 0

    # load images from excel file
    if os.path.exists(excel_file) == True:
        data = pd.read_excel(excel_file, usecols=['Label', 'SVC_angio', 'DVC_angio', 'SVC_struct', 'DVC_struct']) 
        df = pd.DataFrame(data)
    else:
        # empty dataframe.. double check the variable excel_file in your client configuration file
        df = pd.DataFrame({'A' : []})

    # importing and saving images
    img, lbl = import_and_save(base_dir, INPUT_SHAPE, class_list, df, original_stratification, image_breakdown, evaluate_volume_bool, loc_stamp)

    # shuffle and split the data in a stratified way
    kfold_indices = stratified_shuffle(labels=lbl, n_splits=num_folds, random_state=3)

    # after the stratified shuffle, we need to convert the labels back to the specified classification problem defined class_definitions
    conversion_arr = []

    # convert from the list of stratifications to create a list of classes according to the class split
    for curr_class in original_stratification:
        for idx, s in enumerate(class_definitions):
            if curr_class in s:
                conversion_arr.append(idx)
    # loop through the array and convert according to the array
    for original_index, conversion_index in enumerate(conversion_arr):
        # conversion index if it is the original index, or else, we don't replace
        lbl = [conversion_index if x==original_index else x for x in lbl]

    # save the folds for consistency of future trials as well as reserve a proper test fold
    save_folds(img, lbl, num_folds, base_dir, image_size, original_stratification, kfold_indices, INPUT_SHAPE, AUG_NUM, loc_stamp)
    save_folds_stratified(img, lbl, num_folds, base_dir, image_size, original_stratification, kfold_indices, INPUT_SHAPE, AUG_NUM, loc_stamp)
    
    prev_outer = None
    prev_inner = None

    # Main train loop
    while time.time() - start_time < args.walltime_h * 3600:
        # clearing memory from multiple model creation: https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()
        print('Cleared backend tensorflow session')

        #     # Grabs the latest train_config ini file
        current_fmod_ini, ini_path = get_latest(c_server, '.ini', current_fmod_ini, model_dir)

        # train_config is the ini file entered in the args
        train_config = configparser.ConfigParser()
        train_config.read(ini_path)
        CURRENT_EPOCH = int(train_config['params']['current_epoch'])
        LEARN_RATE = float(train_config['params']['learn_rate'])
        scratch_flag = int(train_config['env']['scratch'])

        ### also read the current outer and inner fold 
        outer_fold = int(train_config['params']['outer_fold'])
        inner_fold = int(train_config['params']['inner_fold'])

        # until it is the end and test_epoch is not 0
        # filestamp is the universal identifier for each neural net,
        # this is usually a datetime conversion, but can be changed to the preference
        filestamp = train_config['env']['name']
        # naming convention to record the outer and inner folds when saving each client model and csv file
        model_save_path = os.path.join(model_dir, filestamp + '_OF_'+ str(outer_fold) + '_IF_' + str(inner_fold) + '_client.h5')
        csv_save_path = os.path.join(csv_dir, filestamp + '_OF_'+ str(outer_fold) + '_IF_' + str(inner_fold) + '_log.csv')

        # if it is the same fold as previous epoch
        if prev_outer == outer_fold and prev_inner == inner_fold:
            prev_outer = outer_fold
            prev_inner = inner_fold
        else:
        # otherwise, need to re-load the fold-specific data
            print('loading new fold..')
            img_val, lbl_val, img_train, lbl_train = load_data_from_folds(base_dir, outer_fold, inner_fold, image_size, INPUT_SHAPE, num_folds, loc_stamp, 'train_val')
            prev_outer = outer_fold
            prev_inner = inner_fold

        # Prepare for categorical or binary classification (define loss function, etc..)
        if (int(len(class_list))<=2): 
            class_num = 1
            loss_fn = 'binary_crossentropy'
        else:
            class_num = len(class_list)
            loss_fn = 'categorical_crossentropy'

        model_path_arr = []
        model_loss_arr = []
        
        poll_start_time = time.time()

        if scratch_flag or CURRENT_EPOCH < 1:
            print('training first epoch from scratch for outer fold:', outer_fold, 'and inner fold:', inner_fold)
            ### Load model - if model architecture needs to be changed, do so at keras_model_lib.py  ###
            model = keras_models(INPUT_SHAPE, class_num, weight_config, num_layers, model_type)
        else:
            current_fmod_h5, h5_path = get_latest(c_server, '_agg.h5', current_fmod_h5, model_dir)
            # Finalize and Compile model
            model = load_model(h5_path)
            model = lock_weights(model, num_layers)

        # Defining callbacks used in training
        csv_logger = CSVLogger(filename=csv_save_path, append=True)

        optim = optimizers.Adam(lr=LEARN_RATE)

        # Because of aggregation, we have to re-compile the model
        model.compile(optimizer=optim, loss=loss_fn,
                        metrics=['acc'])

        # Randomly augment the balanced training set - define the augmentations
        datagen = ImageDataGenerator(
                vertical_flip=True,
                rotation_range=10,
                horizontal_flip=True,
                width_shift_range=0.05,
                height_shift_range=0.05)

        # no augmentations for the validation set
        val_datagen = ImageDataGenerator()

        # seed for repeatibility
        train_generator = datagen.flow(img_train, y=lbl_train, batch_size=BATCH_SIZE, shuffle=True, seed=13)
        validation_generator = val_datagen.flow(img_val, y=lbl_val, batch_size=BATCH_SIZE, shuffle=True, seed=13)
           
        # try/except block due to OOM issues
        try:
            model.fit(
                train_generator,
                epochs=CURRENT_EPOCH + 1,
                initial_epoch=CURRENT_EPOCH,
                callbacks=[csv_logger],
                validation_data=validation_generator,
                shuffle=True,
            )

            # save upload model and csv file to the server
            model.save(model_save_path)
            put_result = redundant_put(c_server, '/', model_save_path)
            put_result = redundant_put(c_server, '/', csv_save_path)
            current_time = time.time() - start_time
            print("Done! Time elapsed: %f seconds." % current_time)

            # clear and garbage collect unused variables
            model = None
            datagen = None
            val_datagen = None
            train_generator = None
            validation_generator = None
            gc.collect()
            continue
        
        # upon error, try again after garbage collecting
        except tf.errors.ResourceExhaustedError as e:
            print(e)
            model = None
            train_generator = None
            validation_generator = None
            datagen = None
            val_datagen = None
            prev_outer = None
            prev_inner = None
            img_val = None
            lbl_val = None 
            img_train  = None 
            lbl_train = None
            gc.collect()
            print('Error.. restarting epoch and reloading data')
            continue
        


