# Residual UNET For segmenting the vessels in OCT-A enface images

# Import Stuff
import time
import cv2
import argparse
import os
import configparser
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger
from resunet_lib_keras import resunet_keras
from img_load_ops import import_and_save, augment
import owncloud
from owncloud_ops import get_latest, redundant_put

'''
Explanation of arguments:
-f: the local config ini that stores the locations of the training data, saved models, and connection details
-w: maximum time this whole script will run (in hours)
-p: time the script will wait in between checking for available files
'''

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('-f', '--file', type=str, default="local_client_test.ini")
parser.add_argument('-w', '--walltime_h', type=int, default=24)
parser.add_argument('-p', '--polltime_s', type=int, default=15)
args = parser.parse_args()


# These lines are required for the code to run successfully on TF 2.1
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set some parameters
IMG_CHANNELS = 1
N_LABELS = 2
DROPOUT_RATE = 0.5
AUG_NUM = 5

base_config = configparser.ConfigParser()
base_config.read(args.file)
model_dir = os.path.normpath(base_config['training']['model_dir'])
base_dir = os.path.normpath(base_config['training']['data_dir'])
csv_dir = os.path.join(model_dir, 'logs')

BATCH_SIZE = int(base_config['training']['batch_size'])

c_server = owncloud.Client.from_public_link(base_config['cloud']['link'], folder_password=base_config['cloud']['pw'])


if __name__ == '__main__':

    os.makedirs(csv_dir, exist_ok=True)

    start_time = time.time()
    current_fmod_h5 = 0
    current_fmod_ini = 0

    img_train, lbl_train = import_and_save(base_dir, 'train')
    img_val, lbl_val = import_and_save(base_dir, 'validation')
    img_test, lbl_test = import_and_save(base_dir, 'test')

    print(img_train.shape, lbl_train.shape)
    print(img_test.shape, lbl_test.shape)

    # Main train loop
    while time.time() - start_time < args.walltime_h * 3600:

        poll_start_time = time.time()

        # Grabs the latest train_config ini file
        current_fmod_ini, ini_path = get_latest(c_server, '.ini', current_fmod_ini, model_dir)

        # train_config is the ini file entered in the args
        train_config = configparser.ConfigParser()
        train_config.read(ini_path)
        CURRENT_EPOCH = int(train_config['params']['current_epoch'])
        LEARN_RATE = float(train_config['params']['learn_rate'])
        scratch_flag = int(train_config['env']['scratch'])

        # filestamp is the universal identifier for each neural net,
        # this is usually a datetime conversion, but can change in the future
        filestamp = train_config['env']['name']
        model_save_path = os.path.join(model_dir, filestamp + '_client.h5')
        csv_save_path = os.path.join(csv_dir, filestamp + '_log.csv')

        if scratch_flag and CURRENT_EPOCH < 1:
            print('training first epoch from scratch')
            model = resunet_keras(N_LABELS, DROPOUT_RATE, IMG_CHANNELS)
        else:
            current_fmod_h5, h5_path = get_latest(c_server, '_agg.h5', current_fmod_h5, model_dir)
            # Finalize and Compile model
            print('warm starting...')
            model = load_model(h5_path)

        # Defining callbacks used in training
        csv_logger = CSVLogger(filename=csv_save_path, append=True)

        optim = optimizers.SGD(lr=LEARN_RATE, momentum=0.9)

        img_train_aug, lbl_train_aug = augment(img_train, lbl_train, AUG_NUM)

        # Because of aggregation, we have to re-compile the model
        model.compile(optimizer=optim, loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        model.fit(
            x=img_train_aug, y=lbl_train_aug,
            batch_size=BATCH_SIZE,
            epochs=CURRENT_EPOCH + 1,
            initial_epoch=CURRENT_EPOCH,
            verbose=2,
            callbacks=[csv_logger],
            validation_data=(img_val, lbl_val),
            shuffle=True
        )

        model.save(model_save_path)

        put_result = redundant_put(c_server, '/', model_save_path)
        put_result = redundant_put(c_server, '/', csv_save_path)

        current_time = time.time() - start_time
        print("Done! Time elapsed: %f seconds." % current_time)
