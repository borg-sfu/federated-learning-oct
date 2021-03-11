''' This is the aggregator script that notifies the participating clients through a configuration file and receives the trained models every epoch. The trained models are aggregated. '''

# import libraries
import time
import numpy as np
import os
from tensorflow.keras.models import load_model, clone_model
import tensorflow as tf
import gc
import configparser
import csv
import owncloud
import argparse
import pandas as pd
import math

# import from local files
from owncloud_ops import get_latest, redundant_put, clear_folder
from img_loading import import_and_save, load_data_from_folds
from misc_functions import str2bool


def lr_scheduler(epoch, total_epochs, LR_Specs): # triangular2 (halved every full cycle)
    ''' learning rate scheduler with 4 options
        - steady - constant learning rate (returns the max LR defined in LR_Specs constantly)
        - step_decay - decays with the number of steps defined in LR_Specs
        - tri1 - cyclic learning rate which cycles from max to min for the number of cycles defiend in LR_Specs
        - tri2 - cyclic learning rate which cycles like tri1 by decays exponentially after each cycle
    '''
    # obtain values from LR_Specs
    # minimum learning rate
    lr_floor = LR_Specs[1]
    decay_factor = 0.5 # used for tri2 decay cyclic lr

    # maximum learning rate, step size, and cycle count (if cyclic)
    lr_max = LR_Specs[0]
    step_size = total_epochs/(LR_Specs[3]*2) # into 100/10 -> step size of 10
    cycle_count = np.floor((epoch+0.0001)/step_size) # if odd, we increase to max

    if LR_Specs[2] == 'steady':
        # steady unchanging learning rate
        return lr_max

    elif LR_Specs[2] == 'step_decay':
        # decaying constantly by step_size
        delta_lr_per_step = (lr_max-lr_floor)/LR_Specs[3]
        step_size = total_epochs/(LR_Specs[3]) # into 100/10 -> step size of 10
        return lr_max - (delta_lr_per_step * (epoch // step_size))

    elif LR_Specs[2] == 'tri1':
        # cycle from max to min with no decay
        delta_lr_per_step = (lr_max-lr_floor)/step_size
        if ((cycle_count + 1) % 2) == 0: # Even
            outputLR = lr_floor + (delta_lr_per_step * ((epoch) % step_size))
        else: # Odd
            # decreasing from max to floor
            outputLR = lr_max - (delta_lr_per_step * ((epoch) % step_size))
        return outputLR 

    elif LR_Specs[2] == 'tri2':
        # cycle from max to min with decay by half each cycle
        curr_lr_max  = lr_max * (decay_factor ** np.floor(cycle_count//2))
        delta_lr_per_step = (curr_lr_max-lr_floor)/step_size
        if ((cycle_count + 1) % 2) == 0: # Even
            outputLR = lr_floor + (delta_lr_per_step * ((epoch) % step_size))
        else: # Odd
            # decreasing from max to floor
            outputLR = curr_lr_max - (delta_lr_per_step * ((epoch) % step_size))
        return outputLR 

    else:
        # must be one of the above learning rates. redefine this below in the definiton of LR_Specs
        raise Exception("Improper learning rate type - needs to be one of (steady, step_decay, tri1, tri2). Fix in fed_aggregator_oc.py and see more details lr_scheduler of fed_aggregator_oc.py")



def get_lr(initial_lr, epoch, decay_factor, step_const, max_steps):
    """
    :param initial_lr: the learn rate that all calculations will be based off of
    :param epoch: the current epoch
    :param decay_factor: the constant the initial_lr will be multiplied by
    :param step_const: how many epochs before one decay step
    :param max_steps: the max amount of decay steps before decay reversal (triangle) or reset (default)
    :return: the current learning rate used for this specific epoch
    """

    if args.wave == "default":
        # right triangle wave
        return initial_lr * (decay_factor ** (np.floor(epoch / step_const) % max_steps))
    elif args.wave == "triangle":
        # triangle wave
        return initial_lr * (
                    decay_factor ** (max_steps - np.abs((np.floor(epoch / step_const) % (max_steps * 2)) - max_steps)))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU isn't needed for aggregation so this prevents CUDA_MEMORY errors
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''
Explanation of arguments:
-f: the local config ini that stores the information for each client 
    (namely how to access their respective drop-off folders)
-f_config: configuration file that informs the algorithm about variables like number of folds, image size, etc..
-of: outer fold in cross validation to continue from
-if: inner fold in cross validation to continue from

-cb: boolean for whether we are starting from scratch ('False') or continuing from a specific epoch ('True')
-ce: (int) which epoch to continue from within the folds specified by -of and -if
-ts: the specific time_stamp corresponding to the training that would like to be continued

-p: time the script will wait in between checking for available files
-m: maximum number of times the script will try downloading a file before it gives up and returns None
-w: the learning rate curve  that will be used for training
'''

# acquire input variables from the user
parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('-f', '--file', type=str, default="fed_clients.ini")
parser.add_argument('-f_config', '--config_file', type=str, default="fed_config.ini")
parser.add_argument('-of', '--outer_fold', type=int, default=0)
parser.add_argument('-if', '--inner_fold', type=int, default=0)

parser.add_argument('-cb', '--continue_bool', type=str, default='False')
parser.add_argument('-ce', '--continue_epoch', type=int, default=1)
parser.add_argument('-ts', '--time_stamp', type=str, default='20201214-220202')

parser.add_argument('-p', '--polltime_s', type=int, default=15)
parser.add_argument('-m', '--maxpoll', type=int, default=10000)
parser.add_argument('-w', '--wave', type=str, default="default")
args = parser.parse_args()


# fed config file
base_config_fed = configparser.ConfigParser()
base_config_fed.read(args.config_file)
image_size = int(base_config_fed['params']['image_size'])
num_folds = int(base_config_fed['params']['num_folds'])

# directories and institution ID to load data for malicious model detection
institution_ID  = str(base_config_fed['params']['institution_ID'])
base_dir = os.path.normpath(base_config_fed['params']['base_dir'])
test_dir = os.path.normpath(base_config_fed['params']['test_dir'])

# number of epochs to train each fold
epochs = int(base_config_fed['params']['epochs'])

# read LR params from the agg config file
max_lr = float(base_config_fed['params']['max_lr'])
min_lr = float(base_config_fed['params']['min_lr'])
num_lr_cycles = int(base_config_fed['params']['num_lr_cycles'])
lr_type = str(base_config_fed['params']['lr_type'])

# tolerance where a model that performs with accuracy on sample data will be removed - usually set low ( < 0.3) to remove bias - the value depends on the use case.
test_tolerance = float(base_config_fed['params']['test_tolerance'])

# set the number of outer_folds to loop through for this test
num_outer = num_folds

LR_Specs = [max_lr, min_lr, lr_type, num_lr_cycles]

# config file to direct the aggregator to the drop-off folder
base_config = configparser.ConfigParser()
base_config.read(args.file)

# set outer_fold_start and inner_fold_start to the folds that testing should start at. 
# i.e. if the first set of outer_fold has been completed, you can resume training at the second set of outer_folds by setting outer_fold_start = 1
outer_fold_start = args.outer_fold
inner_fold_start = args.inner_fold

# image shape from the 1D size specified - if not square image, must change this along with the participating institutions
INPUT_SHAPE = (image_size, image_size, 3)

resume_flag = str2bool(args.continue_bool)

resume_epoch = args.continue_epoch-1  # 1+ the last successfully-trained epoch
resume_filestamp = args.time_stamp

# check the institution ID specified for model verification
if institution_ID != 'SFU' and institution_ID != 'OHSU' and institution_ID != 'UW' and institution_ID != 'CUHK':
    raise Exception("Please specify institution ID: OHSU, UW, or SFU with flag -ID")

# Each net has a filename with the date and time at which it started training
datetime_filestamp = time.strftime("%Y%m%d-%H%M%S")

scratch_flag = True

first_loop = False
filestamp = resume_filestamp if resume_flag else datetime_filestamp

if resume_epoch < 1:
    resume_flag = False

if __name__ == '__main__':

    # Client info is found in config files
    base_config = configparser.ConfigParser()
    base_config.read(args.file)
    client_list = base_config.sections()
    num_clients = len(client_list)
    agg_dir = os.path.join(base_dir, filestamp)
    os.makedirs(agg_dir, exist_ok=True)
    
    # loading data from one of the clients - this is only for verification that there are no malicious networks being sent back (as long as performance is > test_tolerance, models will be aggregated the same)
    outer_fold = 0
    inner_fold = 0
    img_test, lbl_test, img_strat_test, lbl_strat_test, img_val, lbl_val = load_data_from_folds(test_dir, outer_fold, inner_fold, image_size, INPUT_SHAPE, num_folds, institution_ID, 'test')

    current_fmod_h5 = np.zeros(len(client_list))        # used to find the newest files on the drop-off folder
    current_fmod_csv = np.zeros(len(client_list))

    if resume_flag:
        loop_start = resume_epoch
        print('resuming from epoch ' + str(resume_epoch + 1))
        first_loop = True
    else:
        loop_start = 0

    # for loop that loops through the specified number of outer_folds
    for curr_outer_fold in range(num_outer):
        # offset by the starting outer fold
        outer_fold_idx = curr_outer_fold + outer_fold_start # starts at 0 if outer_fold_start is 0. 

        if outer_fold_idx > num_outer-1:
            continue
        
        # for loop that loops through the number of inner folds
        for curr_inner_fold in range(num_folds-1):
            
            # offset by the starting inner fold
            inner_fold_idx = curr_inner_fold + inner_fold_start # starts at 0 if inner_fold_start is 0. 

            if inner_fold_idx > num_folds-2:
                continue
            ''' begin testing the specified fold '''

            # reset val loss and min_val_loss of the networks
            val_loss = np.zeros(len(client_list), dtype=np.float32)

            # always put first model as aggregator or there will be an error
            min_val_loss = 1000

            # only the first fold starts from 'e'
            if first_loop:
                e = loop_start
                # no subsequent folds start from this epoch
                first_loop = False
            else:
                e = 0

            print('############# begin testing outer fold:', outer_fold_idx, 'and inner fold:', inner_fold_idx, '#############')

            # for each fold set, loop through to the number of epochs
            while e < epochs:
                if e < 1:
                    scratch_flag = True
                # clearing memory from multiple model creation: https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
                tf.keras.backend.clear_session()
                print('cleared backend tensorflow session...')
                print('starting epoch ' + str(e + 1))

                model_paths = list()

                # setting lrate based on epoch num
                lrate = round(lr_scheduler(e, epochs, LR_Specs),7)
                print('learning rate:', lrate)
                train_config = configparser.ConfigParser()

                # saving only the best aggregated model
                agg_save_filename = filestamp + '_OF_'+ str(outer_fold_idx) + '_IF_' + str(inner_fold_idx) + '_agg.h5'
                # agg_save_filename = filestamp + '-' + str(e) + '_OF_'+ str(outer_fold_idx) + '_IF_' + str(inner_fold_idx) + '_agg.h5'
                agg_send_path = '/' + filestamp + '_OF_'+ str(outer_fold_idx) + '_IF_' + str(inner_fold_idx) + '_agg.h5'
                agg_save_path = os.path.join(agg_dir, agg_save_filename)

                # *_p variables are for the previous epoch, which is useful when distributing last epoch's files to clients
                agg_save_filename_p = filestamp + '_OF_'+ str(outer_fold_idx) + '_IF_' + str(inner_fold_idx) +'_agg.h5'
                agg_save_path_p = os.path.join(agg_dir, agg_save_filename_p)

                # Main put loop - distributes files to clients to start training
                for i in range(len(client_list)):
                    header = client_list[i]
                    print('client ID:' + header)
                    client_dir = os.path.join(agg_dir, header)
                    config_path = os.path.join(client_dir, 'train_specs.ini')
                    os.makedirs(client_dir, exist_ok=True)

                    # configparser requires inputs to be strings
                    train_config = configparser.ConfigParser()
                    train_config['env'] = {}
                    train_config['env']['name'] = filestamp

                    # accounting for cross validation. If resume, we set scratch equal to 1 only the first time. Afterwards, nothing will be from scratch except epochs less than 0
                    if resume_flag or scratch_flag:
                        scratch_str = '1'
                        # update flags after all clients have run
                        if i == len(client_list)-1:
                            scratch_flag = False
                            resume_flag = False
                    else:
                        scratch_str = '0'

                    train_config['env']['scratch'] = scratch_str

                    # update new training params for current epoch
                    train_config['params'] = {}
                    train_config['params']['current_epoch'] = str(e)
                    train_config['params']['learn_rate'] = str(lrate)

                    # update the current outer fold and inner fold
                    train_config['params']['outer_fold'] = str(outer_fold_idx)
                    train_config['params']['inner_fold'] = str(inner_fold_idx)

                    with open(config_path, 'w') as config_file:
                        train_config.write(config_file)

                    if base_config[header]['method'] == 'owncloud':
                        c_server = owncloud.Client.from_public_link(base_config[header]['link'],
                                                                    folder_password=base_config[header]['pw'])
                        if e == loop_start:
                            clear_folder(c_server)      # clear out any files in the cloud folder

                        # send both the train_config and aggregate h5 files to each client
                        put_result = redundant_put(c_server, '/', config_path)
                        if e > 0 or resume_flag:
                            print('put result')
                            put_result = redundant_put(c_server, agg_send_path, agg_save_path_p)
                    else:
                        # TODO: add ssh ops here
                        print('Something went wrong. Check base_config.')
                        pass
                # Main get loop - receives client h5 and csvs from each client
                for i in range(len(client_list)):
                    header = client_list[i]
                    print('client ID:' + header)
                    client_dir = os.path.join(agg_dir, header)
                    name_stamp = 'client' + str(i) + '_OF_'+ str(outer_fold_idx) + '_IF_' + str(inner_fold_idx) + '-' + str(e)

                    if base_config[header]['method'] == 'owncloud':
                        c_server = owncloud.Client.from_public_link(base_config[header]['link'],
                                                                    folder_password=base_config[header]['pw'])

                        # get latest h5 and csv files
                        current_fmod_h5[i], h5_path = get_latest(c_server,
                                                                filestamp + '_OF_'+ str(outer_fold_idx) + '_IF_' + str(inner_fold_idx) + '_client.h5', current_fmod_h5[i], client_dir,
                                                                name_stamp + '.h5', args.polltime_s, args.maxpoll)
                        current_fmod_csv[i], csv_path = get_latest(c_server,
                                                                filestamp + '_OF_'+ str(outer_fold_idx) + '_IF_' + str(inner_fold_idx) + '_log.csv', current_fmod_csv[i], client_dir,
                                                                name_stamp + '.csv', args.polltime_s, args.maxpoll)
                    else:
                        # TODO: add ssh ops here
                        print('Something went wrong. Check base_config.')
                        pass

                    if h5_path is not None:
                        model_paths.append(h5_path)

                        with open(csv_path) as f:
                            row = list(csv.reader(f))[-1]
                            # val_loss[i] = float(row[3])
                            val_loss[i] = float(row[4])
                            print('Validation loss for client ' + str(i) + ' (' + header + ') = ' + str(val_loss[i]))
                    else:
                        print('Skipping client ' + str(i) + ' (' + header + ')')

                if len(model_paths) == 0:
                    print('No models returned. Restarting epoch ' + str(e + 1))
                    continue

                agg_start_time = time.time()
                print('aggregating weights for epoch ' + str(e + 1))
                print('aggregating number of models: ', len(model_paths))
                models = list()
                weights = list()
                new_weights = list()

                for model_path in model_paths:
                    model = load_model(model_path)

                    try:
                        # this weeds out client models that have diverged or would otherwise disrupt the federated process
                        eval_result = model.evaluate(x=img_test, y=lbl_test, batch_size=1)

                        if eval_result[1] > test_tolerance:
                            models.append(model)
                            weights.append(model.get_weights())
                    except Exception as e:
                        print("Warning: potential error:", e)
                        continue

                # get mean weights in each model
                for weights_list_tuple in zip(*weights):
                    new_weights.append(np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))
                
                mean_val_loss = np.mean(val_loss)

                # all the models are saved, but this provides info on which one to keep
                if mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss
                    print('New lowest average validation loss at ' + str(min_val_loss))
                    
                    # create a new model and set its weights to our new aggregate ONLY SAVING THE BEST AGGREGATED MODEL AND OVERWRITING ALL
                    new_model = clone_model(models[0])
                    new_model.set_weights(new_weights)
                    new_model.save(agg_save_path)  # model.compile must be called before training
                    # new_model.save(os.path.join(agg_dir, filestamp + '-' + str(e) + '-best.h5'))

                current_time = time.time() - agg_start_time
                print("Aggregation time: %f seconds." % current_time)

                # increment the loop iterator and garbage collect to remove unnecessary variables to clear up memory
                e += 1
                new_model = None
                models = None
                weights = None
                model_path = None
                gc.collect()

        # restart inner loop from start after first run
        inner_fold_start = 0
