import time
import numpy as np
import os
from tensorflow.keras.models import load_model, clone_model
from owncloud_ops import get_latest, redundant_put, clear_folder
from img_load_ops import import_and_save
import configparser
import csv
import owncloud
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU isn't needed for aggregation so this prevents CUDA_MEMORY errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''
Explanation of arguments:
-f: the local config ini that stores the information for each client 
    (namely how to access their respective drop-off folders)
-p: time the script will wait in between checking for available files
-m: maximum number of times the script will try downloading a file before it gives up and returns None
-w: the learning rate curve  that will be used for training
'''

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('-f', '--file', type=str, default="fed_clients.ini")
parser.add_argument('-c', '--config', type=str, default="fed_config.ini")
parser.add_argument('-p', '--polltime_s', type=int, default=15)
parser.add_argument('-m', '--maxpoll', type=int, default=10000)
parser.add_argument('-w', '--wave', type=str, default="default")
args = parser.parse_args()


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


if __name__ == '__main__':

    # defining paths from config file
    path_config = configparser.ConfigParser(inline_comment_prefixes = (";",))
    path_config.read(args.config)

    base_dir = os.path.normpath(path_config['train_params']['base_dir'])
    test_dir = os.path.normpath(path_config['train_params']['test_dir'])
    epochs = int(path_config['train_params']['epochs'])
    test_tolerance = float(path_config['train_params']['test_tolerance'])

    resume_flag = int(path_config['resume_params']['resume_flag'])
    resume_epoch = path_config['resume_params']['resume_epoch']  # 1 + the last successfully-trained epoch
    resume_filestamp = path_config['resume_params']['resume_filestamp']

    # Each net has a filename with the date and time at which it started training
    datetime_filestamp = time.strftime("%Y%m%d-%H%M%S")
    filestamp = resume_filestamp if resume_flag else datetime_filestamp

    # Client info is also found in config files
    base_config = configparser.ConfigParser()
    base_config.read(args.file)
    client_list = base_config.sections()
    num_clients = len(client_list)
    agg_dir = os.path.join(base_dir, filestamp)
    os.makedirs(agg_dir, exist_ok=True)

    img_test, lbl_test = import_and_save(test_dir, 'test')

    current_fmod_h5 = np.zeros(len(client_list))        # used to find the newest files on the drop-off folder
    current_fmod_csv = np.zeros(len(client_list))

    if resume_flag:
        loop_start = resume_epoch
        print('resuming from epoch ' + str(resume_epoch + 1))
    else:
        loop_start = 0

    val_loss = np.zeros(len(client_list), dtype=np.float32)
    min_val_loss = 1

    e = loop_start
    while e < epochs:
        print('starting epoch ' + str(e + 1))

        model_paths = list()

        # setting lrate based on epoch num
        lrate = get_lr(initial_lr=0.1, epoch=e, decay_factor=0.1, step_const=1, max_steps=3)

        train_config = configparser.ConfigParser()

        agg_save_filename = filestamp + '-' + str(e) + '_agg.h5'
        agg_send_path = '/' + filestamp + '_agg.h5'
        agg_save_path = os.path.join(agg_dir, agg_save_filename)

        # *_p variables are for the previous epoch, which is useful when distributing last epoch's files to clients
        agg_save_filename_p = filestamp + '-' + str(e - 1) + '_agg.h5'
        agg_save_path_p = os.path.join(agg_dir, agg_save_filename_p)

        # Main put loop - distributes files to clients to start training
        for i, header in enumerate(client_list):

            print('client ID:' + header)
            client_dir = os.path.join(agg_dir, header)
            config_path = os.path.join(client_dir, 'train_specs.ini')
            os.makedirs(client_dir, exist_ok=True)

            # configparser requires inputs to be strings
            train_config = configparser.ConfigParser()
            train_config['env'] = {}
            train_config['env']['name'] = filestamp
            train_config['env']['scratch'] = '0' if resume_flag else '1'

            # update new training params for current epoch
            train_config['params'] = {}
            train_config['params']['current_epoch'] = str(e)
            train_config['params']['learn_rate'] = str(lrate)
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
                    put_result = redundant_put(c_server, agg_send_path, agg_save_path_p)
            else:
                # TODO: add ssh ops here
                print('Something went wrong. Check base_config.')
                pass

        # Main get loop - receives client h5 and csvs from each client
        for i, header in enumerate(client_list):
            print('client ID:' + header)
            client_dir = os.path.join(agg_dir, header)
            name_stamp = 'client' + str(i) + '-' + str(e)

            if base_config[header]['method'] == 'owncloud':
                c_server = owncloud.Client.from_public_link(base_config[header]['link'],
                                                            folder_password=base_config[header]['pw'])

                # get latest h5 and csv files
                current_fmod_h5[i], h5_path = get_latest(c_server,
                                                         filestamp + '_client.h5', current_fmod_h5[i], client_dir,
                                                         name_stamp + '.h5', args.polltime_s, args.maxpoll)
                current_fmod_csv[i], csv_path = get_latest(c_server,
                                                           filestamp + '_log.csv', current_fmod_csv[i], client_dir,
                                                           name_stamp + '.csv', args.polltime_s, args.maxpoll)
            else:
                # TODO: add ssh ops here
                print('Something went wrong. Check base_config.')
                pass

            if h5_path is not None:
                model_paths.append(h5_path)

                with open(csv_path) as f:
                    row = list(csv.reader(f))[-1]
                    val_loss[i] = float(row[3])
                    print('Validation loss for client ' + str(i) + ' (' + header + ') = ' + str(val_loss[i]))
            else:
                print('Skipping client ' + str(i) + ' (' + header + ')')

        if len(model_paths) == 0:
            print('No models returned. Restarting epoch ' + str(e + 1))
            continue

        agg_start_time = time.time()
        print('aggregating weights for epoch ' + str(e + 1))
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

        # create a new model and set its weights to our new aggregate
        new_model = clone_model(models[0])
        new_model.set_weights(new_weights)
        new_model.save(agg_save_path)  # model.compile must be called before training

        mean_val_loss = np.mean(val_loss)

        # all the models are saved, but this provides info on which one to keep
        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            print('New lowest average validation loss at ' + str(min_val_loss))
            # new_model.save(os.path.join(agg_dir, filestamp + '-' + str(e) + '-best.h5'))

        current_time = time.time() - agg_start_time
        print("Aggregation time: %f seconds." % current_time)

        # increment the loop iterator
        e += 1
