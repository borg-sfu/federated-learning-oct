import os
import time
import owncloud
import numpy as np


def clear_folder(server):

    print('Clearing cloud folder.')
    clear_result = False

    while not clear_result:
        f_list = server.list('')

        if f_list is None:  # this is usually as a result of an error, normally empty folder = []
            print('server.list() resulted in None. Trying again...')
            continue
        elif len(f_list) == 0:
            clear_result = True
            break

        for file in f_list:
            fname = file.get_name()
            try:
                print('Removing', fname)
                del_result = server.delete(path='/' + fname)
            except Exception as e:
                print("Warning: potential error:", e)
                print("Trying again...")
                continue


def get_latest(server, file_id, current_fmod, dest_dir, dest_name=None, poll_time=15, max_poll=10000):
    """
    :param server: owncloud.Client object that links to the drop-off folder
    :param file_id: identifier for the cloud file to download
    :param current_fmod: time at which the last file was modified (int converted from DateTime object)
    :param dest_dir: local directory where cloud file is downloaded
    :param dest_name: (optional) cloud file is renamed to this when downloaded
    :param poll_time: time in seconds for loop to wait before checking cloud directory again
    :param max_poll: maximum number of checks before giving up (returns None)
    :return: new last modified time, local path to downloaded file (None if op failed)
    """
    available_flag = False
    poll_count = 1
    while not available_flag:
        try:
            f_list = server.list('')
        except Exception as e:
            print("Warning: potential error:", e)
            print("Trying again...")
            time.sleep(2)
            continue

        if f_list is None:      # this is usually as a result of an error
            print('server.list() resulted in None. Trying again...')
            time.sleep(2)
            continue

        for file in f_list:
            fname = file.get_name()

            if file_id in fname:
                fmod = file.get_last_modified()
                fmod_i = int(fmod.strftime("%Y%m%d%H%M%S"))

                # only trigger if file is what we're looking for and is newer than the previous fmod
                if fmod_i > current_fmod or np.isnan(current_fmod):
                    print('\nFile found: ' + fname)

                    dest_path = os.path.join(dest_dir, fname) if dest_name is None else os.path.join(dest_dir,
                                                                                                     dest_name)

                    # get_file returns true if success, false if not
                    try:
                        available_flag = server.get_file(remote_path='/' + fname, local_file=dest_path)
                    except Exception as e:
                        print("Warning: potential error:", e)
                        print("Trying again...")
                        continue

        if poll_count > max_poll:
            return None, None       # main script recognizes this as a failure

        if not available_flag:
            print("File not found. Trying again in " + str(poll_time) +
                  " seconds." + "(" + str(poll_count) + ")", end='\r')
            time.sleep(poll_time)
            poll_count += 1

    print('Successfully downloaded.')
    return fmod_i, dest_path


def redundant_put(server, dest_path, local_path):
    """
    :param server: owncloud.Client object that links to the drop-off folder
    :param dest_path: owncloud path where file is uploaded
    :param local_path: path where file is uploaded from
    :return: True if success, False if not
    """
    result = False
    timeout = 20 * 60
    put_start_time = time.time()
    while not result and time.time() - put_start_time < timeout:

        try:
            result = server.put_file(remote_path=dest_path,
                                     local_source_file=local_path,
                                     chunked=False,
                                     keep_mtime=False)
        except Exception as e:
            print("Warning: potential error:", e)
            print("Trying again...")
            continue

        if result:
            print("Successfully uploaded " + os.path.basename(local_path))

    return result
