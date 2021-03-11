''' 
Some helper functions to assist in logging, visualization of images, listing directories, etc... 
May be helpful, but the current implementation only requires 'str2bool' to process the user inputs from argparse.
'''

from PIL import Image
import os
from os import listdir
from num2words import num2words 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_roc_curve, roc_curve, auc                            

def list_files(directory, extension):
    ''' listing files in directory - http://www.martinbroadhurst.com/listing-all-files-in-a-directory-with-a-certain-extension-in-python.html'''
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def str2bool(v):
    ''' converting string variable into boolean variable (used to process the user inputs from argparse) - https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise 'Boolean value expected.'

def invert_matrix(matrix):
    ''' invert the matrix '''
    return np.amax(matrix) - matrix

def print_and_log(string_to_log, file1):
    ''' log string to file '''
    print(string_to_log)
    file1.write(string_to_log+"\n")
    return file1

def saveImages(image, title_name, path):
    ''' saving images as .png '''
    pathname = os.path.join(path, title_name + '.png')
    im = Image.fromarray(image)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(pathname)
    print('sucessfully saved:', title_name)