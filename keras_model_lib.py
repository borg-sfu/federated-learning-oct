''' Consists of two functions used for architecture definition and to freeze weights (transfer learning) '''

# import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.applications import vgg16, vgg19, resnet50
from tensorflow import keras


def lock_weights(temp_model, num_layers):
    ''' locks num_layers starting from the input layer of temp_model '''
    model = []
    model = Sequential()

    # Locking the specified number of layers starting from the bottom layers of the model
    for i, layer in enumerate(temp_model.layers):
        if i < num_layers:
            layer.trainable = False
        model.add(layer)

    return model
    
    
def keras_models(input_shape, class_num, weight_config, num_layers, model_type):
    ''' Keras models for loading VGG16, VGG19, or ResNet50 '''
    base_model = []
    if weight_config == "None":
        weight_config = None

    if model_type == 'VGG16':
        base_model = vgg16.VGG16(include_top=False, weights='imagenet',
                input_shape=input_shape)
        
    elif model_type == 'VGG19':
        base_model = vgg19.VGG19(include_top=False, weights='imagenet',
                input_shape=input_shape)

    elif model_type == 'ResNet50':
        base_model = resnet50.ResNet50(include_top=False, weights='imagenet',
                input_shape=input_shape)
                 
    else:
        raise Exception("Invalid model_type listed in the configuration file. Must be VGG16, VGG19, or ResNet50... otherwise, edit and add the base_model in the function 'keras_models' within keras_model_lib.py")

    model = []
    model = Sequential()
    ctr = 0

    # Locking the specified number of layers starting from the bottom layers of the model
    for i, layer in enumerate(base_model.layers):
        if i <= num_layers:
            layer.trainable = False
            ctr = ctr + 1
        else:
            layer.trainable = True
        model.add(layer)
        total = i

    # adding one 'hidden layer' as part of the classifier - can be altered easily
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(class_num, activation='sigmoid'))

    
    return model
