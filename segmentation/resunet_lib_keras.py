import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, \
    concatenate, add, ReLU


def resunet_keras(nlabels, dr_rate, num_channels):
    # Model
    ModelIn = Input((None, None, num_channels))

    # Left Layer 1 - 128
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(ModelIn)
    c1_bn = BatchNormalization()(c1)
    c1_dr = Dropout(dr_rate)(c1_bn)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1_dr)
    c2_bn = BatchNormalization()(c2)
    c2_dr = Dropout(dr_rate)(c2_bn)

    s1 = Conv2D(32, (1, 1), activation='relu', padding='same')(ModelIn)
    c2_out = add([c2_dr, s1])

    # Left Layer 2 - 64
    p1 = MaxPooling2D((2, 2), padding='same')(c2_out)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c3_bn = BatchNormalization()(c3)
    c3_dr = Dropout(dr_rate)(c3_bn)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3_dr)
    c4_bn = BatchNormalization()(c4)
    c4_dr = Dropout(dr_rate)(c4_bn)

    s2 = Conv2D(64, (1, 1), activation='relu', padding='same')(p1)
    c4_out = add([c4_dr, s2])

    # Left Layer 3 - 32
    p2 = MaxPooling2D((2, 2), padding='same')(c4_out)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c5_bn = BatchNormalization()(c5)
    c5_dr = Dropout(dr_rate)(c5_bn)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5_dr)
    c6_bn = BatchNormalization()(c6)
    c6_dr = Dropout(dr_rate)(c6_bn)

    s3 = Conv2D(128, (1, 1), activation='relu', padding='same')(p2)
    c6_out = add([c6_dr, s3])

    # Left Layer 4 - 16
    p3 = MaxPooling2D((2, 2), padding='same')(c6_out)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c7_bn = BatchNormalization()(c7)
    c7_dr = Dropout(dr_rate)(c7_bn)
    c8 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7_dr)
    c8_bn = BatchNormalization()(c8)
    c8_dr = Dropout(dr_rate)(c8_bn)

    s4 = Conv2D(256, (1, 1), activation='relu', padding='same')(p3)
    c8_out = add([c8_dr, s4])

    # Bottom Layer - 8
    p4 = MaxPooling2D((2, 2), padding='same')(c8_out)
    c9 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c9_bn = BatchNormalization()(c9)
    c9_dr = Dropout(dr_rate)(c9_bn)
    c10 = Conv2D(512, (3, 3), activation='relu', padding='same')(c9_dr)
    c10_bn = BatchNormalization()(c10)
    c10_dr = Dropout(dr_rate)(c10_bn)

    s5 = Conv2D(512, (1, 1), activation='relu', padding='same')(p4)
    c10_out = add([c10_dr, s5])

    # Right Layer 4 - 16
    uc1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c10_out)
    uc1 = concatenate([uc1, c8_out])
    c11 = Conv2D(256, (3, 3), activation='relu', padding='same')(uc1)
    c11_bn = BatchNormalization()(c11)
    c11_dr = Dropout(dr_rate)(c11_bn)
    c12 = Conv2D(256, (3, 3), activation='relu', padding='same')(c11_dr)
    c12_bn = BatchNormalization()(c12)
    c12_dr = Dropout(dr_rate)(c12_bn)

    s6 = Conv2D(256, (1, 1), activation='relu', padding='same')(uc1)
    c12_out = add([c12_dr, s6])

    # Right Layer 3 - 32
    uc2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c12_out)
    uc2 = concatenate([uc2, c6_out])
    c13 = Conv2D(128, (3, 3), activation='relu', padding='same')(uc2)
    c13_bn = BatchNormalization()(c13)
    c13_dr = Dropout(dr_rate)(c13_bn)
    c14 = Conv2D(128, (3, 3), activation='relu', padding='same')(c13_dr)
    c14_bn = BatchNormalization()(c14)
    c14_dr = Dropout(dr_rate)(c14_bn)

    s7 = Conv2D(128, (1, 1), activation='relu', padding='same')(uc2)
    c14_out = add([c14_dr, s7])

    # Right Layer 2 - 64
    uc3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c14_out)
    uc3 = concatenate([uc3, c4_out])
    c15 = Conv2D(64, (3, 3), activation='relu', padding='same')(uc3)
    c15_bn = BatchNormalization()(c15)
    c15_dr = Dropout(dr_rate)(c15_bn)
    c16 = Conv2D(64, (3, 3), activation='relu', padding='same')(c15_dr)
    c16_bn = BatchNormalization()(c16)
    c16_dr = Dropout(dr_rate)(c16_bn)

    s8 = Conv2D(64, (1, 1), activation='relu', padding='same')(uc3)
    c16_out = add([c16_dr, s8])

    # Right Layer 1 - 128
    uc4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c16_out)
    uc4 = concatenate([uc4, c2_out])
    c17 = Conv2D(32, (3, 3), activation='relu', padding='same')(uc4)
    c17_bn = BatchNormalization()(c17)
    c17_dr = Dropout(dr_rate)(c17_bn)
    c18 = Conv2D(32, (3, 3), activation='relu', padding='same')(c17_dr)
    c18_bn = BatchNormalization()(c18)
    c18_dr = Dropout(dr_rate)(c18_bn)

    s9 = Conv2D(32, (1, 1), activation='relu', padding='same')(uc4)
    c18_out = add([c18_dr, s9])

    ModelOut = Conv2D(nlabels, (1, 1), activation='softmax')(c18_out)

    # Finalize and Compile model
    output_model = Model(inputs=[ModelIn], outputs=[ModelOut])

    return output_model
