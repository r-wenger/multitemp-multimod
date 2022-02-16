from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout, ConvLSTM2D
from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import plot_model
from segmentation_models.losses import CategoricalCELoss
from tensorflow.keras.utils import to_categorical

import os
from glob import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from pathlib import Path
import shutil
import random
from random import sample, choice, shuffle
import tifffile as tiff
from skimage.transform import rotate
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import rotate
import time
from Patch import Patch

def array_to_normalized(img):
    #Means and standard deviation for each 10 bands for 17 days gap for monthes ['202007', '202008', '202009', '202011']
    stds_s2_total = [439.62790900199644, 702.2237497462481, 786.5948955924482, 1142.9799955196413, 2129.662793514077,
                    2498.619403949213, 2620.957668050562, 2816.7809424162315, 2212.9231966132015, 1352.3410809599484]

    means_s2_total = [252.62887057435196, 323.5247227959752, 508.1530443259149, 457.6545855011113, 506.34562773934624,
                    628.1464899719209, 691.054464119701, 675.529670439067, 720.6092056255868, 612.9317259184817]

    for idx, mean_value in enumerate(means_s2_total):
        img[..., idx] -= mean_value
        img[..., idx] /= stds_s2_total[idx]
    return img


def time_series_image_gen(
        list_patches,
        path_gr,
        path_s2,
        img_size,
        n_channels,
        n_dates,
        classes,
        batch_size=32,
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False
):
    while True:
        n_classes = len(classes) + 1

        X = np.empty((batch_size, n_dates, img_size, img_size, n_channels))
        Y = np.empty((batch_size, img_size, img_size))
        Y_one_hot = np.empty((batch_size, img_size, img_size, n_classes))

        batch = sample(list_patches, batch_size)

        for i, patch in enumerate(batch):
            batch_X = []

            for s2 in patch.extracted_s2:
                tmp_path_s2 = os.path.join(path_s2, s2)
                tmp = np.array(tiff.imread(tmp_path_s2).astype(float))
                tmp = array_to_normalized(tmp)
                batch_X.append(tmp)
            batch_X = np.stack(batch_X, axis=0)

            label = np.array(tiff.imread(os.path.join(path_gr, patch.reconstruct_filename())).astype(float))
            label = np.where(label >= 6, 6, label)

            Y[i, ...] = np.array(label[:, :])
            X[i, ...] = np.array(batch_X)

        for s in range(batch_size):
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    for d in range(n_dates):
                        for f in range(n_channels):
                            X[s, d, ..., f] = np.flipud(X[s, d, ..., f])
                    Y[s, ...] = np.flipud(Y[s, ...])
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    for d in range(n_dates):
                        for f in range(n_channels):
                            X[s, d, ..., f] = np.fliplr(X[s, d, ..., f])
                    Y[s, ...] = np.fliplr(Y[s, ...])
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                for d in range(n_dates):
                    for f in range(n_channels):
                        X[s, d, ..., f] = rotate(X[s, d, ..., f],
                                                 angle,
                                                 mode='reflect',
                                                 order=1,
                                                 preserve_range=True
                                                 )
                Y[s, ...] = rotate(Y[s, ...],
                                   angle,
                                   mode='reflect',
                                   order=1,
                                   preserve_range=True
                                   )

            object_3D = np.zeros((img_size, img_size, n_classes), dtype='float32')

            for c in range(0, n_classes):
                object_3D[:,:,c] = np.where(Y[s, ...]==c+1, 1, 0)


            Y_one_hot[s, ...] = object_3D

        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(Y))

        yield (X, Y_one_hot)




def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(
        input)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    
    return x


def convLSTM_block(input):
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='relu'
                       , activation='relu'
                       , padding='same', return_sequences=False)(input)
    return x


def unet_lstm_model(im_sz, n_channels, time, n_classes, weights, lr):
    """ Input """
    input_shape = (time, im_sz, im_sz, n_channels)
    inputs = Input(input_shape)
    dropout_value = 0.5
    
    """ convLSTM2D block """
    convlstm = convLSTM_block(inputs)

    """ VGG16 Model """
    vgg16 = VGG16(include_top=False, weights=None, input_tensor=convlstm)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output  ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output  ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output  ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output  ## (64 x 64)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d1 = Dropout(dropout_value)(d1)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d2 = Dropout(dropout_value)(d2)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d3 = Dropout(dropout_value)(d3)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)
    d4 = Dropout(dropout_value)(d4)

    """ Output """
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(d4)

    loss = CategoricalCELoss(class_weights=weights)

    def dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator

    def focal_loss(alpha=0.25, gamma=2):
        def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
            targets = tf.cast(targets, tf.float32)
            weight_a = alpha * (1 - y_pred) ** gamma * targets
            weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

            return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

        def loss(y_true, logits):
            y_pred = tf.math.sigmoid(logits)
            loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

            return tf.reduce_mean(loss)

        return loss

    loss = 'categorical_crossentropy'

    model = Model(inputs, outputs, name="VGG16_LSTM-U-Net")

    def dice(y_true, y_pred, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    model.compile(optimizer=Adam(lr),
                  loss=loss,
                  metrics=['categorical_accuracy', dice])
    return model


def set_config_memory():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = InteractiveSession(config=config)

    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()

monthes = ['202007', '202008', '202009', '202011']
start_time = time.time()
list_patches = Patch.generate_list_patches('/media/wenger/DATA2/dataset_v1/labels')
print("--- %s seconds ---" % (time.time() - start_time))

dates_to_keep_s2 = []
list_extracted_patches = []

for e in list_patches:
    days_gap = e.has_days_gap_s2(monthes, date_format='%Y%m', days_gap=17)
    if days_gap[1]:
        e.extracted_s2 = e.reconstruct_filename('s2', days_gap[0])
        list_extracted_patches.append(e)
        dates_to_keep_s2.append(e.extracted_s2)



#Class weights for 6 classes for 17 days gap for monthes ['202007', '202008', '202009', '202011']
class_weights = [365.65572762,
                 42.46117546,
                 71.36785131,
                 350.52613946,
                 168.5718001,
                 1.41730604]

set_config_memory()

random.Random(4).shuffle(list_extracted_patches)
training = list_extracted_patches[:int(len(list_extracted_patches)*0.7)]
validation = list_extracted_patches[-int(len(list_extracted_patches)*0.2):]
testing = list_extracted_patches[-int(len(list_extracted_patches)*0.2):]

batch_size = 4
lr = 0.001
epochs = 100
patch_size = 256
n_channels = 10
n_classes = 6
classes = [1, 2, 3, 4, 5]
n_time = 4
path_s2 = '/media/wenger/DATA2/dataset_v1/s2'
path_gr = '/media/wenger/DATA2/dataset_v1/ground_reference'
model_name= '17dg-4m_7-8-9-11'

train_generator = time_series_image_gen(
        training,
        path_gr,
        path_s2,
        patch_size,
        n_channels,
        n_time,
        classes,
        batch_size=batch_size,
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True
)
train_steps = len(training)/batch_size

val_generator = time_series_image_gen(
        validation,
        path_gr,
        path_s2,
        patch_size,
        n_channels,
        n_time,
        classes,
        batch_size=batch_size
)
val_steps = len(validation)/batch_size

#Callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)

earlystopper = EarlyStopping(
    monitor='val_categorical_accuracy',
    min_delta=0.01,
    patience=20,
    verbose=1,
    mode='auto',
    baseline=0.5,
    restore_best_weights=True
)

tensorboard = TensorBoard(
    log_dir=f'./reports/logs/{model_name}',
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

model = unet_lstm_model(patch_size, n_channels, n_time, n_classes, class_weights, lr)

plot_model(model, to_file='./models/model_{model_name}.png', show_shapes=True)

start_time = time.time()
history = model.fit(train_generator, steps_per_epoch=100, epochs=epochs,
        validation_data=val_generator, validation_steps=10, callbacks=[reduce_lr, earlystopper, tensorboard])
end_time = time.time()

model.save(f'./models/model_{model_name}.h5')

with open(f'./reports/plots_and_times/{model_name}_log.txt', 'w+') as file:
    file.write(f'Training completed in {end_time-start_time:0.1f} seconds.\n')