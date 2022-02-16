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
import time
from Patch import Patch

class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, list_patches, means, stds, path_s2, path_gr, n_classes = 14, batch_size=16, shuffle=True, dataAugment=True):
        'Initialization'
        self.list_patches = list_patches
        self.means = means
        self.stds = stds
        self.path_s2 = path_s2
        self.path_gr = path_gr
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataAugment = dataAugment
        self.on_epoch_end()

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_patches) / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_patches_temp = [self.list_patches[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_patches_temp)

        return X, y

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_patches))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    def array_to_normalized(self, img):

        for idx, mean_value in enumerate(self.means):
            img[:,:, idx] -= mean_value
            img[:,:, idx] /= self.stds[idx]
        return img
            
            
    def data_augmentation(self, img, mask):
        """
        :param img: ndarray with shape (x_sz, y_sz, num_channels)
        :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
        :return: patch with shape (sz, sz, num_channels)
        """

        patch_img = img.astype(float)
        patch_mask = mask

        random_transformation = np.random.randint(1,5)
        if random_transformation == 1: #rotate 90 degrees
            patch_img = np.rot90(patch_img, 1)
            patch_mask = np.rot90(patch_mask, 1)
        elif random_transformation == 2: #rotate 180 degrees
            patch_img = np.rot90(patch_img, 2)
            patch_mask = np.rot90(patch_mask, 2)
        elif random_transformation == 3: #rotate 270 degrees
            patch_img = np.rot90(patch_img, 3)
            patch_mask = np.rot90(patch_mask, 3)
        elif random_transformation == 4: #flipping up to down
            patch_img = np.flipud(patch_img)
            patch_mask = np.flipud(patch_mask)
        elif random_transformation == 5: #flipping left to right
            patch_img = np.fliplr(patch_img)
            patch_mask = np.fliplr(patch_mask)

        return patch_img, patch_mask

    
    def __data_generation(self, list_patches_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_patches_temp:
            img = []
            label = tiff.imread(os.path.join(path_gr, i.reconstruct_filename())).astype(int)

            object_3D = np.zeros((256, 256, self.n_classes), dtype='float32')
            for c in range(0, self.n_classes):
                object_3D[:,:,c] = np.where(label==c+1, 1, 0)
            label = object_3D
            
            for s2 in i.extracted_s2:
                tmp_path_s2 = os.path.join(self.path_s2, s2)
                tmp = tiff.imread(tmp_path_s2).astype(float)
                tmp = self.array_to_normalized(tmp)
                img.append(tmp)
            img = np.stack(img, axis=0)
            	
            if self.dataAugment:
                batch_labels.append(label)
                batch_imgs.append(img)
                img, label = self.data_augmentation(img, label)
            else:
                batch_labels.append(label)
                batch_imgs.append(img)
            
        return np.array(batch_imgs), np.array(batch_labels)


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

    #loss = CategoricalCELoss(class_weights=weights)

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

    loss = focal_loss(alpha=0.25, gamma=2)

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

#Means and standard deviation for each 10 bands for 17 days gap for monthes ['202007', '202008', '202009', '202011']
stds_s2_total = [439.62790900199644, 702.2237497462481, 786.5948955924482, 1142.9799955196413, 2129.662793514077,
                2498.619403949213, 2620.957668050562, 2816.7809424162315, 2212.9231966132015, 1352.3410809599484]

means_s2_total = [252.62887057435196, 323.5247227959752, 508.1530443259149, 457.6545855011113, 506.34562773934624,
                628.1464899719209, 691.054464119701, 675.529670439067, 720.6092056255868, 612.9317259184817]

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

batch_size = 2
lr = 0.0001
epochs = 100
patch_size = 256
n_channels = 10
n_classes = 14
n_time = 4
path_s2 = '/media/wenger/DATA2/dataset_v1/s2'
path_gr = '/media/wenger/DATA2/dataset_v1/ground_reference'
model_name= '17dg-4m_7-8-9-11'

train_generator = DataGenerator(training, means_s2_total, stds_s2_total, path_s2, path_gr, n_classes, batch_size=batch_size, shuffle=True, dataAugment=True)
train_steps = 100

val_generator = DataGenerator(validation, means_s2_total, stds_s2_total, path_s2, path_gr, n_classes, batch_size=batch_size, shuffle=True, dataAugment=False)
val_steps = 10

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

plot_model(model, to_file=f'./models/model_{model_name}.png', show_shapes=True)

start_time = time.time()
history = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=epochs,
        validation_data=val_generator, validation_steps=val_steps,callbacks=[reduce_lr, earlystopper, tensorboard])
end_time = time.time()

model.save(f'./models/model_{model_name}.h5')

with open(f'./reports/plots_and_times/{model_name}_log.txt', 'w+') as file:
    file.write(f'Training completed in {end_time-start_time:0.1f} seconds.\n')