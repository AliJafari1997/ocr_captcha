import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from keras import layers
from typing import List, Optional, Tuple, Dict
import numpy as np
import os
import glob
from tensorflow import Tensor
import pandas as pd


epochs = 300                # shows number of epochs
batch_size:int = 16       # this number shows size of mini_batch

# Desired image dimensions
img_width:int = 200   # number of pixels tha shows width of images
img_height:int = 50   # number of pixels tha shows height of images

downsample_factor:int = 4   # shows the ratio that we want to downsample the images with that ratio


csv_path:str = "./files/data.csv"     # model metrics are saved in a file with this relative path
# you must use the command mkdir files to use these callbacks
my_callbacks:callbacks = [
        callbacks.CSVLogger(csv_path),              # callback for saving metrics and loss in csv
        callbacks.TensorBoard(),                  
        callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)               # callback for stopping whenever validation_loss doesn't decrease for 50 epochs
    ]  

def split_data(images: List[np.ndarray], labels:List[np.ndarray], train_size:float=0.9)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    splits Dataset into training, validation sections

    Args:
        images (List[np.ndarray]): _description_
        labels (List[np.ndarray]): _description_
        train_size (float, optional): _description_. Defaults to 0.9.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a tuple of x_train, x_valid, y_train, y_valid
    """    
    # Get the total size of the dataset
    size = int(len(images))
    # Get the size of training samples
    train_samples = int(size * train_size)
    # Split data into training and validation sets
    # choosing trainig images and training labels with the ratio of train_size
    x_train, y_train = images[:train_samples], labels[:train_samples]
    # choosing the rest of the images and labels as our validation images and validation labels
    x_valid, y_valid = images[train_samples:], labels[train_samples:]  
    return x_train, x_valid, y_train, y_valid


class CTCLayer(layers.Layer):
    """
    calculates custom loss function named Connectionist Temporal Classification

    """    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
    

def create_dir(path:str):
    """
    creates directory

    Args:
        path (str): string path of the directory that we want to generate it
    """  
    # if this directory doesn't exist, create that with defined string path, otherwise throw an Error   
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def plot_train_results():
    '''plot the training results'''
    csv = pd.read_csv(os.path.join(os.getcwd(), './files/data.csv'))
    plt.plot(csv['loss'], label='loss')
    plt.plot(csv['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('results of a simple fully connected network on mnist dataset')
    plt.legend()
    plt.savefig('loss_on _epochs.png')
    plt.show()

