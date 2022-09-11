import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import ndarray
from tensorflow import Tensor
import utils
from typing import Dict, Tuple, List, Optional, Union, Generator
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.models import Model
from model import build_model

def get_data()->Tuple[List[str], List[str], set, int]:  
    """
    finds the path of images, and corrsponding labels and collects all of them in two lists, besides that finds number of characters in all labels and
    max length of labels
    Returns:
        Tuple[List[str], List[str], set, int]: returns two lists containing path of images and labels, 
        and a set which contains characters in labels, and max length of labels
    """     
    # list of all images with png format in the defined path
    images = sorted(list(glob.glob(os.getcwd() + '/samples/*.png')))
    # list of all labels in the defined path
    labels = [img.split('/')[-1].split('.')[0] for img in images]
    # finds all unique characters in all labels
    characters = set(char for item in labels for char in item)
    # finds the max length of all labels
    max_length = max([len(label) for label in labels])
    return images, labels, characters, max_length


def get_test_data()->Tuple[List[str], List[str]]:
    """
    generates two lists containing path of test images and test labels
    Returns:
        Tuple[List[str], List[str]]: x_test is list which contains path test images, and y_test is list which contains path of labels
    """    
    # finding path of test images with glob library and putting all of them in a alist
    x_test = sorted(list(glob.glob(os.getcwd() + '/samples/*.jpg')))
    # finds labels of each image in test dataset
    y_test = [img.split('/')[-1].split('.')[0] for img in x_test]
    return x_test, y_test

def encode_single_sample(img_path:str, label:str)->Dict[str, np.ndarray]:
    """
    gets image_path and label, and resize the gray_scaled version of that image and returns the 
    trasposed of that preprocessed image, and letters of the label
    Args:
        img_path (str): shows the string_path of an image
        label (str): string label that shows the content of images

    Returns:
        Dict[str, np.ndarray]: preprocessed image and string version of its label
    """    
    # Read image from a string path
    img = tf.io.read_file(img_path)
    #Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize to the desired size
    img = tf.image.resize(img, [utils.img_height, utils.img_width])
    # ranspose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

def encode_single_sample_test(img_path:str, label:str)->Dict[str, np.ndarray]:
    """
    gets image_path and label, and resize the gray_scaled version of that image and returns the 
    trasposed of that preprocessed image, and letters of the label
    Args:
        img_path (str): shows the string_path of an image
        label (str): string label that shows the content of images

    Returns:
        Dict[str, np.ndarray]: preprocessed image and string version of its label
    """    
    # Read image from a string path
    img = tf.io.read_file(img_path)
    #Decode and convert to grayscale
    img = tf.io.decode_jpeg(img, channels=1)
    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize to the desired size
    img = tf.image.resize(img, [utils.img_height, utils.img_width])
    # ranspose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


def data_generator(x_train:np.ndarray, y_train:np.ndarray, x_valid:np.ndarray, y_valid:np.ndarray, x_test:np.ndarray, y_test:np.ndarray)->Tuple[Generator, Generator, Generator]:
    """
    generates train, validation, and test data generators
    Args:
        x_train (np.ndarray): an array of training images
        y_train (np.ndarray): an array which contains training labels
        x_valid (np.ndarray): an array which contains validation images
        y_valid (np.ndarray): an array which contains validation labels
        x_test (np.ndarray): an array which contains test images
        y_test (np.ndarray): an array which contains test labels

    Returns:
        Tuple[Generator, Generator, Generator]: _description_
    """    
    # performs preprocessing on training dataset and splits training dataset into batches
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(utils.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    # performs preprocessing on validation dataset and splits training dataset into batches
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(utils.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    # performs preprocessing on test dataset and splits training dataset into batches
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = (
        test_dataset.map(
            encode_single_sample_test, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(utils.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    return train_dataset, validation_dataset, test_dataset

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




def fit_model(model:Model):
    """
    this function is for fitting the defined model on trainin data, and finding loss value on both training and validation data

    Args:
        model (Model): 
    """    
    # Adam optimizer as our optimizer
    opt = keras.optimizers.Adam()
    # Compile the model with the defined optimizer
    model.compile(optimizer=opt)
    # Train the model on with traininn_images and training_labels
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=utils.epochs,
        callbacks=[utils.my_callbacks])

    # for test part, we set trainable=False for all layers, and 
    print('Test Step')
    for k,v in model._get_trainable_state().items():
        k.trainable = False

    # Don't forget to re-compile the model
    model.compile(optimizer=opt, lr=1e-4)
    model.fit(test_dataset,epochs=1)

    # contains information related to loss, val_loss, and learning_rate value
    #  history.history contains a dictionary with keys=[loss, val_loss, learning_rate]
    return history


# A utility function to decode the output of the network
def decode_batch_predictions(pred:Tensor)->List[str]:
    """
    Args:
        pred (Tensor): tensor of predictions for a batch of dataset

    Returns:
        List[str]: list of predicted labels in string format
    """    
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, we  can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    # Iterate over the results and get back the text
    output_text = []
    # converts tensors to string characters, and concatenates those characters
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    # returns a list which contains string characters of predicted labels in a batch
    return output_text

def show_prediction(prediction_model:Model)->None:
    """
    function for plotting a batch of test data and predicted labels for them

    Args:
        prediction_model (Model): functional model in tensorflow 
    """
    # takes 1 batch of defined data
    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]
        # predicts outputs of defined batch
        preds = prediction_model.predict(batch_images)
        # converts the tensor to a list of predicted characters
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
        
    plt.savefig('test_results.png')
    plt.show() 

if __name__ == "__main__":
    utils.create_dir('./files')   # creates files directory
    images, labels, characters, max_length = get_data()
    x_test, y_test = get_test_data()
    # Mapping original characters to integers
    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    # splits data into x_train, x_valid, y_train, y_valid
    x_train, x_valid, y_train, y_valid = split_data(images, labels)

    # creates data generators train_dataset, validation_dataset, test_dataset
    train_dataset, validation_dataset, test_dataset = data_generator(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # builds the model
    model = build_model(char_to_num)
    history = fit_model(model)

    # gets the model for plotting predicted labels
    prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)

    # plots the predicted labels on test data
    show_prediction(prediction_model)

    # plots the loss on validation/train sections
    utils.plot_train_results()
