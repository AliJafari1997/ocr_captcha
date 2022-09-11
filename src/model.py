import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers
import utils
from typing import Optional
from keras.models import Model
from tensorflow import Tensor

def relu_bn(inputs: Tensor) -> Tensor:
    """
    applies ReLU activation function and BatchNormalization sequentially to the inputs

    Args:
        inputs (Tensor): input tensor that we want to apply ReLU Activation Function, and Batchnormalization layer to it

    Returns:
        Tensor: Tensor after applying ReLU Activation Function, and Batchnormalization layer to the input tensor
    """    
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: Optional[int] = 3) -> Tensor:
    """
    this block consists of 2 convolutional layers in the main path with defined filter_size, and another convolutional layer with filter_size=1 in skip_connection path,
    which facilitates the flow of gradient, leading to prevention of gradient vanishing problems in deep neural networks. finally we add the results from these two pathes.

    Args:
        x (Tensor): input Tensor to the residual block
        downsample (bool): if downsample=True, that means the stride size in first convolutions layer will be 2, otherwise it will be 1
        filters (int): shows number of filters that we want to apply
        kernel_size Optional[int] shows size of kernel. Defaults to 3.

    Returns:
        Tensor: output of the residual block after applying convolution, batchnormalization, and activation layers to the initial featuremap
    """    
    # first convolution layer that we want to apply in residual_block. if downsample=True, that means strides will be 2 in this layer, otherwise strides will be 1
    y = layers.Conv2D(kernel_size=kernel_size,
               strides= 1,
               filters=filters, padding='same')(x)
    # after each convolution layer, we apply batchnormalization layer, and activation function in this residual block
    y = relu_bn(y)
    # second convolutional layer in the main path with strides=1
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters, padding='same')(y)

    # stride of this convolutional layer in the skip connection path will be 2 if downsample=True, otherwise strides will be 1
    x = layers.Conv2D(kernel_size=1,
                   strides=1,
                   filters=filters)(x)
    # finally we add the result from the main path with the output of the skip connection path
    out = layers.Add()([x, y])
    # finally perform relu activation function, and batchnormalizatin layer to the output of the former featuremap
    out = relu_bn(out)
    if downsample:
        out = layers.MaxPooling2D((2, 2))(out)

    return out



def build_model(char_to_num:Tensor):

    """

    Args:
        char_to_num (Tensor): Tensor of integer indices

    Returns:
        Model: functional model which consists of CNN and RNN layers
    """    
    # Inputs to the model
    input_img = layers.Input(
        shape=(utils.img_width, utils.img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = residual_block(input_img, downsample=True, filters=32, kernel_size=3)
    print(x.shape)
    x = residual_block(x, downsample=False, filters=32, kernel_size=3)
    print(x.shape)

    x = residual_block(x, downsample=True, filters=64, kernel_size=3)
    print(x.shape)

    x = residual_block(x, downsample=False, filters=64, kernel_size=3)
    print(x.shape)
    # First conv block
          

    # We have used two residual blocks with and strides = 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((utils.img_width // 4), (utils.img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = utils.CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
    return model
