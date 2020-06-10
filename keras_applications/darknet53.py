"""DarkNet53 model for Keras.

# Reference:

- https://pjreddie.com/darknet/

Adapted from code contributed by BigMoyan.
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# This is acting as a placeholder to allow us to specify local configs
# until we have a global way of doing this (i.e. weights hosted somewhere)
from decouple import config

import os
import warnings

from . import get_submodules_from_kwargs
from . import imagenet_utils
from .imagenet_utils import decode_predictions
from .imagenet_utils import _obtain_input_shape

preprocess_input = imagenet_utils.preprocess_input

# TODO: Update these with an external web address that is accessible by everyone
# WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
#                 'releases/download/v0.2/'
#                 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
#
# WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
#                        'releases/download/v0.2/'
#                        'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

# TODO: Fix as per previous TODO
WEIGHTS_PATH = WEIGHTS_PATH_NO_TOP = config("LOCAL_WT_FILE_PATH")

backend = None
layers = None
models = None
keras_utils = None


def conv_block(input_tensor, kernel_size, filters,
               stride=(1, 1), activation="leaky_relu",
               alpha=0.1, batch_norm=True, l2_decay=0.0005):
    """A convolutional block utilizing batch normalization and leaky relu

        CONV > BN > ACTIVATION

    Args:
        input_tensor (tf.Tensor): input tensor passed into
            the convolutional block
        kernel_size (tuple of ints): The kernel size used for the
            convolution layer
        filters (int): The number of filters used in the convolution layer
        stride (tuple of ints, optional): The stride sued for the
            convolutional layer
        activation (str, optional): The activation function used following
            batch normalization.
        alpha (float, optional): Alpha value for the optimizer function
        batch_norm (bool, optional): Whether or not to include a batch
            normalization layer following the convolutional layer
        l2_decay (float, optional): Value to be used for convolutional
            layer kernel regularization decay/momentum

    Returns:
        Output tensor for the block.
    """
    if stride == (1, 1):
        pad_style = "same"
    else:
        input_tensor = layers.ZeroPadding2D(((1, 0), (1, 0)))(input_tensor)
        pad_style = "valid"

    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=stride, padding=pad_style,
                      use_bias=not batch_norm,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_decay))(input_tensor)

    if batch_norm:
        x = layers.BatchNormalization()(x)

        if activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=alpha)(x)
        else:
            raise NotImplementedError("Anything besides "
                                      "LeakyReLU has not "
                                      "yet been implemented")

    return x


def res_block(input_tensor, filters, repetitions,
              kernel_size_1=(1, 1), kernel_size_2=(3, 3), ):
    """The residual block utilizes skip connections and convolutional layers

        INPUT > {{ [ CONV > CONV > ADD ] x desired-repetitions }}

    Args:
        input_tensor (tf.Tensor): input tensor passed into
            the residual block
        kernel_size_1 (tuple of ints): The kernel size used for the first
            convolution layer within each residual repetition (1x1)
        kernel_size_2 (tuple of ints): The kernel size used for the second
            convolution layer within each residual repetition (3x3)
        filters (int): The number of filters used in the first convolution layer
            within each residual repetition. The second convolution layer
            will have twice this number of filters
        repetitions (int): How many residual units (CONV>CONV>ADD) make up
            the complete residual block (1, 2, 4, or 8)

    Returns:
        Output tensor for the block.
    """
    x = input_tensor
    for i in range(repetitions):
        skip_x = x
        x = conv_block(skip_x, kernel_size_1, filters)
        x = conv_block(x, kernel_size_2, filters * 2)
        x = layers.add([x, skip_x])
    return x


def DarkNet53(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000, ):
    """Instantiates the DarkNet53 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        include_top (bool, optional): whether to include the
            fully-connected layer at the top of the network.
        weights (str, optional): one of ...
            `None` (random initialization),
            `imagenet` (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor (tf.Tensor): Tensorlflow/Keras tensor
            i.e. output of `layers.Input()` - to use as input for the model.
        input_shape (tuple, optional): Shape of input
            only to be specified if `include_top` is False
            otherwise the input shape has to be either:
                - `(224, 224, 3)` with `channels_last` data format
                -  `(3, 224, 224)` with `channels_first` data format
            It should have exactly 3 inputs channels,and width and height
            should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling (str, optional): The pooling mode for feature extraction
            when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError:
            - in case of invalid argument for `weights`
            - in case of invalid input shape.
    """

    # TODO - Understand this better and update if necessary
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # TODO - Integrate this _obtain_input_shape function properly
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    # Create input layer using input shape calculated previously or
    # pass the input_tensor directly as the input layer
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # TODO - This can probably be removed
    # Batch Normalization Axis
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # BLOCK 1 - CONV 1 - [416x416 --> 416x416]
    x = conv_block(img_input, filters=32, kernel_size=(3, 3), stride=(1, 1))

    # BLOCK 2 - CONV 2 - [416x416 --> 208x208]
    x = self.conv_block(x, filters=64, kernel_size=(3, 3), stride=(2, 2))

    # BLOCK 3 - RES 1 - [208x208 --> 208x208]
    x = self.res_block(x, filters=32, repetitions=1)

    # BLOCK 4 - CONV 3 - [208x208 --> 104x104]
    x = self.conv_block(x, filters=128, kernel_size=(3, 3), stride=(2, 2))

    # BLOCK 5 - RES 2 - [104x104 --> 104x104]
    x = self.res_block(x, filters=64, repetitions=2)

    # BLOCK 6 - CONV 4 - [104x104 --> 52x52]
    x = self.conv_block(x, filters=256, kernel_size=(3, 3), stride=(2, 2))

    # BLOCK 7 - RES 3 - [104x104 --> 104x104]
    x = self.res_block(x, filters=128, repetitions=8)

    # BLOCK 8 - CONV 5 - [104x104 --> 52x52]
    x = self.conv_block(x, filters=512, kernel_size=(3, 3), stride=(2, 2))

    # BLOCK 9 - RES 4 - [52x52 --> 52x52]
    x = self.res_block(x, filters=256, repetitions=8)

    # BLOCK 10 - CONV 6 - [52x52 --> 26x26]
    x = self.conv_block(x, filters=1024, kernel_size=(3, 3), stride=(2, 2))

    # BLOCK 11 - RES 5 - [26x26 --> 26x26]
    x = self.res_block(x, filters=512, repetitions=4)

    # TODO - Review this...
    #   - It could be done in a few different ways...
    #   - Ensure this is the best way
    if include_top:
        x = layers.AveragePooling2D()(x)
        x = layers.Conv2D(filters=classes,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding="same",
                          use_bias=True,
                          activation="linear",
                          name=self.get_name("conv2d"))(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Softmax()(x)

    # TODO: Do we want a final globalaveragepooling2d layer if no head is present?
    #   - Should we use the pooling option like they do in the ResNet50 architecture?
    #   - i.e.
    #         if pooling == 'avg':
    #             outputs = layers.GlobalAveragePooling2D()(x)
    #         elif pooling == 'max':
    #             outputs = layers.GlobalMaxPooling2D()(x)
    #         else:
    #             outputs = x
    else:
        outputs = layers.GlobalAveragePooling2D()(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = models.Model(inputs, outputs, name='darknet53')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'darknet53_weights.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'darknet53_weights_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)

        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)

    elif weights is not None:
        model.load_weights(weights)

    return model
