'''
Input: 
Pandas table with coordinates and gradients, and original data.

Output: 
A column attatched to the original table, indicating how likely this grid point is the boundry of heat plume.
Its range is [-1, 1], or np.nan(for points with np.nan gradient)
Its magnitude indicates its liklihood of being a heat plume.
When it is positive, it indicates a hot plume;
when it is negative, it indicates a cold plume.
'''


import pandas as pd
import typing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras # Installed automatically with tensorflow


def data_arranger(data: pd.DataFrame, resolution: list) -> typing.Tuple[np.ndarray, typing.List]:
    '''
    Arg:
        Read in pandas table with header:
        x,y,(maybe z),<other parameters>,temperature_gradient,velocity_magnitude_gradient,z_velocity_gradient

        Also a list of resolution, corresponding of its original coordinate.

    Returns:
        array:
            Drop the coordinate and other params except for gradients of pandas table, 
            normalize the data in range 0-1, and rearrange the data to an 3D/4D numpy array:
            For each columns, there will be a 2D/3D array recording the position of each grid points and their value of each columns,
            ans these arrays are stored in a big array.
            i.e. first index determines which header you are referencing, the 2nd, 3rd, 4th index represents x, y, z coordinates respectively.

        headers:
            A list of strings of names of headers for the array above, corresponding headers to arrays,
            since numpy array doesn't have a header.
    '''
    # List of columns to be retained and normalized
    headers = [col for col in data.columns if col in ['temperature_gradient', 'velocity_magnitude_gradient', 'z_velocity_gradient']]
    # Normalize data
    normalized_data = data[headers].copy()
    print("Normalizing gradient datas...")
    for col in headers:
        col_min = normalized_data[col].min(skipna=True)
        col_max = normalized_data[col].max(skipna=True)
        if col_max != col_min:  # Check to avoid division by zero
            normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
        else:
            normalized_data[col] = 0.0

    # Determine coordinate columns
    coord_cols = [col for col in data.columns if col in ['x', 'y', 'z']]
    # Create the array
    array = np.full([len(headers)] + resolution, np.nan)
    
    # Calculate increment of coordinates of each grid of numpy arrays
    if coord_cols == ['x', 'y', 'z']:
        steps = np.array([
            (data['x'].max() - data['x'].min()) / resolution[0],
            (data['y'].max() - data['y'].min()) / resolution[1],
            (data['z'].max() - data['z'].min()) / resolution[2]
        ])
    else:  # sliced
        steps = np.array([
            (data['x'].max() - data['x'].min()) / resolution[0],
            (data['y'].max() - data['y'].min()) / resolution[1]
        ])

    # Calculate indices for each coordinate
    print('Converting the data to tensor...')
    ranges = np.array([data[col].agg([min, max]) for col in coord_cols]) # agg could do more calculation ad same time
    indices = np.floor((data[coord_cols].values - ranges[:, 0]) / steps).astype(int)
    # Clip indices to ensure they're within bounds
    indices = np.clip(indices, 0, np.array(resolution[:len(coord_cols)]) - 1)

    # Assign values to grid points of array
    for i, var in enumerate(headers):
        if len(coord_cols) == 2:
            # Use advanced indexing to assign values
            array[i][tuple(indices.T)] = normalized_data[var].values
        elif len(coord_cols) == 3:
            # Use advanced indexing to assign values
            array[i][tuple(indices.T)] = normalized_data[var].values
    print("Finished converting.")
    return array, headers


def loss_function(data: tf.Tensor, headers: list, classification: tf.Tensor) -> tf.Tensor:
    '''
    The function to calculate and tell the model how bad it performs prediction.

    Args: 
        data: A tensor recording the data of a table. 1st index determines which header you are referencing, 
        the 2nd, 3rd, 4th index represents x, y, z coordinates respectively.

        header: a list storing the name of variables telling how variable temperature_gradient,
        velocity_magnitude_gradient, z_velocity_gradient are correlated the 1st index of the data.

        classification: a 2D/3D tensor, storing classification results between 0-1, or NaN, for each grid points for this table.

    Returns: 
        loss: a SCALAR(single-value) tensor representing how bad a model predicts in this table. A point with
        high gradient and low classification value, or low gradient and high classification value will contribute
        to higher loss. The loss wil also be high if the classificaition is close to 0.5, to encourage certain classification results.
    '''
    # Convert data to float32 if it's not already
    data = tf.cast(data, tf.float32)
    classification = tf.cast(classification, tf.float32)

    # Extract the indices of the gradients from the header
    temp_grad_idx = headers.index('temperature_gradient')
    vel_mag_grad_idx = headers.index('velocity_magnitude_gradient')
    z_vel_grad_idx = headers.index('z_velocity_gradient')

    # Extract the gradient values from the data tensor
    if len(data.shape) == 3: # 2D
        temperature_gradient = data[temp_grad_idx, :, :]
        velocity_magnitude_gradient = data[vel_mag_grad_idx, :, :]
        z_velocity_gradient = data[z_vel_grad_idx, :, :]
    else: # 3D
        temperature_gradient = data[temp_grad_idx, :, :, :]
        velocity_magnitude_gradient = data[vel_mag_grad_idx, :, :, :]
        z_velocity_gradient = data[z_vel_grad_idx, :, :, :]

    # Calculate the primary gradient loss. 
    # If the dimension of these expressions are wrong(e.g. you miscalculated the geometric average of gradient), 
    # or didn't write the expression according to the scale of the param(e.g., whether it is ranging form [0,1] or [-1,1]),
    # The model will behave very strangely.
    gradient_avg = (temperature_gradient * velocity_magnitude_gradient * z_velocity_gradient) ** (1/3)
    loss_high_class_low_grad = classification * (1 - gradient_avg)
    loss_low_class_high_grad = (1 - classification) * gradient_avg

    def reduce_mean_ignore_nan(x, axis=None): # ignore nan points for the loss
        mask = tf.math.is_finite(x)
        x_masked = tf.where(mask, x, tf.zeros_like(x))
        sum_masked = tf.reduce_sum(x_masked, axis=axis, keepdims=True)
        count = tf.reduce_sum(tf.cast(mask, tf.float32), axis=axis, keepdims=True)
        return tf.math.divide_no_nan(sum_masked, count)
    
    primary_loss = reduce_mean_ignore_nan(loss_high_class_low_grad + loss_low_class_high_grad)

    # Add regularization loss to encourage certain properties in classification
    regularization_loss = reduce_mean_ignore_nan(tf.square(classification - 0.5))
    
    # Total loss. 0.1 is added to avoid the loss approach to 0.693(ln2), which doesn't sounds good.
    loss = primary_loss + 0.1 * regularization_loss

    return loss


class CustomPadding2D(keras.layers.Layer):
    '''
    Let the boarder grid points have NaN value convolution layer output, because at the boarder 
    there is not enough points to do convolution, and I want to keep the same shape with the original data and output.
    '''
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = padding # The size of padding on the boarder
        super(CustomPadding2D, self).__init__(**kwargs)

    def call(self, inputs):
        
        # Create NaN padding
        padded_input = tf.pad(inputs, 
                              paddings=[[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]],
                              constant_values=np.nan)
        return padded_input


def model_2D(resolution: list, header: list) -> keras.models.Model:
    '''
    The function to store the Convolutional Neural Network 2D model.
    
    Parameters:
        resolution: The shape of the input data for each parameter [res1, res2].
        header: A list of headers representing the parameters.
    
    Returns:
        Model: A Keras Model object.
    '''
    
    inputs = [keras.layers.Input(shape=(resolution[0], resolution[1], len(header))) for _ in range(len(header))]
    masked = CustomMasking(mask_value=np.nan)(inputs)
    masked = NaNHandlingLayer(masked)
    
    conv_layers = []
    
    for input_layer in masked:
        padded = CustomPadding2D(padding=(1, 1))(input_layer)
        conv = keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(padded) # 3 filters of size (3,3).
        conv = keras.layers.BatchNormalization()(conv)
        conv_layers.append(conv)
    
    # Concatenate the conv layers along the channel dimension
    merged_conv = keras.layers.concatenate(conv_layers, axis=-1)
    
    # Flatten each grid point's features separately
    flattened = keras.layers.Flatten()(merged_conv)
    
    # Reshape to (grid_points, features)
    reshaped = keras.layers.Reshape((resolution[0] * resolution[1], len(header) * 3))(flattened)
    
    # Apply dense layers to each grid point's features
    dense = keras.layers.Dense(3, activation='relu')(reshaped)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dense(1, activation='sigmoid')(dense)
    
    # Reshape back to original grid shape with single channel
    outputs = keras.layers.Reshape((resolution[0], resolution[1], 1))(dense)
    
    model = keras.layers.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.adam_v2, loss=loss_function)
    
    return model


class CustomPadding3D(keras.layers.Layer):
    '''
    Let the border grid points have NaN value convolution layer output, 
    because at the border there are not enough points to do convolution, 
    and I want to keep the same shape with the original data and output.
    '''
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = padding
        super(CustomPadding3D, self).__init__(**kwargs)

    def call(self, inputs):
        # Create NaN padding
        padded_input = tf.pad(inputs, 
                              paddings=[[0, 0], 
                                        [self.padding[0], self.padding[0]], 
                                        [self.padding[1], self.padding[1]], 
                                        [self.padding[2], self.padding[2]], 
                                        [0, 0]],
                              constant_values=np.nan)
        return padded_input


def CNN_3D_model(resolution: list, header: list) -> keras.models.Model:
    '''
    The function to store the Convolutional Neural Network 3D model.
    
    Parameters:
        resolution: The shape of the input data for each parameter [res1, res2].
        header: A list of headers representing the parameters.
    
    Returns:
        Model: A Keras Model object.
    '''
    
    inputs = [Input(shape=(resolution[0], resolution[1], resolution[2], len(header))) for _ in range(len(header))]
    masked = CustomMasking(mask_value=np.nan)(inputs)
    
    conv_layers = []
    
    for input_layer in masked:
        padded = CustomPadding2D(padding=(1, 1, 1))(input_layer)
        conv = Conv3D(3, (3, 3, 3), activation='relu', padding='same')(padded) # 3 filters of size (3,3).
        conv = BatchNormalization()(conv)
        conv_layers.append(conv)
    
    # Concatenate the conv layers along the channel dimension
    merged_conv = concatenate(conv_layers, axis=-1)
    
    # Flatten each grid point's features separately
    flattened = Flatten()(merged_conv)
    
    # Reshape to (grid_points, features)
    reshaped = Reshape((resolution[0] * resolution[1] * resolution[2], len(header) * 3))(flattened)
    
    # Apply dense layers to each grid point's features
    dense = Dense(3, activation='relu')(reshaped)
    dense = BatchNormalization()(dense)
    dense = Dense(1, activation='sigmoid')(dense)
    
    # Reshape back to original grid shape with single channel
    outputs = Reshape((resolution[0], resolution[1], resolution[2], 1))(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss=loss_function_CNN)
    
    return model
