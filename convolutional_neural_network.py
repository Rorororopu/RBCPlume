import pandas as pd
import typing
import numpy as np
import tensorflow as tf

import keras.layers


def data_arranger_CNN(data:pd.DataFrame, resolution:list) -> typing.Tuple[np.ndarray, typing.List]:
    '''
    Arg:
        Read in pandas table with header:
        x,y,(maybe z),temperature,temperature_gradient,velocity_magnitude_gradient,z_velocity_gradient

        Also a list of resolution, corresponding of its original coordinate.

    Returns:
        array:
            drop the coordinate and "temperature" column of pandas table, 
            normalize the data in range 0-1, and rearrange the data to an 3D/4D numpy array:
            For each columns, there will be a 2D/3D array recording the position of each grid points and their value of each columns,
            ans these arrays are stored in a big array.
            i.e. first index determines which header you are referencing, the 2nd, 3rd, 4th index represents x, y, z coordinates respectively.

        header:
            A list of strings of names of headers for the array above, corresponding headers to arrays,
            since numpy array doesn't have a header.
    '''
    for col in data.columns: # Normalize data
        if col not in ['x', 'y', 'z', 'temperature']:
            col_min = data[col].min()
            col_max = data[col].max()
            if col_max != col_min:  # Check to avoid division by zero
                data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                data[col] = 0.0

    # Calculate increment of coordinates of each grid of numpy arrays
    coord_cols = [col for col in ['x', 'y', 'z'] if col in data.columns]
    if coord_cols == ['x', 'y', 'z']: 
        xstep = (data['x'].max() - data['x'].min())/resolution[0]
        ystep = (data['y'].max() - data['y'].min())/resolution[1]
        zstep = (data['z'].max() - data['z'].min())/resolution[2]
        headers = [col for col in data.columns if col not in ['x', 'y', 'z', 'temperature']]
        array = np.full((len(headers), resolution[0], resolution[1], resolution[2]), np.nan) # Generate the numpy array filled with NaN
    else: # sliced, coord_cols == ['x', 'y']
        xstep = (data['x'].max() - data['x'].min())/resolution[0]
        ystep = (data['y'].max() - data['y'].min())/resolution[1]
        headers = [col for col in data.columns if col not in ['x', 'y', 'z', 'temperature']]
        array = np.full((len(headers), resolution[0], resolution[1]), np.nan)
    
     # Populate the array with data
    for index, row in data.dropna(subset=['temperature']).iterrows():
        # Calculate grid indices of each row
        indices = []
        if 'x' in coord_cols:
            x_index = int((row['x'] - data['x'].min()) / xstep)
            indices.append(x_index)
        if 'y' in coord_cols:
            y_index = int((row['y'] - data['y'].min()) / ystep)
            indices.append(y_index)
        if 'z' in coord_cols:
            z_index = int((row['z'] - data['z'].min()) / zstep)
            indices.append(z_index)
        
        # Assign the values to the array
        for i, var in enumerate(headers):
            try:
                if len(indices) == 2:
                    array[i, indices[0], indices[1]] = row[var]
                elif len(indices) == 3:
                    array[i, indices[0], indices[1], indices[2]] = row[var]
            except IndexError:  
                pass  # Handle cases where the calculated index is out of bounds

    return array, headers


def loss_function_CNN(data: tf.Tensor, header: list, classification: tf.Tensor) -> tf.Tensor:
    '''
    The function to calculate and tell the model how bad it performs prediction.

    Args: 
        data: a tensor (tensorflow will automatically convert numpy array to tensorflow array),
        recording the data of a batch. 1st index determines which header you are referencing, 
        the 2nd, 3rd, 4th index represents x, y, z coordinates respectively.

        header: a list storing the name of variables telling how variable temperature_gradient,
        velocity_magnitude_gradient, z_velocity_gradient are correlated the 1st index of the data.

        classification: a 2D/3D tensor, storing classification results between 0-1 for each grid points for this batch.

    Returns: 
        loss: a SCALAR(single-value) tensor representing how bad a model predicts in a batch. A point with
        high gradient and low classification value, or low gradient and high classification value will contribute
        to higher loss. The loss wil also be high if the classificaition is close to 0.5, to encourage certain classification results.
    '''

    # Extract the indices of the gradients from the header
    temp_grad_idx = header.index('temperature_gradient')
    vel_mag_grad_idx = header.index('velocity_magnitude_gradient')
    z_vel_grad_idx = header.index('z_velocity_gradient')

    # Extract the gradient values from the data tensor
    if len(data.shape) == 3: # 2D
        temperature_gradient = data[temp_grad_idx, :, :]
        velocity_magnitude_gradient = data[vel_mag_grad_idx, :, :]
        z_velocity_gradient = data[z_vel_grad_idx, :, :]
    else: # 3D
        temperature_gradient = data[temp_grad_idx, :, :, :]
        velocity_magnitude_gradient = data[vel_mag_grad_idx, :, :, :]
        z_velocity_gradient = data[z_vel_grad_idx, :, :, :]

    # Calculate the primary gradient loss
    gradient_sum = temperature_gradient + velocity_magnitude_gradient + z_velocity_gradient
    loss_high_class_low_grad = classification * (1 - gradient_sum)
    loss_low_class_high_grad = (1 - classification) * gradient_sum

    primary_loss = tf.reduce_mean(loss_high_class_low_grad + loss_low_class_high_grad)
    
    # Add regularization loss to encourage certain properties in classification
    regularization_loss = tf.reduce_mean(tf.square(classification - 0.5))
    
    # Total loss
    loss = primary_loss + regularization_loss

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


def CNN_2D_model(resolution: list, header: list) -> keras.models.Model:
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
    model.compile(optimizer=keras.optimizers.adam_v2, loss=loss_function_CNN)
    
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
