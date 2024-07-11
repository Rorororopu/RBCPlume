'''
Read in pandas table with header:
x,y,(maybe z),temperature,temperature_gradient,velocity_magnitude_gradient,z_velocity_gradient

Using Dense and Convolutional Neural Network, attatch a column titled`is_boundry` to the original table in range [-1, 1].

The larger absolute value means the larger confidence that its corresponding grid point in the boundry of a heat plume.
If the value of `is_boundry` is positive, it is a hot plume boundry. If it is negative, it is a cold plume boundry.

'''
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf # deep learning library
import typing # Clearly indicating the type of data in the return tuple of a function

# automatically installed with tensorflow
# Don't use the 'from *** import ***', it will cause errors!
import keras.layers
import keras.models
import keras.optimizers
import keras.backend as K

import sklearn.model_selection

def data_arranger_NN(data: pd.DataFrame) -> typing.Tuple[np.ndarray, typing.List]:
    '''
    Arg:
        Read in pandas table with header:
        x,y,(maybe z),temperature,temperature_gradient,velocity_magnitude_gradient,z_velocity_gradient

    Returns:
        array:
            drop the coordinate and "temperature" column of pandas table, 
            normalize the data in range 0-1, and rearrange the data to an 2D numpy array.
            The first dimension records the index of data points, 
            the second dimension records values of each column at that grid point.

        header:
            A list of strings of names of headers for the array above, since numpy array doesn't have a header.
    '''
    # List of columns to be retained and normalized
    columns_to_normalize = [col for col in data.columns if col not in ['x', 'y', 'z', 'temperature']]
    
    # Normalize data
    for col in columns_to_normalize:
        col_min = data[col].min()
        col_max = data[col].max()
        if col_max != col_min:  # Check to avoid division by zero
            data[col] = (data[col] - col_min) / (col_max - col_min)
        else:
            data[col] = 0.0
    
    # Convert the normalized data to a 2D numpy array manually
    array = np.array([list(row) for row in data[columns_to_normalize].itertuples(index=False, name=None)])
    
    # Header for the numpy array
    header = columns_to_normalize

    return array, header


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


def loss_function_NN(data:tf.Tensor, header:list, classification:tf.Tensor) -> tf.Tensor:
    '''
    The function to calculate and tell the model how bad it performs prediction.

    Args: 
        data: a tensor(tensorflow will automatically convert numpy array to tensorflow array),
        recording the data of a batch. 1st index represents a data point, 2nd index represents
        the values of each column at that grid point

        header: a list storing the name of variables telling how variable temperature_gradient,
        velocity_magnitude_gradient, z_velocity_gradient are correlated the 2nd index of the data.

        classification: a 1D tensor, storing classification results between 0-1 for each grid points for this batch.

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
    temperature_gradient = data[:, temp_grad_idx]
    velocity_magnitude_gradient = data[:, vel_mag_grad_idx]
    z_velocity_gradient = data[:, z_vel_grad_idx]

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

    
class CustomMasking(keras.layers.Layer):
    '''
    Add a mask(Boolean mark) for points with NaN values.
    '''
    def __init__(self, mask_value=np.nan, **kwargs):
        super(CustomMasking, self).__init__(**kwargs)
        self.mask_value = mask_value

    def build(self, input_shape):
        super(CustomMasking, self).build(input_shape)

    def call(self, inputs):
        mask = K.not_equal(inputs, self.mask_value)
        return K.switch(mask, inputs, K.constant(np.nan, dtype=inputs.dtype))

    def compute_output_shape(self, input_shape):
        return input_shape
    

class NaNHandlingLayer(keras.layers.Layer):
    '''
    Let points with NaN values have NaN classification result.
    '''
    def call(self, inputs):
        non_nan_mask = tf.math.logical_not(tf.math.is_nan(inputs))
        outputs = tf.where(non_nan_mask, inputs, tf.constant(np.nan, dtype=inputs.dtype))
        return outputs


def NN_model(header: list) -> keras.models.Model:
    '''
    Function to create a Neural Network model.

    Args:
        header: A list of headers representing the parameters.

    Returns:
        Model: A Keras Model object.
    '''
    
    input_layer = keras.layers.Input(shape=(len(header),))
    masked = CustomMasking(mask_value=np.nan)(input_layer)# Exclude NaN values. They will be kept as NaN.
    masked = NaNHandlingLayer(masked) 
    x = keras.layers.Dense(len(header), activation='relu')(masked) # relu is a function: when x<0, y=0; when x>0, y=x.
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(len(header), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(len(header), activation='relu')(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x) # An s-curve whose output is between 0-1.
    
    model = keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=keras.optimizers.adam_v2, loss=loss_function_NN)

    return model


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


# Define a custom callback to log the loss
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def model_train(model:keras.models.Model, data:np.ndarray, batch_size:int=10000, epochs:int=10, test_size:float=0.2) -> tuple:
    '''
    From a compiled model, train the model with provided data. 
    The provided data will be split into training and validation set(To see how the model performs).

    Args:
        model: The compiled model.
        data: The arranged numpy array for training.
        batch_size: Number of samples for every training.
        epochs: Number of passes for each whole data set.
        test_size: The poportion of validation set over the whole data.

    Returns:
        model: The trained model
        loss_history: A History object for visualization
    '''
    train, val = sklearn.model_selection.train_test_split(data, test_size=test_size, random_state=42) # Use a 42(or any other number) as a seed so the result of split is always the same 
    loss_hist = LossHistory()
    model.fit(train,None,batch_size=batch_size,epochs=epochs,validation_data = (val,None), callbacks=[loss_hist])
    
    return model, loss_hist


def model_visualize(loss_hist: LossHistory, path:str):
    '''
    Output the graph of the performance (loss) over each batch.

    Args: 
        loss_hist: The record of loss over batches from the function model_train.
        path: The path to save the image.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(loss_hist.losses, label='Loss over batches')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.title('Loss Over Batches')
    plt.legend()
    plt.plt.savefig(path)


def model_classify(model:keras.models.Model, array:np.ndarray, data:pd.DataFrame) -> pd.DataFrame:
    '''
    Classify about whether a grid point is the boundry of a heat plume.

    Args:
        model: The trained model.
        array: Arranged data from the pandas table.
        data: The original mapped table to append the result
    
    Return:
        data: The result is stored in the column 'is_boundry', ranging from -1 to 1.
        If the value of `is_boundry` is positive, it is a hot plume boundry. If it is negative, it is a cold plume boundry.
    '''

    predicted_data = pd.DataFrame(model.predict(array))
    data['is_boundry'] = predicted_data
    
    if data['temperature'] < 0:
        data['is_boundry'] = (-1) * data['is_boundry']

    return data