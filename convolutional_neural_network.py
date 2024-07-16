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
import keras.layers


def data_arranger(data: pd.DataFrame, resolution: list) -> typing.Tuple[np.ndarray, typing.List]:
    '''
    Arg:
        Read in pandas table with header:
        x,y,(maybe z),<other parameters>,temperature_gradient,velocity_magnitude_gradient,z_velocity_gradient
        Also a list of resolution, corresponding of its original coordinate.
    Returns:
        array:
            Drop the coordinate and other params except for gradients of pandas table,
            normalize the data in range 0-1, and rearrange the data to an 4D/5D numpy array.
            The shape of the data is [1, *resolution, number_of_variables] (The first dimension indicates that there is only one batch in the file)
        headers:
            A list of strings of names of headers for the array above, corresponding headers to arrays,
            since numpy array doesn't have a header.
        indices:
            A numpy array in shape [*resolution], to store their original index of the table for each grid points, 
            so that the classification result could be correctly placed into their original position in the pandas table.
    '''
    # List of columns to be retained and normalized
    headers = [col for col in data.columns if col in ['temperature_gradient', 'velocity_magnitude_gradient', 'z_velocity_gradient']]
    
    # Normalize data
    normalized_data = data[headers].copy()
    print("Normalizing gradient data...")
    for col in headers:
        col_min = normalized_data[col].min(skipna=True)
        col_max = normalized_data[col].max(skipna=True)
        if col_max != col_min:  # Check to avoid division by zero
            normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
        else:
            normalized_data[col] = 0.0
    
    # Determine coordinate columns
    coord_cols = [col for col in data.columns if col in ['x', 'y', 'z']]
    
    # Create the array with correct shape
    array = np.full([1] + resolution + [len(headers)], np.nan)
    
    # Calculate increment of coordinates of each grid of numpy arrays
    if len(coord_cols) == 3:
        steps = np.array([
            (data['x'].max() - data['x'].min()) / resolution[0],
            (data['y'].max() - data['y'].min()) / resolution[1],
            (data['z'].max() - data['z'].min()) / resolution[2]
        ])
    else:  # sliced (2D)
        steps = np.array([
            (data['x'].max() - data['x'].min()) / resolution[0],
            (data['y'].max() - data['y'].min()) / resolution[1]
        ])
    
    # Calculate indices for each coordinate
    print('Converting the data to tensor...')
    ranges = np.array([data[col].agg([min, max]) for col in coord_cols])
    indices = np.floor((data[coord_cols].values - ranges[:, 0]) / steps).astype(int)
    
    # Clip indices to ensure they're within bounds
    indices = np.clip(indices, 0, np.array(resolution[:len(coord_cols)]) - 1)
    
    # Assign values to grid points of array
    for i, var in enumerate(headers):
        if len(coord_cols) == 2:
            array[0, indices[:, 0], indices[:, 1], i] = normalized_data[var].values
        elif len(coord_cols) == 3:
            array[0, indices[:, 0], indices[:, 1], indices[:, 2], i] = normalized_data[var].values
    
    print("Finished converting.")
    return array, headers


def loss_function(data: tf.Tensor, headers: list, classification: tf.Tensor) -> tf.Tensor:
    '''
    The function to calculate and tell the model how bad it performs prediction.

    Args: 
        data: A tensor. in shape [1, *resol, len(headers)]

        headers: a list storing the name of variables telling how variable temperature_gradient,
        velocity_magnitude_gradient, z_velocity_gradient are correlated the 1st index of the data.

        classification: a tensor, in shape [1, *resol, 1], the last index is storing classification results between 0-1, or NaN, for each grid points for this table.

    Returns: 
        loss: a SCALAR(single-value) tensor representing how bad a model predicts in this table. A point with
        high gradient and low classification value, or low gradient and high classification value will contribute
        to higher loss. The loss wil also be high if the classificaition is close to 0.5, to encourage certain classification results.
    '''
    # Convert data to float32 if it's not already
    data = tf.cast(data, tf.float32)
    classification_tensor = tf.cast(classification_tensor, tf.float32)

    # Extract the indices of the gradients from headers
    temp_grad_idx = headers.index('temperature_gradient')
    vel_mag_grad_idx = headers.index('velocity_magnitude_gradient')
    z_vel_grad_idx = headers.index('z_velocity_gradient')

    # Extract the gradient values from the data tensor
    if len(data.shape) == 3: # 2D
        temperature_gradient = data[:, :, :, temp_grad_idx]
        velocity_magnitude_gradient = data[:, :, :, vel_mag_grad_idx]
        z_velocity_gradient = data[:, :, :, z_vel_grad_idx]
        classification = classification_tensor[:, :, :, 0]
    else: # 3D
        temperature_gradient = data[:, :, :, :, temp_grad_idx]
        velocity_magnitude_gradient = data[:, :, :, :, vel_mag_grad_idx]
        z_velocity_gradient = data[:, :, :, :, z_vel_grad_idx]
        classification = classification_tensor[:, :, :, 0]

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


class CustomModel2D(keras.Model):
    '''
    The model of convolutional neural network. Highly customized.
    The input is a tensor with shape [1, x, y, len(headers)], the output is a tensor with shape [1, x, y, 1].
    For each variable, there will be first a convolutional neural network(CNN), 
    then a dense layer to take the value of concolutional layer of each variable as input.
    '''
    def __init__(self, headers: list, resolution: list):
        super(CustomModel2D, self).__init__()
        self.headers = headers
        self.resolution = resolution

        # Convolutional layers for each variable. 
        # The layer have 1 filter(number of kernel), [3,3] size kernel.
        # For points at the boundry, the 'same' paddling wikk keep the output having the same dimension with the input.
        self.conv_layers = [
            keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same', input_shape=(resolution[0], resolution[1], len(headers)))
            for _ in range(len(headers))
        ]

        # Dense layers
        self.dense1 = keras.layers.Dense(len(headers), activation='relu')
        self.dense2 = keras.layers.Dense(len(headers), activation='relu')
        self.dense3 = keras.layers.Dense(1, activation='sigmoid')

        # Reshape layer
        self.output_reshape = keras.layers.Reshape((resolution[0], resolution[1], 1))

    def call(self, inputs):
        # Apply CNN to each variable separately
        conv_outputs = []
        for conv in self.conv_layers:
        # Each conv layer processes all input channels and outputs a single channel
            conv_output = conv(inputs)
            conv_outputs.append(conv_output)
        # Concatenate the conv outputs along the channel dimension, along the first axis
        # (combines the outputs from all the individual convolutional layers into a single tensor)
        x = tf.concat(conv_outputs, axis=0) # Resulting shape:(batch_size, height, width, number_of_conv_layers)

        # Flatten each grid point's features separately, to the shape (resolution[0] * resolution[1], len(headers))
        x = keras.layers.Reshape((self.resolution[0] * self.resolution[1], -1))(x)

        # Apply dense layers to each grid point's features
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        # Reshape back to original grid shape with single channel
        return self.output_reshape(x)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            classification = self(data, training=True)
            loss = loss_function(data, self.headers, classification)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss} # will be seen during training!
    

def model_2D_create_compile(headers:list, learning_rate:float, resolution:list) -> CustomModel2D:
    '''
    Args: 
        headers: list of headers of params. Determines the structure of model.
        learning_rate: The size of the steps taken during optimization to reach the minimum of the loss function.
    Returns:
        model
    '''
    # Create, compile the model
    model = CustomModel2D(headers, resolution)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


class LossHistory(tf.keras.callbacks.Callback):
    '''
    Recording the loss for tach batches.
    '''
    # Called at the beginning of training.
    def on_train_begin(self, logs={}):
        self.losses = []

    # Called at the end of each batch.
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def model_2D_train(model:CustomModel2D,data:tf.Tensor, batch_size:int, epochs:int) -> typing.Tuple[CustomModel2D, LossHistory]:
    '''
    Args: 
        model: model to be trained
        data: arranged non-NaN datas.
        You need to try to get the optimal one.
        batch_size: number of lines for one training step.
        epoches: number of passes for the whole data.

    Returns:
        model: Trained model.
        history: The history of loss over batches.
    '''
    # Train the model, returning the loss over batches.
    loss_hist = LossHistory()
    history = model.fit(data, batch_size=batch_size, epochs=epochs, callbacks=[loss_hist])

    return model, history


def view_loss_history(history:LossHistory, path:str):
    '''
    Save the image of loss over batches.

    Args:
        history: The information to plot.
        path: The path to save the image.
    '''
    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.title('Loss  over each epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig(path)


def model_2D_classification(model: CustomModel2D, data: tf.Tensor, non_nan_indices: list, num_grid_data: int, table:pd.DataFrame) -> pd.DataFrame:
    '''
    Args:
        model: trained model.
        data: arranged data.
        table: The original table with temperature information.
    Returns:
        The original table with a column 'is_boundry' indicating how likely it is to be a boundry, and sign indicating its temperature.
    '''
    classification = model.predict(data)
    classification = classification.reshape(table.shape[0])  # Reshape to match table size
    
    table['is_boundary'] = classification

    # Normalize to range of 0-1
    table['is_boundary'] = (table['is_boundary'] - table['is_boundary'].min()) / (table['is_boundary'].max() - table['is_boundary'].min())
    
    # Adjust sign based on temperature
    table.loc[table['temperature'] < 0, 'is_boundary'] *= -1
    
    print('Finished classifying data.')
    return table


def CNN_3D_model(resolution: list, header: list) -> keras.models.Model:
    pass