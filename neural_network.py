'''
Input: 
Pandas table with coordinates and gradients, and original data.

Output: 
A column, indicating how likely this grid point is the boundry of heat plume.
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


def data_arranger(data: pd.DataFrame) -> typing.Tuple[tf.Tensor, typing.List, typing.List, int]:
    '''
    Arg:
        Read in pandas table with header:
        x,y,(maybe z),temperature,temperature_gradient,velocity_magnitude_gradient,z_velocity_gradient

    Returns:
        array:
            Drop the coordinate and "temperature" column of pandas table, 
            normalize the data in range 0-1, and rearrange the data to an 2D numpy array.
            
            The first dimension records the index of data points, 
            the second dimension records values of each column at that grid point.

            Points with NaN values will also be dropped.

        header:
            A list of strings of names of headers for the array above, since numpy array doesn't have a header.
        
        non_nan_indices:
            A list of indices of all non_nan_points, so that when the model finishs predicting non NaN points, 
            the results can be placed to their original order and fill in the result for NaN points.

        len(data):
            Number of rows of the original data. With this and non_nan_indices, the program will know where
            is the points of NaN values.

    The reason why outputing non NaN indices in this function is because currently I con't devise a function
    to properly deal with NaNs in the model, so I have to remove these points before data are inputed to the model.
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
    original_array = np.array([list(row) for row in data[columns_to_normalize].itertuples(index=False, name=None)])
    
    # Header for the numpy array
    header = columns_to_normalize

    # Get the indices of non-nan values
    non_nan_indices_raw = np.argwhere(~np.isnan(original_array)) # corrently it is an 2D
    non_nan_indices = np.unique(non_nan_indices_raw[:, 0])

    # Get the non-nan values
    non_nan_values = original_array[~np.isnan(original_array)]

    # Combine indices and values
    result = np.column_stack((non_nan_indices, non_nan_values))

    # Convert result to Tensorflow tensor
    result = tf.convert_to_tensor(result, dtype='float32')

    return result, header, non_nan_indices, len(data)


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
    # Convert data to float32 if it's not already
    data = tf.cast(data, tf.float32)
    
    # Extract the indices of the gradients from the header
    temp_grad_idx = header.index('temperature_gradient')
    vel_mag_grad_idx = header.index('velocity_magnitude_gradient')
    z_vel_grad_idx = header.index('z_velocity_gradient')
    
    # Extract the gradient values from the data tensor
    temperature_gradient = data[:, temp_grad_idx]
    velocity_magnitude_gradient = data[:, vel_mag_grad_idx]
    z_velocity_gradient = data[:, z_vel_grad_idx]
    
    # Calculate the primary gradient loss
    gradient_avg = (temperature_gradient + velocity_magnitude_gradient + z_velocity_gradient)/3
    loss_high_class_low_grad = classification * (1 - gradient_avg)
    loss_low_class_high_grad = (1 - classification) * gradient_avg
    primary_loss = tf.reduce_mean(loss_high_class_low_grad + loss_low_class_high_grad)
    
    # Add regularization loss to encourage certain properties in classification
    regularization_loss = tf.reduce_mean(tf.square(classification - 0.5))
    
    # Total loss. 0.1 is added to avoid the loss approach to 0.693(ln2), which doesn't sounds good.
    loss = primary_loss + 0.1 * regularization_loss
    return loss


class CustomModel(keras.Model):
    '''
    The model of neural network. Highly customed.
    '''

    # Called everytime a new instance is created.
    def __init__(self:keras.Model, header:list):
        super(CustomModel, self).__init__()
        self.header = header
        self.dense1 = keras.layers.Dense(len(header), activation='relu')
        self.bn1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(len(header), activation='relu')
        self.bn2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(len(header), activation='relu')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    # Called during the forward pass of the model, e.g. fitting, predicting, evaluating.
    def call(self:keras.Model, inputs:np.ndarray):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dense3(x)
        return self.output_layer(x)

    # It defines the logic for a single training step. Called for each batch of data during model.fit().
    # It's not called during inference (model.predict) or evaluation (model.evaluate).
    def train_step(self, data):
        with tf.GradientTape() as tape:
            classification = self(data, training=True)
            loss = loss_function_NN(data, self.header, classification)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}# will be seen during training!


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


def model_create_compile_train(data:tf.Tensor, header:list, learning_rate:float, batch_size:int, epochs:int) -> typing.Tuple[CustomModel, LossHistory]:
    '''
    Args: 
        data: arranged non-NaN datas.
        header: list of headers of params. Determines the structure of model.
        learning_rate: The size of the steps taken during optimization to reach the minimum of the loss function.
        You need to try to get the optimal one.
        batch_size: number of lines for one training step.
        epoches: number of passes for the whole data.

    Returns:
        model: Trained model.
        history: The history of loss over batches.
    '''
    # Create, compile the model
    model = CustomModel(header)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate))

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


def model_classification(model:CustomModel, data:tf.Tensor, non_nan_indices:list, num_grid_data:int) -> np.ndarray:
    '''
    Args: 
        model: trained model.
        data: arranged data.
        non_nan_indices: List of indices of points with non_nan_value.
        num_grid_data: Number of grid points for the whole data, to plug in nan to grid points with nan value.

    Returns:
        A 1D numpy array of classification result. NaN value is included.
    '''
    classification = model.predict(data)

    result = np.full(num_grid_data, np.nan) # create a full list with nan values.
    result[non_nan_indices] = classification
    
    return result


