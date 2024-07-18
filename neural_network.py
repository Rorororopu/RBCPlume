'''
Input: 
Pandas table with coordinates and gradients, and original data.

Output: 
A column attatched to the original table, indicating how likely this grid point is the boundry of heat plume.
Its range is [-1, 1], or np.nan(for points with np.nan gradient).
Its magnitude indicates its likelihood of being a heat plume.
When it is positive, it indicates the boundry of a hot plume;
when it is negative, it indicates the boundry of a cold plume.
'''


import pandas as pd
import typing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras # Installed automatically with tensorflow
import keras.layers
import keras.optimizers

def data_arranger(data: pd.DataFrame) -> typing.Tuple[tf.Tensor, typing.List, typing.List, int]:
    '''
    Arg:
        Read in pandas table with header:
        x,y,(maybe z),<other parameters>,temperature_gradient,velocity_magnitude_gradient,z_velocity_gradient
    Returns:
        array:
            Drop the coordinate and columns for other parameters of pandas table,
            normalize the data in range 0-1, and rearrange the data to an 2D numpy array.
            The first dimension records the index of data points,
            the second dimension records values of each column at that grid point.
            Points with NaN values will also be dropped.
        headers:
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
    
    # Convert the normalized data to a 2D numpy array
    print("Rearranging data to tensor for model classification...")
    original_array = normalized_data.to_numpy()
    
    # Get the indices of non-nan values
    non_nan_mask = ~np.isnan(original_array).any(axis=1)
    non_nan_indices = np.where(non_nan_mask)[0].tolist()
    
    # Get the non-nan values
    non_nan_values = original_array[non_nan_mask]
    print("Obtained regularized tensor.")
    
    return non_nan_values, headers, non_nan_indices, len(data)


class CustomModel(keras.Model):
    '''
    The model of neural network. Highly customed.
    '''

    # Called everytime a new instance is created.
    def __init__(self:keras.Model, headers:list):
        super(CustomModel, self).__init__()
        self.headers = headers
        self.dense1 = keras.layers.Dense(len(headers), activation='relu')
        self.dense2 = keras.layers.Dense(len(headers), activation='relu')
        self.dense3 = keras.layers.Dense(len(headers), activation='relu')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    # Called during the forward pass of the model, e.g. fitting, predicting, evaluating.
    def call(self:keras.Model, inputs:np.ndarray):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)
    
    def loss_function(data:tf.Tensor, headers:list, classification:tf.Tensor) -> tf.Tensor:
        '''
        The function to calculate and tell the model how bad it performs prediction.

        Args: 
            data: a tensor recording the data of a batch. 1st index represents a data point, 2nd index represents
            the values of each column at that grid point

            headers: a list storing the name of variables telling how variable temperature_gradient,
            velocity_magnitude_gradient, z_velocity_gradient are correlated the 2nd index of the data.

            classification: a 1D tensor, storing classification results between 0-1 for each grid points for this batch.

        Returns: 
            loss: a SCALAR(single-value) tensor representing how bad a model predicts in a batch. A point with
            high gradient and low classification value, or low gradient and high classification value will contribute
            to higher loss. The loss wil also be high if the classificaition is close to 0.5, to encourage certain classification results.
        '''
        # Convert data to float32 if it's not already
        data = tf.cast(data, tf.float32)
        
        # Extract the indices of the gradients from headers
        temp_grad_idx = headers.index('temperature_gradient')
        vel_mag_grad_idx = headers.index('velocity_magnitude_gradient')
        z_vel_grad_idx = headers.index('z_velocity_gradient')
        
        # Extract the gradient values from the data tensor
        temperature_gradient = data[:, temp_grad_idx]
        velocity_magnitude_gradient = data[:, vel_mag_grad_idx]
        z_velocity_gradient = data[:, z_vel_grad_idx]
        
        # Calculate the primary gradient loss. 
        # If the dimension of these expressions are wrong(e.g. you miscalculated the geometric average of gradient), 
        # or didn't write the expression according to the scale of the param(e.g., whether it is ranging form [0,1] or [-1,1]),
        # The model will behave very strangely.
        gradient_avg = (temperature_gradient * velocity_magnitude_gradient * z_velocity_gradient) ** (1/3)
        loss_high_class_low_grad = classification * (1 - gradient_avg)
        loss_low_class_high_grad = (1 - classification) * gradient_avg
        primary_loss = tf.reduce_mean(loss_high_class_low_grad + loss_low_class_high_grad)
        
        # Add regularization loss to encourage certain properties in classification
        regularization_loss = tf.reduce_mean(tf.square(classification - 0.5))
        
        # Total loss. 0.1 is added to avoid the loss approach to 0.693(ln2), which doesn't sounds good.
        loss = primary_loss + 0.1 * regularization_loss
        return loss

    # It defines the logic for a single training step. Called for each batch of data during model.fit().
    # It's not called during inference (model.predict) or evaluation (model.evaluate).
    def train_step(self, data):
        with tf.GradientTape() as tape:
            classification = self(data, training=True)
            loss = self.loss_function(data, self.headers, classification)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss} # will be seen during training!


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


def model_create_compile(headers:list, learning_rate:float) -> CustomModel:
    '''
    Args: 
        headers: list of headers of params. Determines the structure of model.
        learning_rate: The size of the steps taken during optimization to reach the minimum of the loss function.
    Returns:
        model
    '''
    # Create, compile the model
    model = CustomModel(headers)
    model.compile(optimizer = keras.optimizers.adam_v2(learning_rate=learning_rate))
    return model


def model_train(model:CustomModel,data:tf.Tensor, batch_size:int, epochs:int) -> typing.Tuple[CustomModel, LossHistory]:
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


def model_classification(model: CustomModel, data: tf.Tensor, non_nan_indices: list, num_grid_data: int, table:pd.DataFrame) -> pd.DataFrame:
    '''
    Args:
        model: trained model.
        data: arranged data.
        non_nan_indices: List of indices of points with non_nan_value. To put the classification result to their original position.
        num_grid_data: Number of grid points for the whole data, to plug in NaN to grid points with NaN value.
        table: The original table with temperature information.
    Returns:
        The original table with a column 'is_boundry' indicating how likely it is to be a boundry, and sign indicating its temperature.
    '''
    classification = np.array(model.predict(data)).flatten()
    result = np.full(num_grid_data, np.nan)  # create a full array with NaN values.
    
    # Assign the classification results to the correct indices in the result array
    for i, index in enumerate(non_nan_indices):
        result[index] = classification[i]
    
    table['is_boundary'] = result
    
    # Normalize to range of 0-1
    table['is_boundary'] = (table['is_boundary'] - table['is_boundary'].min()) / (table['is_boundary'].max() - table['is_boundary'].min())
    
    # Adjust sign based on temperature
    table.loc[table['temperature'] < 0, 'is_boundary'] *= -1
    
    print('Finished classifying data.')
    return table


