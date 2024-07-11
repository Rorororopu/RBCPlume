
'''must split the nan and other points before the data goes into model, because I didn't figure out a way to let the model correctly deal with nans.'''

def data_arranger(data: pd.DataFrame) -> typing.Tuple[np.ndarray, typing.List, typing.List]:
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
    non_nan_indices = np.argwhere(~np.isnan(original_array))

    # Get the non-nan values
    non_nan_values = original_array[~np.isnan(original_array)]

    # Combine indices and values
    result = np.column_stack((non_nan_indices, non_nan_values))

    print("Result array:")
    print(result)

    return result, header, non_nan_indices

# %%
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
    
    # Create a mask for non-NaN values
    mask = tf.math.logical_not(tf.math.is_nan(temperature_gradient) | 
                               tf.math.is_nan(velocity_magnitude_gradient) | 
                               tf.math.is_nan(z_velocity_gradient))
    
    # Apply the mask to gradients and classification
    temperature_gradient = tf.boolean_mask(temperature_gradient, mask)
    velocity_magnitude_gradient = tf.boolean_mask(velocity_magnitude_gradient, mask)
    z_velocity_gradient = tf.boolean_mask(z_velocity_gradient, mask)
    classification = tf.boolean_mask(classification, mask)
    
    # Calculate the primary gradient loss
    gradient_avg = (temperature_gradient + velocity_magnitude_gradient + z_velocity_gradient)/3
    loss_high_class_low_grad = classification * (2 - gradient_avg)
    loss_low_class_high_grad = (1 - classification) * gradient_avg
    primary_loss = tf.reduce_mean(loss_high_class_low_grad + loss_low_class_high_grad)
    
    # Add regularization loss to encourage certain properties in classification
    regularization_loss = tf.reduce_mean(tf.square(classification - 0.5))
    
    # Total loss
    loss = primary_loss + 0.1 * regularization_loss
    return loss


header = ['temperature_gradient','velocity_magnitude_gradient','z_velocity_gradient']


class CustomModel(tf.keras.Model):
    def __init__(self, header):
        super(CustomModel, self).__init__()
        self.header = header
        self.dense1 = keras.layers.Dense(len(header), activation='relu')
        self.bn1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(len(header), activation='relu')
        self.bn2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(len(header), activation='relu')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dense3(x)
        return self.output_layer(x)

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = loss_function_NN(x, self.header, y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}

# Create and compile the model
model = CustomModel(header)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# Train the model
loss_hist = LossHistory()
input_tensor = tf.convert_to_tensor(xxxx, dtype='float32')
history = model.fit(input_tensor, batch_size=1, epochs=50, callbacks=[loss_hist])