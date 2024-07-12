'''
Doing calculations like gradient, vorticity and multiplication of variables.
'''

import numpy as np
import pandas as pd

import mapper

def normalizer(data_object:mapper.Data, data:pd.DataFrame, var:str, range:list=[0, 1]) -> pd.DataFrame:
    '''
    Normalize indicated variable (except for the x, y, or z coordinates, or temperature) to the range of [min, max]. 
    The default of range is [0, 1]
    Don't use batch because it will not correctly handle the data for unknown readon.
    You can tell the anomaly from different batches.
    Args:
        data_object: The data object being used.
            self.var_ranges: dict in format {varname:(min, max), ...}, not including x, y, or z columns.
        data: The pandas table to be normalized.
        var: the header of variable to normalize
        range: The range of reglarized data
    Returns:
        Normalized pandas table.
    '''
    
    var_names = list(data_object.var_ranges.keys()) # Convert var_range dict to numpy arrays for min and max values
    min_vals = np.array([data_object.var_ranges[var_name][0] for var_name in var_names])
    max_vals = np.array([data_object.var_ranges[var_name][1] for var_name in var_names])
    
    # Calculate ranges and replace any zero ranges with 1 to avoid division by zero
    print("Regularizing parameters...")
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    data[var] = (data[var] - min_vals[var_names.index(var)]) / ranges[var_names.index(var)] * (range[1] - range[0]) + range[0]
    print("Finished regularizing this data.")
    return data

# The available funciton in numpy to convert from pandas table to numpy array can't work well on the data we're using.
# So I have to make my own one. However, it is quite slow, so please refine it if you need.
def pandas_to_numpy(data_object:mapper.Data, data: pd.DataFrame, var:str) -> np.ndarray:
    '''
    Convert tables of pandas to numpy array of vars indicated. Points out of the range will be NaN.
    
    Args: 
        data_object: the object containing basic information of data.
        data: the pandas data to plot
        var: the variable to convert

    Returns:
        An 2D or 3D numpy array.
    
    Example:
        data:
        x y var
        0 0 13
        0 1 14
        1 0 16
        NaN NaN NaN
    
        output:
        [[13, 16],
        [14, NaN]]
    '''

    array = np.full(data_object.resolution, np.nan) # Generate the numpy array filled with NaN

    # Calculate increment of coordinates of each grid of numpy arrays
    if not data_object.slicing: 
        coord_cols = ['x', 'y', 'z']
        xstep = (data_object.x_range[1] - data_object.x_range[0])/data_object.resolution[0]
        ystep = (data_object.y_range[1] - data_object.y_range[0])/data_object.resolution[1]
        zstep = (data_object.z_range[1] - data_object.z_range[0])/data_object.resolution[2]
    else: # sliced
        coord_cols = ['x', 'y']
        xstep = (data_object.x_range[1] - data_object.x_range[0])/data_object.resolution[0]
        ystep = (data_object.y_range[1] - data_object.y_range[0])/data_object.resolution[1]
    
    print(f"Converting the {var} data to multi-dimensionl array...")
    for index, row in data.dropna(subset=coord_cols).iterrows():
        # Calculate grid indices of each rows
        indices = []
        for col in coord_cols:
            col_range = getattr(data_object, f"{col}_range")
            col_step = locals()[f"{col[0]}step"]
            col_index = int((row[col] - col_range[0]) / col_step)
            indices.append(col_index)

        # Assign the value to the array
        try:
            if len(indices) == 2:
                array[indices[0], indices[1]] = row[var]
            elif len(indices) == 3:
                array[indices[0], indices[1], indices[2]] = row[var]
        except IndexError:  
            pass # Handle cases where the calculated index is out of bounds
    print("Finished converting.")
    return array.T # Attatched from debugging


def calculate_gradients(data_object:mapper.Data, data: pd.DataFrame, var:str) -> pd.DataFrame:
    '''
    Calculate the ABSOLUTE VALUE of the gradient of a given variable of points in a scattered dataset,
    keeping the coordinate columns in the output.

    Points with NaN values and points at the boundry will have NaN value as gradient value

    Batch is not used because it may mess with the boundry points of batches.
    
    Args:
        data_object: An Data object containing many information.
        data: The pandas table to calculate
        var: The name of columns to calculate the gradient for.

    Returns:
        A NEW DataFrame with the coordinate columns and the gradient values for the specified variable.
        the header for gradient is f'{var}_gradient'.

        The sequence of coordinate of the return is the same as the original table.
    '''
    if not data_object.slicing: 
        coord_cols = ['x', 'y', 'z']
    else: # sliced
        coord_cols = ['x', 'y']

    # Prepare the grid for gradient calculation

    if data_object.slicing:
        # convert to numpy array without coordinate data
        grid = pandas_to_numpy(data_object, data, var)
        # Compute gradients
        print(f"Calculating the gradient of {var}...")
        grad_y, grad_x = np.gradient(grid, edge_order=2)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        print("Finished calculating gradient.")
        # Get the coordinate of data
        arrays = {col_var: data[col_var].values for col_var in coord_cols}
        # Converting to numpy array
        for col_var, array in arrays.items():
            arrays[col_var] = np.array(array)

            # Create result dataframe with flattened data
        result_df = pd.DataFrame({
            coord_cols[0]: arrays[coord_cols[0]].ravel(),
            coord_cols[1]: arrays[coord_cols[1]].ravel(),
            f'{var}_gradient': gradient_magnitude.ravel()
            })
    else:
        grid = pandas_to_numpy(data_object, data, var)
        print(f"Calculating the gradient of {var}...")
        grad_x, grad_y, grad_z = np.gradient(grid, edge_order=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        print("Finished calculating gradient.")
        # Get the coordinate of data
        arrays = {col_var: data[col_var].values for col_var in coord_cols}
        # Converting to numpy array
        for col_var, array in arrays.items():
            arrays[col_var] = np.array(array)

            # Create result dataframe with flattened data
        result_df = pd.DataFrame({
            coord_cols[0]: arrays[coord_cols[0]].ravel(),
            coord_cols[1]: arrays[coord_cols[1]].ravel(),
            coord_cols[2]: arrays[coord_cols[2]].ravel(),
            f'{var}_gradient': gradient_magnitude.ravel()
            })

    return result_df

# Not applied in model predicting, just for previous analysis, 
# and it showed that vorticity is has no correlation with temperature gradient.
# Also, ew can't calculate 3D vorticity for a 2D slice.
# Only available if you exported vector/velocity at VisIt.
def calculate_vorticity(data_object:mapper.Data, data: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the vorticity of a velocity field from a given data object using KDTree for nearest neighbor search.

    Args:
        data_object (object): Contains the data in a table and the slicing direction.
        data: the pandas table to calculate.
        batch_size (int): Number of points to process in each batch to handle large datasets efficiently.

    Returns:
        Pandas tables containing the original coordinates and the calculated vorticities. 
        If the data is 2D, the heading will be coordinates and 'vorticity' (with corresponding direction)
        If the data is 3D, the heading will be coordinates, and'x_vorticity', 'y_vorticity', and 'z_vorticity', and 'vorticity_magnitude',
        
    '''
    if not data_object.slicing: 
        coord_cols = ['x', 'y', 'z']
    else: # sliced
        coord_cols = ['x', 'y']

    # Prepare the grid for gradient calculation
    if data_object.slicing:
        vx_grid = pandas_to_numpy(data_object, data, "velocity[0]") # Regardless of slicing direction, only the first two of velocity vectors will not be 0.
        vy_grid = pandas_to_numpy(data_object, data, "velocity[1]")

        dvx_dy, dvx_dx = np.gradient(vx_grid, edge_order=2)
        dvy_dy, dvy_dx = np.gradient(vy_grid, edge_order=2)

        vorticity = dvy_dx - dvx_dy

        # Get the coordinate of data
        arrays = {col_var: data[col_var].values for col_var in coord_cols}
        # Converting to numpy array
        for col_var, array in arrays.items():
            arrays[col_var] = np.array(array)

            # Create result dataframe with flattened data
        result_df = pd.DataFrame({
            coord_cols[0]: arrays[coord_cols[0]].ravel(),
            coord_cols[1]: arrays[coord_cols[1]].ravel(),
            'vorticity': vorticity.ravel()
            })
    else: # 3D
        vx_grid = pandas_to_numpy(data_object, data, "velocity[0]")
        vy_grid = pandas_to_numpy(data_object, data, "velocity[1]")
        vz_grid = pandas_to_numpy(data_object, data, "velocity[2]")

        dvx_dz, dvx_dy, dvx_dx = np.gradient(vx_grid, edge_order=2)
        dvy_dz, dvy_dy, dvy_dx = np.gradient(vy_grid, edge_order=2)
        dvz_dz, dvz_dy, dvz_dx = np.gradient(vz_grid, edge_order=2)

        x_vorticity = dvz_dy - dvy_dz
        y_vorticity = dvx_dz - dvz_dx
        z_vorticity = dvy_dx - dvx_dy

        vorticity_magnitude = np.sqrt(x_vorticity **2 + y_vorticity**2 + z_vorticity**2)

        # Get the coordinate of data
        arrays = {col_var: data[col_var].values for col_var in coord_cols}
        # Converting to numpy array
        for col_var, array in arrays.items():
            arrays[col_var] = np.array(array)

            # Create result dataframe with flattened data
        result_df = pd.DataFrame({
            coord_cols[0]: arrays[coord_cols[0]].ravel(),
            coord_cols[1]: arrays[coord_cols[1]].ravel(),
            coord_cols[2]: arrays[coord_cols[2]].ravel(),
            'x_vorticity': x_vorticity.ravel(),
            'y_vorticity': y_vorticity.ravel(),
            'z_vorticity': z_vorticity.ravel(),
            'vorticity_magnitude': vorticity_magnitude.ravel()
            })

    return result_df

# Not applied in model predicting, just for previous analysis.
def multiplier(data_object:mapper.Data, var1: str, var2: str) -> pd.DataFrame:
    '''
    Multiplies two specified columns in a DataFrame and includes coordinate columns based on slicing.

    Args:
        data_object: An Data object containing the DataFrame and slicing information.
        var1, var2: The name of the first columns to multiply.

    Return: 
        A new rable with the coordinate columns and the result of the multiplication, with header f'{var1}*{var2}'.
    '''
    # Determine coordinate columns based on slicing
    if not data_object.slicing:  # 3D
        coord_cols = ['x', 'y', 'z']
    else:  # sliced
        coord_cols = ['x', 'y']
    
    # Prepare the output DataFrame
    result_df = pd.DataFrame(columns=coord_cols + [f'{var1}*{var2}'])
        
    # Append required coordinate columns
    for col in coord_cols:
        result_df[col] = data_object.data[col]
    
    # Calculate the product of var1 and var2
    result_df[f'{var1}*{var2}'] = data_object.data[var1] * data_object.data[var2]

    return result_df

