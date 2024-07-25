'''
Doing calculations like gradient, vorticity and multiplication of variables.

Each data file will be stored in a class called `Data(self, path)`, which have the following properties:
    self.df: The Pandas dataframe (a table) of that file
    self.time: The time value of that file
    self.var_ranges: A dictionary in format {varname:[min, max], ...}. Time and x, y, z elements in this dict are deleted.
    self.resolution: a list in format [resol1, resol2(, maybe resol3)], corresponding to x, y, and z coordinates.

Data(self, path) also inherit from its father class Datas(paths). 
'''

import numpy as np
import pandas as pd

import mapper
import preliminary_processing


# The available funciton in numpy to convert from pandas table to numpy array can't work well on the data we're using.
# Maybe it is because VisIt's data export isn't regularized as numpy expected, so I have to make my own one.
# This function is originally written in for loop and it was extremely slow, even slower than calculating the gradient.
# When changing it to array operation, it becames very fast.


class Data(preliminary_processing.Datas):
    '''
    The class to record the data of one time step.

    Arg:
        path: The corresponding file path for that time step.
        paths: The complete list of file paths.

    Properties:
        self.df: Its corresponding Pandas dataframe.
        self.time: A float value representing its time value.
        self.var_ranges: A dictionary in format {varname:[min, max], ...}. Time, x, y, and z is not included.
        self.resolution: a list in format [resol1, resol2(, maybe resol3)]. If it is none, it means the user have to manually input the resolution in the terminal.
        



    Methods:
        self.pandas_to_numpy: Convert pandas table to 2D/3D numpy array for gradient calculation
        self.normalize: Normalize variables values in self.df
        self.calculate_gradients: Calculate the gradient magnitude of a variable and attatch the reault to self.df
        self.df_process: apply self.normalize and calculate_gradients to get all the data needed.
    '''
    def normalize(self, var:str, range:list=[0, 1]):
        '''
        This is the method of the class Data.
        Normalize indicated variable to the range of [min, max] in self.df, except for the x, y, or z coordinates, or temperature. 
        The default of range is [0, 1].
        Don't use batch because it will not correctly handle the data for unknown readon.
        You can tell the anomaly from different batches.
        Args:
            var: the header of variable to normalize
            range: The range of reglarized data
        '''
        
        var_names = list(self.var_ranges.keys()) # Convert var_range dict to numpy arrays for min and max values
        min_vals = np.array([self.var_ranges[var_name][0] for var_name in var_names])
        max_vals = np.array([self.var_ranges[var_name][1] for var_name in var_names])
        
        # Calculate ranges and replace any zero ranges with 1 to avoid division by zero
        print("Regularizing parameters...")
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1

        self.df[var] = (self.df[var] - min_vals[var_names.index(var)]) / ranges[var_names.index(var)] * (range[1] - range[0]) + range[0]
        print("Finished regularizing this data.")


    def pandas_to_numpy(self, var: str) -> np.ndarray:
        '''
        Convert tables of pandas to numpy array of vars indicated. Points out of the range will be NaN.
        
        Args:
            data_object: the object containing basic information of data.
            var: the variable to convert
        
        Returns:
            An 2D or 3D numpy array.
        
        Example:
            df:
            x y var
            0 0 13
            0 1 14
            1 0 16
            NaN NaN NaN

            output:
            [[13, 16],
            [14, NaN]]
        '''
        array = np.full(self.resolution, np.nan)  # Generate the numpy array filled with NaN

        # Calculate increment of coordinates of each grid of numpy arrays
        if not self.slicing: # 3D
            coord_cols = ['x', 'y', 'z']
            steps = np.array([
                (self.x_range[1] - self.x_range[0]) / self.resolution[0],
                (self.y_range[1] - self.y_range[0]) / self.resolution[1],
                (self.z_range[1] - self.z_range[0]) / self.resolution[2]
            ])
        else:  # sliced
            coord_cols = ['x', 'y']
            steps = np.array([
                (self.x_range[1] - self.x_range[0]) / self.resolution[0],
                (self.y_range[1] - self.y_range[0]) / self.resolution[1]
            ])

        print(f"Converting the {var} data to multi-dimensional array...")

        # Drop NaN values and extract coordinates and variable data
        valid_data = self.df.dropna(subset=coord_cols)
        coords = valid_data[coord_cols].values
        var_values = valid_data[var].values

        # Calculate indices for each coordinate
        ranges = np.array([getattr(self, f"{col}_range") for col in coord_cols])
        indices = np.floor((coords - ranges[:, 0]) / steps).astype(int)

        # Filter out indices that are out of bounds
        mask = np.all((indices >= 0) & (indices < self.resolution[:len(coord_cols)]), axis=1)
        indices = indices[mask]
        var_values = var_values[mask]

        # Assign values to the array
        if len(coord_cols) == 2:
            array[indices[:, 0], indices[:, 1]] = var_values
        elif len(coord_cols) == 3:
            array[indices[:, 0], indices[:, 1], indices[:, 2]] = var_values

        print("Finished converting.")
        return array.T  # Attached from debugging


    def calculate_gradients(self, var:str):
        '''
        This is the method of the class Data.
        Calculate the ABSOLUTE VALUE of the gradient of a given variable at every data points in self.df, 
        and attatch a column with header f{var}_gradient to the self.df

        Points with NaN values and points at the boundry will have NaN value as gradient value.

        Batch is not used because it may mess with the boundry points of batches.
        
        Args:
            data: The pandas table to calculate
            var: The name of columns to calculate the gradient for.
        '''
        if not self.slicing: 
            coord_cols = ['x', 'y', 'z']
        else: # sliced
            coord_cols = ['x', 'y']

        # Prepare the grid for gradient calculation
        if self.slicing: # 3D
            # convert to numpy array without coordinate data
            grid = self.pandas_to_numpy(var)
            # Compute gradients
            print(f"Calculating the gradient of {var}...")
            grad_y, grad_x = np.gradient(grid, edge_order=2)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            print("Finished calculating gradient.")
            # Get the coordinate of data
            arrays = {col_var: self.df[col_var].values for col_var in coord_cols}
            # Converting to numpy array for future operation
            for col_var, array in arrays.items():
                arrays[col_var] = np.array(array)
                # Create result dataframe with flattened data. This is not necessary, 
                # but it makes sure that the value of gradients are corresponding to the correct point in the dataframe.
            result_df = pd.DataFrame({
                coord_cols[0]: arrays[coord_cols[0]].ravel(),
                coord_cols[1]: arrays[coord_cols[1]].ravel(),
                f'{var}_gradient': gradient_magnitude.ravel()
                })
            
            # Attatch the result to the original dataframe
            self.df[f'{var}_gradient'] = result_df[f'{var}_gradient']

        else: # 2D
            # The code is similar to 3D case, so the comment won't be as detailed as 3D case.
            grid = self.pandas_to_numpy(var)
            print(f"Calculating the gradient of {var}...")
            grad_x, grad_y, grad_z = np.gradient(grid, edge_order=1)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            print("Finished calculating gradient.")
            # Get the coordinate of data
            arrays = {col_var: self.df[col_var].values for col_var in coord_cols}
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

            # Attatch the result to the original dataframe
            self.df[f'{var}_gradient'] = result_df[f'{var}_gradient']


    def df_process(self):
        '''Apply the funciton described above'''
        self.normalize('temperature', [-1,1])
        self.normalize('velocity_magnitude', [0,1])
        self.normalize('z_velocity', [-1,1])
        self.calculate_gradients('temperature')
        self.calculate_gradients('velocity_magnitude')
        self.calculate_gradients('z_velocity')


    def __init__(self, path:str, paths:list, resolution:list = None):
        super().__init__(paths)

        var_ranges, _ = preliminary_processing.get_info(path)

        self.time = mapper.get_time(var_ranges) # Record time. If the time is not outputed, the result will be None.
        var_ranges.pop("time_derivative/conn_based/mesh_time") # drop time
        var_ranges.pop('x') # Remove x, y, and z
        var_ranges.pop('y')
        var_ranges.pop('z')

        self.var_ranges = var_ranges # list var ranges

        self.resolution = resolution 
        if self.resolution is None: # if none, the user have to input the resolution themselves
            self.resolution = mapper.get_resolution(self)
        
        self.df = mapper.mapper(self, path, resolution)

        self.df_process() # process the dataframe
        

# Only available if you exported vector/velocity at VisIt, and your data is not sliced.
# Reasons: 1. When the data is sliced, the velocity vector won't include the sliced direction
# 2. Even if you export velocity in differemt directions as 3 scalars, the program can't take the differential in the sliced direction.

# Not applied in model predicting, just for previous analysis, 
# because it showed that vorticity is has no correlation with temperature gradient.
def calculate_vorticity(data_object:Data, data: pd.DataFrame) -> pd.DataFrame:
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
        vx_grid = Data.pandas_to_numpy(data_object, data, "velocity[0]") # Regardless of slicing direction, only the first two of velocity vectors will not be 0.
        vy_grid = Data.pandas_to_numpy(data_object, data, "velocity[1]")

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
        vx_grid = Data.pandas_to_numpy(data_object, data, "velocity[0]")
        vy_grid = Data.pandas_to_numpy(data_object, data, "velocity[1]")
        vz_grid = Data.pandas_to_numpy(data_object, data, "velocity[2]")

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
def multiplier(data_object:Data, var1: str, var2: str) -> pd.DataFrame:
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

