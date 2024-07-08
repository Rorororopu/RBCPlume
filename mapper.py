'''
Map the original datas to a evenly spaced grid with user-specified resolution.

The data itself will be stored to a table, with the following titles:
    x, y, z, temperature, velocity_magnitude, z-velocity
Time column is deleted. If the data is sliced, its corresponding x, y, or z coordinate will not be included.
Points out of the original data range will be converted to NaN.

Each data file will be stored in a class called `Data(self, path)`, which have the following properties:
    self.data: The pandas table of that file
    self.time: The time value of that file
    self.var_ranges: A dictionary in format {varname:[min, max], ...}. Time and x, y, z elements in this dict are deleted.
    self.resolution: a list in format [resol1, resol2(, maybe resol3)]

Data(self, path) also inherit from its father class Datas(paths). 

There will also be a dictionary organizing these files, in format {<prefix>_0:Data(path_0), <prefix>_1:Data(path_1), ...}
The prefix is named by user.
'''


import numpy as np
import pandas as pd
import scipy.interpolate
import preliminary_processing


def get_prefix_name() -> str:
    # Get prefix of data objects
    '''
    For instance, if your answer is 'db', files will be named 'db_0', 'db_1', ... 
    '''
    while True:
        prefix_name = input("\nWhat is the prefix name for the Data objects?\n"
                            "For instance, if your answer is 'db', files will be named 'db_0', 'db_1', ... ")
        if not prefix_name or not prefix_name[0].isalpha():
            print("The prefix name must start with an alphabet character and not be empty.")
        else: 
            return prefix_name


def get_time(var_ranges: dict) -> tuple:
    '''
    Extracts the time value from the var_ranges_raw dictionary at the key.
    The min and max values should be the same.
    
    Args: 
        var_ranges: dictionary in format {varname:[min, max], ...}.

    Returns:
        time: The time value extracted from the dictionary.

    Error: 
        If the key 'time_derivative/conn_based/mesh_time' is not found, prints an error message
        and exits the program.
    '''

    key = 'time_derivative/conn_based/mesh_time'
    
    if key not in var_ranges:
        print("\033[91mError: The key 'time_derivative/conn_based/mesh_time' is not found in the dictionary.\033[0m")
        exit(1)
    
    time_range = var_ranges[key]
    
    if time_range[0] != time_range[1]:
        print("\033[91mError: The min and max values for time in one file are not the same.\033[0m")
        exit(1)
    
    time = time_range[0]
    print("Obtain the time for this data.")
    
    return time


def get_resolution_2D(datas_object:preliminary_processing.Datas) -> list:
    '''
    Args:
        a datas_object. Only slicing and grid_num attribute is needed.
    
    Returns:
        resolution: a list in format [res1, res2], in accordance to vars_plotted
    '''
    while True:
        resolution_list = input(f"\nWhat is your desired resolution for your graph?\n"
                   "Please input two numbers in the format 'num1,num2' or 'num1 num2' to corresponding variables.\n"
                   f"Note that there are {datas_object.grid_num} grid points in your simulation, "
                   "and they are sliced.\n"
                   "So don't set a resolution too high or too low compared to your original one.\n"
                   "If it's too low, the information in your data will be lost.\n"
                   "If it's too high, the program can't correctly calculate derivatives.\n"
                   f"Also, the aspect ratio(diameter/depth) of your simulation is {datas_object.aspect_ratio}.\n"
                   "Choose a resolution whose aspect ratio is close to your original one.\n")
        resolution_list = resolution_list.replace(',', ' ').split()  # Replace commas with spaces and split the input into a list
        if len(resolution_list) != 2:
            print("\033[91mInvalid input. Please input 2 numbers!\033[0m")
            continue # Go to the next loop
        try:
            res1 = int(resolution_list[0])
            res2 = int(resolution_list[1])
            return [res1, res2]
        except ValueError:
            print("\033[91mInvalid input. Please input 2 numbers!\033[0m")
    

def get_resolution_3D(datas_object:preliminary_processing.Datas) -> list:
    '''
    Args:
        a datas_object. Only grid_num attribute is needed.
    
    Returns:
        resolution: a list in format [resol1, resol2, resol3], in accordance to vars_plotted
    '''
    while True:
        resolution_list = input(f"\nWhat is your desired resolution for your 'x', 'y' and 'z' graph?\n"
                   "Please input two numbers in the format 'num1,num2,num3' or 'num1 num2 num3' to corresponding variables.\n"
                   f"Note that there are {datas_object.grid_num} grid points in your simulation, "
                   "So don't set a resolution too high or too low compared to your original one.\n"
                   "Choose a resolution similar to your original one.\n"
                   "If it's too low, the information in your data will be lost.\n"
                   "If it's too high, the program can't correctly calculate derivatives.\n"
                   f"Also, the aspect ratio(diameter/depth) of your simulation is {datas_object.aspect_ratio}.\n"
                   "Choose a resolution whose aspect ratio is close to your original one.\n"
                   "If you want to let this program to automatically calculate resolution,\n"
                   "Please replace that value to '-'(hyphen), let the program determine what is its exact value.\n"
                   "For instance, your aspect ratio is 0.1, and you want your z-axis have 1000 grid points,\n"
                   "you could input '-,-,1000', and the program will know that you actually want the resolution be '100,100,1000'.\n"
                   "Note that resolution for x abd y SHOULD NORMALLY BE EQUAL!")
        resolution_list = resolution_list.replace(',', ' ').split()  # Replace commas with spaces and split the input into a list
        if len(resolution_list) != 3:
            print("\033[91mInvalid input. Please input at least 1 number!\033[0m")
            continue # Go to the next loop
        try:
            res1 = int(resolution_list[0]) if resolution_list[0] not in ['-', '_', '*', '#', '.', '..', '...'] else None
            res2 = int(resolution_list[1]) if resolution_list[1] not in ['-', '_', '*', '#', '.', '..', '...'] else None
            res3 = int(resolution_list[2]) if resolution_list[2] not in ['-', '_', '*', '#', '.', '..', '...'] else None
            if (res1 is not None) and (res2 is not None) and (res3 is not None): # No need to calculate
                if res1 != res2:
                    print("\033[91mInvalid input. Resolution for x and y should be equal.\033[0m")
                    continue
                return [res1, res2, res3]
            elif (res3 is not None) and (res1 is None) and (res2 is None): # input is' -, -, res3'
                res1 = int(res3 * datas_object.aspect_ratio)
                res2 = res1
                return [res1, res2, res3]
            elif (res3 is None) and (res1 is not None) and (res2 is not None):  # input is 'res1, res2, -'
                if res1 != res2:
                    print("\033[91mInvalid input. Resolution for x and y should be equal.\033[0m")
                    continue
                res3 = int(res1 / datas_object.aspect_ratio)
                return [res1, res2, res3]
            elif ((res1 is not None) and (res2 is None) and (res3 is None)) or ((res2 is not None) and (res1 is None) and (res3 is None)): # input is 'res1, -, -' or '-, res2, -'
                res1 = res2 if res1 is None else res1
                res2 = res1 if res2 is None else res2
                res3 = int(res1 / datas_object.aspect_ratio)
                return [res1, res2, res3]
            elif (res1 is None) and (res2 is None) and (res3 is None): # Inpiut is '-,-,-'
                print("\033[91mInvalid input. Please input at least 1 number!\033[0m")
            else: # Input is 'res1, -, res3' or '-, res2, res3'
                res1 = res2 if res1 is None else res1
                res2 = res1 if res2 is None else res2
                return [res1, res2, res3]
        except ValueError:
            print("\033[91mInvalid input. Please input at least 1 number!\033[0m")
            continue


def get_resolution(datas_object:preliminary_processing.Datas) -> list:
    '''
    Args:
        a datas_object. Only slicing and grid_num attribute is needed.
    
    Returns:
        resolution: a list in format [res1, res2], or [res1, res2, res3] in accordance to vars_plotted.

    Error:
        if datas_object.slicing doesn't exist, prompt the user and kill the program.
    '''

    if not datas_object.slicing:
        resolution = get_resolution_3D(datas_object)
    else:
        resolution = get_resolution_2D(datas_object)

    return resolution


def mapper_2D(filepath:str, resolution:list) -> pd.DataFrame:
    '''
    Process a sliced data file by interpolating the variables onto a regularly spaced grid defined by the resolution.
    Time column and one of the z values (because it is sliced and VisIt will always make z column be 0) will be dropped. NaN values will be converted to 0. 
    Points out of the original data range will be converted to NaN.

    (We have tried to store the data into a tensor, but it proves to not be the best choice, because it is really hard to drop points.)

    Args:
        filepath: Path of that data.
        resolution: Resolution in format [res1, res2].
        batch_size: To save the memory usage, large data file will be splitted into batch before further processing. This number indicated how many rows are contained in a batch.
    
    Returns: 
        interpolated_data: the table of that data.
    '''
    # From slicing, know what vars to be plotted
    var_ranges = preliminary_processing.get_var_ranges(filepath)
    
    # Prepare mesh grid
    grid_1, grid_2 = (np.linspace(var_ranges[var][0], var_ranges[var][1], res) for var, res in zip(['x', 'y'], resolution)) # Generate 2 arrays of coords
    mesh_grid_1, mesh_grid_2 = np.meshgrid(grid_1, grid_2) # Repeat elements in grid_1 and grid_2
    coordinates = np.stack((mesh_grid_1.ravel(), mesh_grid_2.ravel()), axis=-1) # Flatten mesh_grid_1 and 2 and stack them.

    # Define columns to read
    columns_to_read = [col for col in var_ranges.keys() if col not in ['time_derivative/conn_based/mesh_time', 'z']] # Excluding time, sliced x, y, or z
    indices = [i for i, var in enumerate(var_ranges.keys()) if var in columns_to_read]
    interpolated_results = []
    
    # I planned to make the data process be in batches, but it shows that then it can't deal with the boundry of batches,
    # making the data blurred and behaving strangely. So I deleted the batch part.
    batch = pd.read_csv(filepath, skiprows=1+2*len(var_ranges), delimiter=r'\s+|,', header=None, names=columns_to_read, usecols=indices, engine='python')     
    '''
    Skip header lines.
    Delimiters are whitespace or comma.
    No header present in the CSV.
    Provide column names for the data.
    Only read data in indicated indices.
    Avoid error message, since regular expression is used.
    '''
    batch.fillna(0, inplace=True)# Replace NaNs with zero
    points = batch[['x','y']].values
    interpolated_batch = np.full((coordinates.shape[0], len(columns_to_read)), np.nan)

    # Interpolation for each column
    for i, col in enumerate(columns_to_read):
        values = batch[col].values
        # For grid points out of the range, add NaN as their value (For instance, when slicing direction is z.)
        # Filling value with NaNs may cause many problems, it would be better if it is filled with 0, 
        # but it will have bugs for reason unknown.
        grid_data = scipy.interpolate.griddata(points, values, coordinates, method='linear', fill_value=np.nan)
        interpolated_batch[:, i] = grid_data
    # Store interpolated results in a list
    interpolated_results.append(interpolated_batch)

    # Concatenate all interpolated results and convert them into a DataFrame
    final_data = np.concatenate(interpolated_results, axis=0)
    interpolated_data = pd.DataFrame(final_data, columns=columns_to_read)
    interpolated_data = interpolated_data.reset_index(drop=True)
    
    # Return the final interpolated DataFrame
    print("Finished mapping this data to your specified resolution.")
    return interpolated_data
        

def mapper_3D(filepath:str, resolution:list) -> pd.DataFrame:
    '''
    Process a 3D data file by interpolating the variables onto a regularly spaced grid defined by the resolution.
    Time column will be dropped. NaN values will be converted to 0. 
    Points out of the original data range will be converted to NaN.

    (We have tried to store the data into a tensor, but it proves to not be the best choice, because it is really hard to drop points.)

    Args:
        filepath: Path of that data.
        resolution: Resolution in format [x_res, y_res, z_res].
        batch_size: To save the memory usage, large data file will be splitted into batch before further processing. This number indicated how many rows are contained in a batch.
    
    Returns: 
        interpolated_data: the table of that data.
    '''
    # From slicing, know what vars to be plotted
    var_ranges = preliminary_processing.get_var_ranges(filepath)
    
    # Prepare mesh grid
    grid_x, grid_y, grid_z = (np.linspace(var_ranges[var][0], var_ranges[var][1], res) for var, res in zip(['x', 'y', 'z'], resolution)) # Generate 3 arrays of coords
    mesh_grid_x, mesh_grid_y, mesh_grid_z = np.meshgrid(grid_x, grid_y, grid_z) # Repeat elements in grid_x, y, and z
    coordinates = np.stack((mesh_grid_x.ravel(), mesh_grid_y.ravel(), mesh_grid_z.ravel()), axis=-1) # Flatten mesh_grid_x, y and z, and stack them.

    # Define columns to read
    columns_to_read = [col for col in var_ranges.keys() if col not in 'time_derivative/conn_based/mesh_time'] # Excluding
    indices = [i for i, var in enumerate(var_ranges.keys()) if var in columns_to_read]
    interpolated_results = []

    # I planned to make the data process be in batches, but it shows that then it can't deal with the boundry of batches,
    # making the data blurred and behaving strangely. So I deleted the batch part.
    batch = pd.read_csv(filepath, skiprows=1+2*len(var_ranges), delimiter=r'\s+|,', header=None, names=columns_to_read, usecols=indices, engine='python')
    '''
    Skip header lines.
    Delimiters are whitespace or comma.
    No header present in the CSV.
    Provide column names for the data.
    Only read data in indicated indices.
    Avoid error message, since regular expression is used.
    '''
    batch.fillna(0, inplace=True)# Replace NaNs with zero
    batch.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
    points = batch[['x', 'y', 'z']].values
    interpolated_batch = np.full((coordinates.shape[0], len(columns_to_read)), np.nan)

    # Interpolation for each column
    for i, col in enumerate(columns_to_read):
        values = batch[col].values
        grid_data = scipy.interpolate.griddata(points, values, coordinates, method='linear', fill_value=np.nan)# For grid points out of the range, add NaN as their value (For instance, when slicing direction is z.)
        interpolated_batch[:, i] = grid_data
    # Store interpolated results in a list
    interpolated_results.append(interpolated_batch)
    
    # Concatenate all interpolated results and convert them into a DataFrame
    final_data = np.concatenate(interpolated_results, axis=0)
    interpolated_data = pd.DataFrame(final_data, columns=columns_to_read)
    interpolated_data = interpolated_data.reset_index(drop=True)
    # Return the final interpolated DataFrame
    print("Finished mapping this data to your specified resolution.")
    return interpolated_data


def mapper(datas_object:preliminary_processing.Datas, filepath:str, resolution:list) -> pd.DataFrame:
    '''
    Process an data file by interpolating the variables onto a regularly spaced grid defined by the resolution.
    Time column and if the data is sliced, z values (it will always be 0 if it is sliced) will be dropped. NaN values will be converted to 0. 
    Points out of the original data range will be converted to NaN.

    Args:
        datas_object: just to jnow the slicing.
        filepath: Path of that data.
        resolution: Resolution in format [res1, res2] ot [x_res, y_res, z_res].
        batch_size: To save the memory usage, large data file will be splitted into batch before further processing. This number indicated how many rows are contained in a batch.
    
    Returns: 
        interpolated_data: the interpolated table data.
    '''
    if not datas_object.slicing:
        interpolated_data = mapper_3D(filepath, resolution)
    else:
        interpolated_data = mapper_2D(filepath, resolution)
    return interpolated_data


class Data(preliminary_processing.Datas):
    '''
    The class to record the data of one time step.

    Arg:
        path: The corresponding file path for that time step.
        paths: The complete list of file paths.

    Properties:
        self.data: Its corresponding data table.
        self.time: A float value representing its time value.
        self.var_ranges: A dictionary in format {varname:[min, max], ...}. Time, x, y, and z is not included.
        self.resolution: a list in format [resol1, resol2(, maybe resol3)]
    '''
    def __init__(self, path:str, paths:list, resolution:list):
        super().__init__(paths)

        var_ranges = preliminary_processing.get_var_ranges(path)
        var_ranges.pop('x') # Remove x, y, and z
        var_ranges.pop('y')
        var_ranges.pop('z')

        self.time = get_time(var_ranges) # Record time
        var_ranges.pop("time_derivative/conn_based/mesh_time") # drop time

        self.var_ranges = var_ranges # list var ranges

        self.resolution = resolution
        self.data = mapper(self, path, resolution)