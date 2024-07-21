'''
Now the user get a list of files from a same simulation, with same slicing, same variable exported.

This script will obtain some common properties from of these datas, including:
    1. determine whether the data is sliced. (somehow, regardless of original slicing direction, VisIt will only drop z axis's data)
    2. Determines or calculates the aspect ratio.
    3. Store the range of x, y, and z values.

    
Class Datas Format:
    The common properties obtained will be stored in a class called Datas.
    These properties include:
        self.paths:The list of paths of all files.
        self.totstep: Number of elements in the paths list.
        self.slicing: If it is sliced.
        self.aspect_ratio: Aspect ratio value.
        self.x_range: list in format [min, max].
        self.y_range: lsit in format [min, max].
        self.z_range: lsit in format [min, max].
        self.grid_num: Number of drid points in a data file.

    Although the instance of Datas will not be created directly, 
    they will be the parent class of Data class (created at `mapper.py`) for each file.
'''

import typing

def get_info(filepath: str) -> typing.Tuple[dict, int]:
    '''
    Convert from a raw Xmdv file to get the number of grids, and
    the ranges of all variable, including the time (will be dropped later).

    Args: 
        The path of the file to be read.

        The format of file should be:
        num_vars num_rows num_cols_per_var
        var_names
        ...
        var_min var_max 10
        ...
        [table without header]

    Returns:
        var_ranges: a dictionary, format {varname:[min, max], ...}.
        grid_num: number of grid points.

    Errors:
        If the file isn't in this format, prompt user that the file must be Xmdv format exported by VisIt and exit.
    '''
    try:
        # Open the file and read lines
        with open(filepath, 'r') as file:
            print(f"Trying to open the file at {filepath}...")
            lines = file.readlines()

    except FileNotFoundError:
        print(f"\033[91mError: The file at {filepath} does not exist.\033[0m")
        exit(1)
    
    except OSError:
        print(f"\033[91mError: Unable to read the file at {filepath}.\033[91m")
        exit(1)
    
    try:
        # Identify how many variables from the header of database
        print(f"Trying to read the file at {filepath}...")
        
    except (IndexError, ValueError):
        print("\033[91mError: The file format is incorrect. Expected a header with the number of variables.\033[0m")
        exit(1)
    
    try:
        header = lines[0].split() # Get a list
        num_vars = int(header[0])
        
        # Read variable names below the first line
        var_name = [lines[i + 1].strip() for i in range(num_vars)]
    except IndexError:
        print("\033[91mError: The file format is incorrect. Not enough variable names found.\033[0m")
        exit(1)

    try:
        # Read variable ranges, output in format {varname:(min, max), ...}
        print(f"Trying to obtain the range of variables of the file at {filepath}...")
        var_ranges = {}
        for i in range(num_vars):
            range_line = lines[num_vars + 1 + i].split()
            min_val, max_val = float(range_line[0]), float(range_line[1])
            var_ranges[var_name[i]] = [min_val, max_val]

        print(f"Trying to obtain the number of grid points of the file at {filepath}...")
        header = lines[0].split()
        grid_num = int(header[1])

    except IndexError:
        print("\033[91mError: The file format is incorrect. Not enough range data or invalid range values found.\033[0m")
        exit(1)

    print(f"Obtained variable ranges of the file at {filepath}.")

    return var_ranges, grid_num


# Note that for 2D data, VisIt will always output the coordinate as x and y, regardless of slicing direction.
def calculate_slicing(var_ranges: dict) -> str:
    '''
    From the range of 'x', 'y' and 'z', calculate if it is a sliced database.

    Args: 
        The dictionary in format {varname:[min, max], ...}.

    Returns: 
        True: data is sliced, 2D
        False: data isn't sliced, 3D

    Error: 
        If column 'x', 'y' or 'z' is not found, prompt the user and kill the program.
    '''
    # Check if 'x', 'y', and 'z' keys are in the dictionary
    print("Trying to determine if the data is sliced...")
    if not all(key in var_ranges.keys() for key in ['x', 'y', 'z']):
        print("\033[91mError: Elements titled 'x', 'y', and 'z' must be found in the input dictionary.\033[0m")
        exit(1) # Kill the program due to error
    
    # Extract the min and max values for 'x', 'y', and 'z'
    xmin, xmax = var_ranges['x']
    ymin, ymax = var_ranges['y']
    zmin, zmax = var_ranges['z']

    # Check for slicing
    if xmax == xmin : 
        print("This data is sliced.")
        return True
    elif ymax == ymin:
        print("This data is sliced.")
        return True
    elif zmax == zmin:
        print("This data is sliced.")
        return True
    else:
        print("This data isn't sliced.")
        return False


def get_aspect_ratio(slicing:str, var_ranges: dict) -> float:
    '''
    From knowing whether the database is 2D or 3D, call the appripriate function to get aspect ratio.

    If the data is 3D, the aspect ratio can be calculated.

    If the database is sliced, the aspect ratio can't be calculated. Hence, we need to ask aspect ratio manually.
    The user's input may be wrong, so be cautious when citing the inputted aspect ratio!

    Args:
        slicing: A boolean value indicates if it is sliced(2D). If it is False, it is 3D data.
        var_ranges: a dictionary, format {varname:[min, max], ...}.
    Returns:
        float: aspect ratio.
    '''
    if not slicing:
        print("Calculating aspect ratio...")
        xmin, xmax = var_ranges['x']
        zmin, zmax = var_ranges['z']

        # Calculate the aspect ratio
        aspect_ratio = (xmax - xmin) / (zmax - zmin)
    else:
        while True:
            try:
                aspect_ratio = float(input("\nWhat is the aspect ratio (diameter/depth) of your container? "))
                break  # Exit the loop after a valid input is received
            except ValueError:
                print("\033[91mInvalid input. Please enter a numerical value for the aspect ratio.\033[0m")

    return aspect_ratio


class Datas():
    '''
    The class that represents common properties of all data files. 

    Properties:
        self.paths:The list of paths of all files.
        self.totstep: Number of elements in the paths list.
        self.slicing: if data is sliced
        self.aspect_ratio: Aspect ratio value.
        self.x_range: list in format [min, max].
        self.y_range: list in format [min, max].
        self.z_range: list in format [min, max].
        self.grid_num: Number of drid points in a data file.
    
    Functions:
        __init__:
            Calculate all of the properties required. 
            The code assumes that all files in the list is exported from a same simulation, with same slicing, same variable exported,
            so they will only take the first file to get properties needed.

            Arg: 
                paths: a list of file paths.
    '''
    def __init__(self, paths):
        self.paths = paths
        self.totstep = len(paths)
         # The rest of them are to be calculated
        self.aspect_ratio = None # Not been applied yet
        self.x_range = None
        self.y_range = None
        self.z_range = None
        self.grid_num = None

        var_ranges, grid_num = get_info(paths[0])
        self.grid_num = grid_num

        self.slicing = calculate_slicing(var_ranges)
        self.x_range = var_ranges['x']
        self.y_range = var_ranges['y']
        self.z_range = var_ranges['z']


