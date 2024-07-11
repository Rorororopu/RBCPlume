'''
This script performs the following tasks from a list of file paths.:

    1. determine whether the data is sliced. (somehow, regardless of original slicing direction, VisIt will only drop z axis's data)
    2. Determines or calculates the aspect ratio.
    3. Store the range of x, y, and z values.

Data Format:
    
    From the input of list of paths, common properties will be extracted,
    and stored in a class called Datas(self, paths)

    These properties include:
        self.paths:The list of paths of all files.
        self.totstep: Number of elements in the paths list.
        self.slicing: If it is sliced.
        self.aspect_ratio: Aspect ratio value.
        self.x_range: list in format [min, max].
        self.y_range: lsit in format [min, max].
        self.z_range: lsit in format [min, max].
        self.grid_num: Number of drid points in a data file.
'''


def get_var_ranges(filepath: str) -> dict:
    '''
    Convert from a raw Xmdv file to get the ranges of all variable, including the time (will be dropped later).
    The format of file should be:
        num_vars num_rows num_cols_per_var
        var_names
        ...
        var_min var_max 10
        ...
        [table without header]

    Args: 
        The path of the file to be read.

    Returns:
        var_ranges: a dictionary, format {varname:[min, max], ...}.

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
        header = lines[0].split() # Get a list
        num_vars = int(header[0])
    except (IndexError, ValueError):
        print("\033[91mError: The file format is incorrect. Expected a header with the number of variables.\033[0m")
        exit(1)
    
    try:
        # Read variable names below the first line
        var_name = [lines[i + 1].strip() for i in range(num_vars)]
    except IndexError:
        print("\033[91mError: The file format is incorrect. Not enough variable names found.\033[0m")
        exit(1)

    try:
        # Read variable ranges, output in format {varname:(min, max), ...}
        var_ranges = {}
        for i in range(num_vars):
            range_line = lines[num_vars + 1 + i].split()
            min_val, max_val = float(range_line[0]), float(range_line[1])
            var_ranges[var_name[i]] = [min_val, max_val]
    except IndexError:
        print("\033[91mError: The file format is incorrect. Not enough range data or invalid range values found.\033[0m")
        exit(1)

    print(f"Obtained variable ranges of the file at {filepath}.")

    return var_ranges


def get_grid_num(filepath: str) -> int:
    '''
    Read from a raw Xmdv file to get the number of grid points (equivalent to number of rows in the data).

    Args: 
        The path of the file to be read.

    Returns:
        grid_num: an integer shows how many grids(NOT ELEMENTS!) in a simulation. 

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
        print(f"Trying to obtain the number of grid points of the file at {filepath}...")
        header = lines[0].split()
        grid_num = int(header[1])
    except (IndexError, ValueError):
        print("\033[91mError: The file format is incorrect. Expected a header with the number of variables.\033[0m")
        exit(1)

    print(f"Obtained the number of grid points of the file at {filepath}...")
    return grid_num

# Seems like this function is actually not needed, because for 2D data, 
# VisIt will always output the coordinate as x and y, regardless of slicing direction.
# However, there are many codes related to this atritube, so I'm lazy to modify it.
def calculate_slicing(var_ranges: dict) -> str:
    '''
    From the range of 'x', 'y' and 'z', calculate if it is a sliced database.

    Args: 
        The dictionary in format {varname:[min, max], ...}.

    Returns: 
        True: data is sliced, 2D
        False data isn't sliced, 3D

    Error: 
        If column 'x', 'y' or 'z' is not found, prompt the user and kill the program.
    '''
    # Check if 'x', 'y', and 'z' keys are in the dictionary
    print("Trring to determine if the data is sliced...")
    if not all(key in var_ranges.keys() for key in ['x', 'y', 'z']):
        print("\033[91mError: Elements titled 'x', 'y', and 'z' must be found in the input dictionary.\033[0m")
        exit(1) # Kill the program due to error
    
    # Extract the min and max values for 'x', 'y', and 'z'
    xmin, xmax = var_ranges['x']
    ymin, ymax = var_ranges['y']
    zmin, zmax = var_ranges['z']

    # Check for slicing
    if xmax - xmin <= 1e-5:  # Allowing a small tolerance
        print("This data is sliced.")
        return True
    elif ymax - ymin <= 1e-5:
        print("This data is sliced.")
        return True
    elif zmax - zmin <= 1e-5:
        print("This data is sliced.")
        return True
    else:
        print("This data isn't sliced.")
        return False


def calculate_aspect_ratio_3D(var_range: dict) -> float:
    '''
    Calculate the aspect ratio (diameter/depth) by doing this calculation: (xmax - xmin)/(zmax - zmin). 
    Only for 3D output data.

    Args: 
        The dictionary in format {varname:(min, max), ...}.

    Returns: 
        Aspect ratio in float.

    Error: 
        If elements titled 'x' and 'z' are not found, prompt the user and raise a KeyError.
    '''
    # Check if 'x' and 'z' keys are in the dictionary
    if not all(key in var_range for key in ['x', 'y', 'z']):
        print("\033[91mError: Elements titled 'x', 'y', and 'z' must be found in the input dictionary.\033[0m")
        exit(1) # Kill the program due to error

    # Extract the min and max values for 'x' and 'z'
    print("Calculating aspect ratio...")
    xmin, xmax = var_range['x']
    zmin, zmax = var_range['z']

    # Calculate the aspect ratio
    aspect_ratio = (xmax - xmin) / (zmax - zmin)

    return aspect_ratio


def ask_aspect_ratio_2D() -> float:
    '''
    If the database is sliced, the aspect ratio can't be calculated. Hence, we need to ask aspect ratio manually.
    The user's input may be wrong, so be cautious when citing the inputted aspect ratio!
    Returns:
    Aspect ratio in float.
    '''
    while True:
        try:
            aspect_ratio = float(input("\nWhat is the aspect ratio (diameter/depth) of your container? "))
            break  # Exit the loop after a valid input is received
        except ValueError:
            print("\033[91mInvalid input. Please enter a numerical value for the aspect ratio.\033[0m")

    return aspect_ratio


def get_aspect_ratio(slicing:str, var_ranges: dict) -> float:

    '''
    From knowing whether the database is 2D or 3D, call the appripriate function to get aspect ratio.

    Args:
        slicing: A letter that can be 'x', 'y', 'z' or 'n'. Only 'n' means 3D data, the reat are all 2D data.
        var_ranges: a dictionary, format {varname:[min, max], ...}.
    Returns:
        float: aspect ratio.
    '''
    if not slicing:
        aspect_ratio = calculate_aspect_ratio_3D(var_ranges)
    else:
        aspect_ratio = ask_aspect_ratio_2D()

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
            From the list of files, if these properties don't agree with each other, prompt the user and kill the program.

            Arg: 
                paths: a list of file paths.
    '''
    def __init__(self, paths):
        self.paths = paths
        self.totstep = len(paths)
        self.slicing = None # The rest of them are to be calculated
        self.aspect_ratio = None # Not been applied yet
        self.x_range = None
        self.y_range = None
        self.z_range = None
        self.grid_num = None

        for i, path in enumerate(paths):
            var_ranges = get_var_ranges(path)

            if self.slicing is None:
                self.slicing = calculate_slicing(var_ranges)
            elif self.slicing != calculate_slicing(var_ranges):
                raise Exception("\033[91mSlicing direction must be consistent across all files.\033[0m")
            
            if self.x_range is None:
                self.x_range = var_ranges['x']
            elif self.x_range != var_ranges['x']:
                raise Exception("\033[91mX range must be consistent across all files.\033[0m")

            if self.y_range is None:
                self.y_range = var_ranges['y']
            elif self.y_range != var_ranges['y']:
                raise Exception("\033[91mY range must be consistent across all files.\033[0m")

            if self.z_range is None:
                self.z_range = var_ranges['z']
            elif self.z_range != var_ranges['z']:
                raise Exception("\033[91mZ range must be consistent across all files.\033[0m")
            
            if self.grid_num is None:
                self.grid_num = get_grid_num(path)
            elif self.grid_num != get_grid_num(path):
                raise Exception("\033[91mNumber of grids must be consistent across all files.\033[0m")

