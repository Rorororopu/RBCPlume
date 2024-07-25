''' 
Functions to get the path of directory and file to read.
'''

from pathlib import Path # Check if the path exist


def get_directory_path() -> str:
    '''
    Prompt the user to enter the path of the directory to open repeatedly until a valid path is provided.

    Returns:
        directory_path: The inputted path to the directory which has been verified to exist.
    '''
    while True:
        directory_path = input("\nWhat is the path of the directory containing VisIt database you want to open? ")
        if Path(directory_path).is_dir():
            print("Directory opened.")
            return directory_path  # Return the valid directory path
        elif directory_path == "":
            print("\033[91mInvalid directory path. No such directory exists. Please try again.\033[0m") # In red
        else:
            print("\033[91mInvalid directory path. No such directory exists. Please try again.\033[0m")


def get_file_names(directory_path: str) -> list:
    '''
    Get a list of files of data to train the model.
    
    Args:
        directory_path: The path of the directory to search.
    
    Returns:
        An ordered list of paths of files indicated by the user, like [db_0000.okc, db_0001.okc, db_0002.okc, ...]
    '''
    file_list = []
    directory = Path(directory_path)
    while True:
        try:
            file_path = input("Enter the file path: ").strip()
            full_path = directory / Path(file_path)
            if not full_path.is_file():
                print(f"\033[91mFile not found: {full_path}. Please try again.\033[0m")
                continue
            file_list.append(file_path)
            print(f"File added: {file_path}")
            add_another = input("Do you want to add another file? (y/n): ").strip().lower()
            if add_another in ['y', 'Y']:
                continue
            elif add_another in ['n', 'N']:
                print(f"Finished. The list of your files as training dataset is {file_list}")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        except Exception as e:
            print(f"\033[91mAn error occurred: {e}. Please try again.\033[0m")
    return file_list


# ---- These three functions are for getting a list of files to generate a movie ---- #
# ---- Returning an ordered list of paths of a sequence of datas to read ---- #

def if_series() -> bool:
    '''
    Ask if the user wants to classify a series of files or just a few files.
    '''
    while True:
        response = input("Do you want to classify a series of files in your simulation or just a few specific files?\n"
                         "Enter 'y' for a series of files, or 'n' for a few specific files: ").strip()
        if response in ['y', 'Y']:
            return True
        elif response in ['n', 'N']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
        

def get_paths_list(directory_path: str) -> list:
    '''
    This function outputs an ordered list of file paths(Path object) from the specified directory 
    based on user-inputted file name prefix and maybe the file suffix, if multiple suffixs detected.

    Args: 
        directory_path: The path of the directory to search.

    Returns: 
        An ordered list of paths of files indicated by the user, like [db_0000.okc, db_0001.okc, db_0002.okc, ...]
    '''
    while True:
        try:
            print("\nWhat is the file name of your database?\n") # use \n to separate from previous terminal output
            print("For instance, if your database files are named in this format: '<prefix>_0000.okc', or just simply'<prefix>.okc'\n")
            file_prefix = input("please input <prefix>: ")
            files_raw = list(Path(directory_path).glob(f"{file_prefix}*")) # Find all files with the inputted prefix
            files_raw = [file for file in files_raw if file.is_file()] 
        
            if files_raw: #If the list is not empty
                suffixes = {file.suffix for file in files_raw if file.is_file()} # Collect all suffixes in this list, check if there multiple suffixs under same file name
                if len(suffixes) <= 1:
                    files = files_raw
                else:
                    print(f"\033[93mMultiple file suffixes found: {', '.join(suffixes)}\033[0m") #Example: .txt, .csv, ..., in yellow
                    while True:
                        chosen_suffix = input("Please enter the suffix of files you want to open (e.g., .txt, .csv).\n",
                        "Remember to add a DOT to your input: ")
                        files = [file for file in files_raw if file.suffix == chosen_suffix]
                        if not files:  # If files becomes empty
                            print("\033[91mInvalid suffix. Please try again.\033[0m") # In red
                        else:  # If files is not empty
                            break

                files = sorted(files)  # Ensure the final list is sorted
                print(f"Success. {len(files)} file(s) found!")
                return files
            else:
                print(f"\033[91mNo data files found with prefix '{file_prefix}'. Please try again.\033[0m")# In red
            
        except Exception as e:
            print(f"\033[91mAn error occurred: {e}. Please try again.\033[0m")  # In red


def get_file_range(file_paths: list) -> list:
    '''
    This function takes a list of file paths and prompts the user to input two integers to determine the time step they want to read.
    If the input in a single number, it means read the file with that index only.
    Also, if the input is 0, it means read all of the files.

    Args:
        file_paths (list): A list containing file paths to read.
        
    Returns:
        A list in format [start_timestep, end_timestep].
        The earliest starting timestep possible is 1, not 0.
    '''
    while True:
        user_input = input("\nPlease enter your desired time step range to read.\n"
                           f"The range of the time step is from 1 to {len(file_paths)}.\n"
                           "Please input one or two numbers to indicate the wile you want to open.\n"
                           "For instance, if you want to read from the 4th file to the 10th file, please input '4,10' or '4 10'.\n"
                           "If you want to open the 3rd file, please input '3'.\n"
                           "Also, input '0' if you want the program to read all of the files you selected: ")
        user_input = user_input.replace(',', ' ').split()  # Replace commas with spaces and split the input into a list

        # Check if the input is valid
        if len(user_input) == 1:
            try:
                num = int(user_input[0])
                if num == 0:
                    return [1, len(file_paths)]
                elif 1 <= num <= len(file_paths):
                    return [num, num]
                else:
                    print(f"\033[91mPlease enter number(s) between 1 and {len(file_paths)}, or 0 to read all files.\033[0m")
            except ValueError:
                print("\033[91mInvalid input. Please enter integers only.\033[0m")
        
        elif len(user_input) == 2:
            try:
                num1 = int(user_input[0])
                num2 = int(user_input[1])
                if 1 <= num1 <= len(file_paths) and 1 <= num2 <= len(file_paths) and num1 <= num2:
                    return [num1, num2]
                else:
                    print(f"\033[91mPlease ensure both numbers are between 1 and {len(file_paths)} and the first number is less than or equal to the second.\033[0m")
            except ValueError:
                print("\033[91mInvalid input. Please enter integers only.\033[0m")
        
        else:
            print(f"\033[91mPlease enter one or two integers within the range 1 to {len(file_paths)}.\033[0m")


def files_truncate(raw_file_paths: list, file_range: list) -> list:
    '''
    Truncate the file path based on the user given indices.

    Args:
        raw_file_paths (list): The original file paths list to be truncated.
        file_range (list): A short list containing two integers, start and end, which are **1-based** indices.

    Returns:
        list: Truncated path list.
    '''
    start, end = file_range
    # Convert 1-based indices to 0-based for Python list slicing
    start -= 1
    # Adjust end index correctly to handle both single file and range requests
    end = start if start == end-1 else end
    print("Get the truncated list of files to read.")
    return raw_file_paths[start:end+1] # Hell, not including the end+1th element!

