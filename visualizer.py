'''
Using Matplotlib to visualize graphs, output them to PNG image.
'''

from pathlib import Path # Check if the path exist
import pandas as pd
import matplotlib.pyplot as plt

def get_directory_path(param_name:str) -> str:
    '''
    Prompt the user to enter the path of the directory to open repeatedly until a valid path is provided.

    Args:
        param_name: The name of params to plot. Just for the ease of visulization.

    Returns:
        str: The inputted path to the directory which has been verified to exist.
    '''
    while True:
        path = input(f"What is the path of the directory you want to store your image of {param_name}?"
                    "\nFor instance, if you want to store your image as 'image.png' to the curent directory,"
                    "you could input './image.png")
        if Path(path).is_dir():
            print("Success.")
            return path  # Return the valid directory path
        else:
            print("\033[91mInvalid directory path. No such directory exists. Please try again.\033[0m") #In red


def plot_2D_data(data: pd.DataFrame, param_name: str, path: str, cmap: str = 'viridis', range: list = None):
    '''
    Generate a 2D scatter plot from a specified parameter in a pandas table. The plot
    includes a color gradient to represent values, complete with a title, axis labels, and a color bar.
    The resulting plot is saved to the specified file path.

    Args:
    data: The table containing coordinates and data.
    param_name: name of the param to plot.
    path: The path to store the image.
    cmap: The colormap of the image. Choices include:
        'viridis': Ranges from dark blue to bright yellow
        'coolwarm': Diverging colormap, blue to red
    range: Optional. The range of data shown in the plot.
        If specified, values outside this range will be capped to the range limits.
        If None, the full range of the data will be used.
    '''
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Prepare the color data
    color_data = data[param_name]
    
    # Create the scatter plot
    if range is not None:
        sc = plt.scatter(data['x'], data['y'], c=color_data, cmap=cmap, vmin=range[0], vmax=range[1])
    else:
        sc = plt.scatter(data['x'], data['y'], c=color_data, cmap=cmap)
    
    # Add a colorbar
    cbar = plt.colorbar(sc, label=param_name)
    
    plt.title(f'Plot of {param_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save the figure
    plt.savefig(path)
    print("Image saved.")
    plt.close()