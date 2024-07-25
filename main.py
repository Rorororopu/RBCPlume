'''
The main program running that will interact with you. The other programs will automatically as you run this program.
'''
import time
start_time = time.time()

# %% ---- Import necessary libraries ---- #
import pandas as pd
import numpy as np
from pathlib import Path

import get_directory_file
import analyzer
import neural_network as nn
import convolutional_neural_network as cnn
import visualizer

# %% ---- Get lists of files to train the model ---- #
print("Now you have to input the path of the directory containing the file to train the model.")

directory_train = get_directory_file.get_directory_path()
list_paths_train = get_directory_file.get_file_names(directory_train)

Data_training = [analyzer.Data]  # Create an empty list to store Data objects
for path in list_paths_train:
    data = analyzer.Data(path, list_paths_train)
    Data_training.append(data)
'''
UNCOMMENT THIS CODE TO MANUALLY PREPARE THE DATA TO TRAIN THE MODEL.

NOTE THAT THE SPECIFIED RESOLUTION SHOULD BE CONSISTENT ALL OVER THE FILE.

resolution = [200,200]

data1 = analyzer.Data("ORIGINAL_TRAINING_DATA/db1.okc", ["ORIGINAL_TRAINING_DATA/db1.okc"], resolution)
data2 = analyzer.Data("ORIGINAL_TRAINING_DATA/db2.okc", ["ORIGINAL_TRAINING_DATA/db2.okc"], resolution)

'''
# %% ---- Create, train the model ---- #
arrays_training = [np.ndarray] # create lists to store arrays to train the model

model_choice = '' # To be accessed outside the loop
model = None

while True:
    model_choice = input("Which structure do you want to use in your model?\nNeural Network (answer NN) or Convolutional Neural Network (answer CNN)? ").lower()

    if model_choice == 'nn':

        arrays_training = []
        for data in Data_training:
            array, headers, *_ = nn.data_arranger(data.df)
            arrays_training.append(array)

        model = nn.model_create_compile(headers, 0.05)

        for array in arrays_training:
            model, hist = nn.model_train(model, array, 1000, 5)
        break

    elif model_choice == 'cnn':

        arrays_training = []
        for data in Data_training:
            array, headers, *_ = cnn.data_arranger(data.df, data.resolution)
            arrays_training.append(array)
            
        model = cnn.model_2D_create_compile(headers, 0.05, Data_training[0].resolution)

        for array in arrays_training:
            model, hist = cnn.model_2D_train(model, array, 3)
        break

    else:
        print("\033[91mPlease input a valid answer!\033[0m")


'''
UNCOMMENT THIS CODE TO MANUALLY CREATE AND TRAIN THE MODEL.

array1, headers1, indices1 = cnn.data_arranger(data1.df, resolution)
array2, headers2, indices2 = cnn.data_arranger(data2.df, resolution)

# The learning rate and epochs is proven to be working.

cnn_model = cnn.model_2D_create_compile(headers1, 0.05, resolution)

cnn_model, loss_hist = cnn.model_2D_train(cnn_model, array1, 3)
cnn_model, loss_hist = cnn.model_2D_train(cnn_model, array2, 3)
'''

# %%  ---- Get lists of files to classify ---- #
print("Now you have to input the path of the directory containing the file to classify.")

directory_classify = get_directory_file.get_directory_path()

if_series = get_directory_file.if_series()

if if_series:
    list_paths_classify = get_directory_file.get_paths_list(directory_classify)
    range_to_read = get_directory_file.get_file_range(list_paths_classify)
    list_paths_classify = get_directory_file.files_truncate(list_paths_classify, range_to_read)
else:
    list_paths_classify = get_directory_file.get_file_names(directory_classify)

Data_classify = [analyzer.Data(path, list_paths_classify, Data_training[0].resolution) for path in list_paths_classify] 
'''
UNCOMMENT THIS CODE TO MANUALLY PREPARE THE DATA TO TRAIN THE MODEL.

list_paths_classify = [f"TEST_VIDEO/db_Y_{i:04d}.okc" for i in range(90, 101)]
'''
# %% ---- Classify the data ---- #
temp = None
while True:
    ans = input("Do you include the temperature information to the classification result?\n i.e. let the range of the column of is_boundary be ranging [0,1] to [-1,1]").lower()
    if ans == 'y':
        temp = True
        break
    elif ans == 'n':
        temp = False
        break
    else:
        print("\033[91mPlease input a valid answer!\033[0m")

if model_choice == 'nn':
    arrays_classify = []
    for data in Data_classify:
        array, headers, non_nan_indices, num_grids = nn.data_arranger(data.df)
        data.df = nn.model_classification(model, array, non_nan_indices, num_grids, data.df, temp)

elif model_choice == 'cnn':
    arrays_classify = []
    for data in Data_classify:
        array, headers, indices = cnn.data_arranger(data.df, data.resolution)
        data.df = cnn.model_2D_classification(model, array, indices, data.df, temp)

'''
UNCOMMENT THIS CODE TO MANUALLY CLASSIFY THE DATA
Data_classify = [analyzer.Data]
for path in list_paths_classify:
    data = analyzer.Data(path, list_paths_classify, resolution)
    
    array, headers, indices = cnn.data_arranger(data.df, resolution)
    
    data.df = cnn.model_2D_classification(cnn_model, array, indices, data.df, False)

    Data_classify.append(data)
'''
# %% ---- Export the data ---- #
def get_prefix_name() -> str:
    while True:
        prefix_name = input("\nWhat is the prefix name for the exported data?\n"
                            "For instance, if your answer is 'db', files will be named 'db_0.png/csv', 'db_1.png/.csv', ... ")
        if not prefix_name or not prefix_name[0].isalpha():
            print("The prefix name must start with an alphabet character and not be empty.")
        else: 
            return prefix_name

prefix_name = get_prefix_name()


while True:
    directory_path = input("\nPlease input the path you want to store the output result.")
    if Path(directory_path).is_dir():
        print("Directory opened.")
        break
    elif directory_path == "":
        print("\033[91mInvalid directory path. No such directory exists. Please try again.\033[0m") # In red
    else:
        print("\033[91mInvalid directory path. No such directory exists. Please try again.\033[0m")

    
while True:
    plot = input("Do you want to export the image plotted based of your results?").lower()
    if plot == 'y':
        for i, data in enumerate(Data_classify):
            visualizer.plot_2D_df(data.df, 'is_boundary', f'{directory_path}/{prefix_name}_{i}.png')
        break
    elif plot == 'n':
        break
    else:
        print("\033[91mPlease input a valid answer!\033[0m")

while True:
    csv = input("Do you want to export the classification results to CSV files?").lower()
    if csv == 'y':
        for i, data in enumerate(Data_classify):
            data.df.to_csv(f'{directory_path}/{prefix_name}_{i}.csv', index=False)
        break
    elif csv == 'n':
        break
    else:
        print("\033[91mPlease input a valid answer!\033[0m")

# %% ---- Calculate the time spent ---- #
end_time = time.time()
elapsed_time = end_time - start_time
print(f"lapsed time: {elapsed_time} seconds")
