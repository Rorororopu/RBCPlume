'''
The main program running that will interact with you. The other programs will automatically as you run this program.
'''

import pandas as pd
import numpy as np
import time

start_time = time.time()
import neural_network as nn
import visualizer as vs

data = pd.read_csv("ORIGINAL_TRAINING_DATA/db1.csv")
input_tensor, header, non_nan_indices, num_grid_points = nn.data_arranger(data)
print(non_nan_indices)
'''
model, history = nn.model_create_compile_train(input_tensor,header,0.001,1000,20)
nn.view_loss_history(history,"hist.png")

to_be_classified = pd.read_csv("ORIGINAL_TRAINING_DATA/db2.csv")
input_tensor, header, non_nan_indices, num_grid_points = nn.data_arranger(data)
result = nn.model_classification(model,input_tensor,non_nan_indices,num_grid_points)
to_be_classified['is_boundry'] = result

vs.plot_2D_data(to_be_classified,'is_boundry',"result.png")

'''
end_time = time.time()
elapsed_time = end_time - start_time
print(f"lapsed time: {elapsed_time} seconds")