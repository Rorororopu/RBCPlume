'''
The main program running that will interact with you. The other programs will automatically as you run this program.
'''
import time
start_time = time.time()


import pandas as pd
import numpy as np

import analyzer
import neural_network as nn
import convolutional_neural_network as cnn

data1 = analyzer.Data("ORIGINAL_TRAINING_DATA/db1.okc", ["ORIGINAL_TRAINING_DATA/db1.okc"], [200,200])
data2 = analyzer.Data("ORIGINAL_TRAINING_DATA/db2.okc", ["ORIGINAL_TRAINING_DATA/db2.okc"], [200,200])
data3 = analyzer.Data("ORIGINAL_TRAINING_DATA/db3.okc", ["ORIGINAL_TRAINING_DATA/db3.okc"], [200,200])

array1, headers1, non_nan_indices1, num_points1 = nn.data_arranger(data1.df)
array2, headers2, non_nan_indices2, num_points2 = nn.data_arranger(data2.df)
array3, headers3, non_nan_indices3, num_points3 = nn.data_arranger(data3.df)

nn_model = nn.model_create_compile(headers1, 0.05)
cnn_model = cnn.model_2D_create_compile(headers1, 0.05, [200,200])

nn_model, loss_hist = nn.model_train(nn_model, array1, 1000, 2)
nn_model, loss_hist = nn.model_train(nn_model, array2, 1000, 2)

data3.df = nn.model_classification(nn_model, array3, non_nan_indices3, num_points3, data3.df)
import visualizer
visualizer.plot_2D_df(data3.df, 'is_boundary', 'is_boundary.png')
visualizer.plot_relevance(data3.df, 'is_boundary', 'temperature_gradient', 't_grad.png')
visualizer.plot_relevance(data3.df, 'is_boundary', 'velocity_magnitude_gradient', 'vmag_grad.png')
visualizer.plot_relevance(data3.df, 'is_boundary', 'z_velocity_gradient', 'vz_grad.png')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"lapsed time: {elapsed_time} seconds")
