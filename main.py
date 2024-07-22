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




array1, headers1, indices1 = cnn.data_arranger(data1.df, [200,200])
array2, headers2, indices2 = cnn.data_arranger(data2.df, [200,200])
array3, headers3, indices3 = cnn.data_arranger(data3.df, [200,200])

cnn_model = cnn.model_2D_create_compile(headers1, 0.05, [200,200])

cnn_model, loss_hist = cnn.model_2D_train(cnn_model, array1, 3)
cnn_model, loss_hist = cnn.model_2D_train(cnn_model, array2, 3)

data3.df = cnn.model_2D_classification(cnn_model, array3, indices3, data3.df)


import visualizer
visualizer.plot_2D_df(data3.df, 'is_boundary', 'is_boundary.png')
visualizer.plot_relevance(data3.df, 'is_boundary', 'temperature', 'cnn_tmp.png')
visualizer.plot_relevance(data3.df, 'is_boundary', 'temperature_gradient', 'cnn_tmp_grad.png')
visualizer.plot_relevance(data3.df, 'is_boundary', 'velocity_magnitude_gradient', 'cnn_vmag_grad.png')
visualizer.plot_relevance(data3.df, 'is_boundary', 'z_velocity_gradient', 'cnn_vz_grad.png')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"lapsed time: {elapsed_time} seconds")
