'''
The main program running that will interact with you. The other programs will automatically as you run this program.
'''
import time
start_time = time.time()
import pandas as pd
import numpy as np
import visualizer as vs
import convolutional_neural_network as cnn
import keras.models
import tensorflow as tf
import keras
import keras.optimizers

df1 = pd.read_csv("ORIGINAL_TRAINING_DATA/CSV/data1.csv")
data1, headers1, indices1 = cnn.data_arranger(df1, [200, 200])

model = cnn.model_2D_create_compile(headers1, 0.01, [200, 200])
model, hist = cnn.model_2D_train(model, data1, 2)

df2 = pd.read_csv("ORIGINAL_TRAINING_DATA/CSV/data2.csv")
data2, headers2, indices12 = cnn.data_arranger(df2, [200, 200])
print(model.predict(data2))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"lapsed time: {elapsed_time} seconds")

'''Data1 = mapper.Data("ORIGINAL_TRAINING_DATA/db1.okc", ["ORIGINAL_TRAINING_DATA/db1.okc"], [200,200])
Data2 = mapper.Data("ORIGINAL_TRAINING_DATA/db2.okc", ["ORIGINAL_TRAINING_DATA/db2.okc"], [200,200])
Data3 = mapper.Data("ORIGINAL_TRAINING_DATA/db3.okc", ["ORIGINAL_TRAINING_DATA/db3.okc"], [200,200])

Data1.data = analyzer.normalizer(Data1, Data1.data, 'temperature', [-1,1])
Data2.data = analyzer.normalizer(Data2, Data2.data, 'temperature', [-1,1])
Data3.data = analyzer.normalizer(Data3, Data3.data, 'temperature', [-1,1])

Data1.data = analyzer.normalizer(Data1, Data1.data, 'velocity_magnitude', [0,1])
Data2.data = analyzer.normalizer(Data2, Data2.data, 'velocity_magnitude', [0,1])
Data3.data = analyzer.normalizer(Data3, Data3.data, 'velocity_magnitude', [0,1])

Data1.data = analyzer.normalizer(Data1, Data1.data, 'z_velocity', [0,1])
Data2.data = analyzer.normalizer(Data2, Data2.data, 'z_velocity', [0,1])
Data3.data = analyzer.normalizer(Data3, Data3.data, 'z_velocity', [0,1])

T_grad1 = analyzer.calculate_gradients(Data1, Data1.data, 'temperature')
T_grad2 = analyzer.calculate_gradients(Data2, Data2.data, 'temperature')
T_grad3 = analyzer.calculate_gradients(Data3, Data3.data, 'temperature')
Data1.data['temperature_gradient'] = T_grad1['temperature_gradient']
Data2.data['temperature_gradient'] = T_grad2['temperature_gradient']
Data3.data['temperature_gradient'] = T_grad3['temperature_gradient']

Vmag_grad1 = analyzer.calculate_gradients(Data1, Data1.data, 'velocity_magnitude')
Vmag_grad2 = analyzer.calculate_gradients(Data2, Data2.data, 'velocity_magnitude')
Vmag_grad3 = analyzer.calculate_gradients(Data3, Data3.data, 'velocity_magnitude')
Data1.data['velocity_magnitude_gradient'] = Vmag_grad1['velocity_magnitude_gradient']
Data2.data['velocity_magnitude_gradient'] = Vmag_grad2['velocity_magnitude_gradient']
Data3.data['velocity_magnitude_gradient'] = Vmag_grad3['velocity_magnitude_gradient']

Vz_grad1 = analyzer.calculate_gradients(Data1, Data1.data, 'z_velocity')
Vz_grad2 = analyzer.calculate_gradients(Data2, Data2.data, 'z_velocity')
Vz_grad3 = analyzer.calculate_gradients(Data3, Data3.data, 'z_velocity')
Data1.data['z_velocity_gradient'] = Vz_grad1['z_velocity_gradient']
Data2.data['z_velocity_gradient'] = Vz_grad1['z_velocity_gradient']
Data3.data['z_velocity_gradient'] = Vz_grad1['z_velocity_gradient']

Data1.data.to_csv('ORIGINAL_TRAINING_DATA/GRADIENTS/data1.csv', index=False)
Data1.data.to_csv('ORIGINAL_TRAINING_DATA/GRADIENTS/data2.csv', index=False)
Data1.data.to_csv('ORIGINAL_TRAINING_DATA/GRADIENTS/data3.csv', index=False)

data1 = pd.read_csv("ORIGINAL_TRAINING_DATA/GRADIENTS/data1.csv")
data2 = pd.read_csv("ORIGINAL_TRAINING_DATA/GRADIENTS/data2.csv")
data3 = pd.read_csv("ORIGINAL_TRAINING_DATA/GRADIENTS/data3.csv")

tensor1, header1, non_nan_indices1, num_grids1 = neural_network.data_arranger(data1)
tensor2, header2, non_nan_indices2, num_grids2 = neural_network.data_arranger(data2)
tensor3, header3, non_nan_indices3, num_grids3 = neural_network.data_arranger(data3)

model = neural_network.model_create_compile(header1, 0.01)

model, hist = neural_network.model_train(model, tensor1, 1000, 4)
model, hist = neural_network.model_train(model, tensor2, 1000, 4)

data3 = neural_network.model_classification(model,tensor3, non_nan_indices3, num_grids3, data1)
data3.to_csv("classified.csv", index=False)

data = pd.read_csv("classified.csv")
visualizer.plot_2D_data(data, 'is_boundary', 'is_boundary.png', 'coolwarm')
visualizer.plot_2D_data(data, 'temperature_gradient', 'temperature_gradient.png', 'viridis')
visualizer.plot_2D_data(data, 'velocity_magnitude_gradient', 'velocity_magnitude_gradient.png', 'viridis')
visualizer.plot_2D_data(data, 'z_velocity_gradient', 'z_velocity_gradient.png', 'viridis')
visualizer.plot_2D_data(data, 'temperature', 'temperature.png', 'coolwarm')
'''