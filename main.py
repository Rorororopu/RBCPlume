'''
The main program running that will interact with you. The other programs will automatically as you run this program.
'''
import time

import mapper, analyzer
start_time = time.time()

Data1 = mapper.Data("ORIGINAL_TRAINING_DATA/db1.okc", ["ORIGINAL_TRAINING_DATA/db1.okc"], [200,200])
Data2 = mapper.Data("ORIGINAL_TRAINING_DATA/db1.okc", ["ORIGINAL_TRAINING_DATA/db1.okc"], [200,200])
Data3 = mapper.Data("ORIGINAL_TRAINING_DATA/db1.okc", ["ORIGINAL_TRAINING_DATA/db1.okc"], [200,200])

Data1.data = analyzer.normalizer(Data1, Data1.data, 'temperature', [0,1])
Data2.data = analyzer.normalizer(Data2, Data2.data, 'temperature', [0,1])
Data3.data = analyzer.normalizer(Data3, Data3.data, 'temperature', [0,1])

Data1.data = analyzer.normalizer(Data1, Data1.data, 'velocity_magnitude', [0,1])
Data2.data = analyzer.normalizer(Data2, Data2.data, 'velocity_magnitude', [0,1])
Data3.data = analyzer.normalizer(Data3, Data3.data, 'velocity_magnitude', [0,1])

Data1.data = analyzer.normalizer(Data1, Data1.data, 'z_velocity', [0,1])
Data2.data = analyzer.normalizer(Data2, Data2.data, 'z_velocity', [0,1])
Data3.data = analyzer.normalizer(Data3, Data3.data, 'z_velocity', [0,1])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"lapsed time: {elapsed_time} seconds")