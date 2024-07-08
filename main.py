'''
The main program running that will interact with you. The other programs will automatically as you run this program.
'''

import model
import pandas as pd
import numpy as np
import time



start_time = time.time()

def main2(path:str):
    data=pd.read_csv(path)
    array=model.data_arranger_NN(data)
    np.savetxt(array, "delete.txt")

main2("ORIGINAL_TRAINING_DATA/db1.csv")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"lapsed time: {elapsed_time} seconds")

'''
def main1(path:str, prefix:str):

    data_object = mapper.Data(path, [path], [200,200])
    data_object.aspect_ratio = 1

    data_object.data = analyzer.normalizer(data_object, data_object.data, "temperature", [-1, 1])

    temp_grad = analyzer.calculate_gradients(data_object, data_object.data, "temperature")
    vmag_grad = analyzer.calculate_gradients(data_object, data_object.data, "velocity_magnitude")
    vz_grad = analyzer.calculate_gradients(data_object, data_object.data, "z_velocity")

    final_grid = pd.DataFrame()
    final_grid['x'] = data_object.data['x']
    final_grid['y'] = data_object.data['y']
    final_grid["temperature"] = data_object.data["temperature"]
    final_grid["temperature_gradient"] = temp_grad["temperature_gradient"]
    final_grid["velocity_magnitude_gradient"] = vmag_grad["velocity_magnitude_gradient"]
    final_grid["z_velocity_gradient"] = vz_grad["z_velocity_gradient"]
    final_grid.to_csv(f"ORIGINAL TRAINING DATA/{prefix}.csv", index=False)
main1("../highRnum/db1.okc","db1")
main1("../highRnum/db2.okc","db2")
main1("../highRnum/db3.okc","db3")
main1("../highRnum/db4.okc","db4")
main1("../highRnum/db5.okc","db5")
main1("../highRnum/db6.okc","db6")
main1("../highRnum/db7.okc","db7")
main1("../highRnum/db8.okc","db8")
'''