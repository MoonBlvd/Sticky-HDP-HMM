import numpy as np
import pickle as pkl
import csv

def save_csv(file_path, data):
    with open(file_path, 'w') as csvfile:
        #field = [field_name]
        writer = csv.writer(csvfile)
        # for line in data:
        writer.writerows(data)


if __name__ == "__main__":
    file_path = '/home/brianyao/Documents/Smart_Black_Box/data/05182017/'
    file_name = '05182017_dictionary.pkl'
    data_dict = pkl.load(open(file_path + file_name, 'rb'), encoding='latin1')


    data = []
    for frame in data_dict:
        curr_data = np.zeros(16)
        for i in range(frame['num_obs']):
            curr_data[i * 4 + 0] = frame['x_obs'][i]
            curr_data[i * 4 + 1] = frame['y_obs'][i]
            curr_data[i * 4 + 2] = frame['x_dot_obs'][i]
            curr_data[i * 4 + 3] = frame['x_ddot_obs'][i]
        # for j in range(frame['num_obs']*4,15):
        #     curr_data[j] = 0
        data.append(curr_data)
    data  = np.array(data)
    print(data[0])


    write_path = "obs_data_4d.csv"
    save_csv(write_path, data)
    # field = ['x','y','v','a',]
    # save_csv(write_path, field, data)

