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


    obs = []
    ego = []
    for frame in data_dict:
        curr_ego = [frame['x_dot_ego'],
                    frame['lat'],frame['long'],
                    frame['yaw'],frame['pitch'],frame['roll'],
                    frame['v_east'],frame['v_north'],frame['v_vertical'],
                    frame['accel_x'],frame['accel_y'],frame['accel_z'],
                    frame['P_rate'],frame['Q_rate'],frame['R_rate'],
                    frame['left_heading'],frame['right_heading'],frame['psi_T_0'],
                    frame['left_dist'], frame['right_dist']]
        ego.append(curr_ego)
        # curr_obs = np.zeros(16)
        curr_obs = np.zeros(24)
        curr_obs[0] = -1
        curr_obs[6] = -1
        curr_obs[12] = -1
        curr_obs[18] = -1
        curr_obs[1] = -1
        curr_obs[7] = -1
        curr_obs[13] = -1
        curr_obs[19] = -1
        curr_obs[2] = 255
        curr_obs[8] = 255
        curr_obs[14] = 255
        curr_obs[20] = 255
        curr_obs[3] = 31
        curr_obs[9] = 31
        curr_obs[15] = 31
        curr_obs[21] = 31
        for i in range(frame['num_obs']):
            curr_obs[i * 6 + 0] = frame['obs_ID'][i]
            curr_obs[i * 6 + 1] = frame['obs_age'][i]
            curr_obs[i * 6 + 2] = frame['x_obs'][i]
            curr_obs[i * 6 + 3] = frame['y_obs'][i]
            curr_obs[i * 6 + 4] = frame['x_dot_obs'][i]
            curr_obs[i * 6 + 5] = frame['x_ddot_obs'][i]
        # for j in range(frame['num_obs']*4,15):
        #     curr_obs[j] = 0
        obs.append(curr_obs)
    obs  = np.array(obs)
    print(obs[0])


    write_path = "obs_data_24d.csv"
    save_csv(write_path, obs)
    # write_path = "ego_data_20d.csv"
    # save_csv(write_path, ego)


