from __future__ import division
import numpy as np
from numpy.random import choice, normal, multivariate_normal
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import glob
from hmmlearn import hmm
import csv
import warnings
from sklearn.externals import joblib
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def read_data(file_path):
    # The read-in data should be a N*W matrix,
    # where N is the length of the time sequences,
    # W is the number of sensors/data features
    i = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for line in reader:
            line = np.array(line, dtype = 'float') # str2float
            if i == 0:
                data = line
            else:
                data = np.vstack((data, line))
            i += 1
    return data

'''
Load the segments where a continuous obs car exists
'''
data_path = "drive_segments/*.csv"
files = glob.glob(data_path)
all_segs = []
for name in files:
    data = read_data(name)#[frame_id, x_obs,y_obs,dv,a_obs]
    data[:,0] += 1149 # change the 
    all_segs.append(data)
'''
load the ego car data
'''
ego_data_path = "ego_data_24d.csv"
ego_data = read_data(ego_data_path)
ego_lane = ego_data[:,[18,19,15,16,20,21,22,23]]
ego_data = ego_data[:,[0,9,17,14,18,19]] # x_dot_ego ,accel_x, psi_T_0, r_rate, left_dist, right_dist
'''
Load the MRM result
'''
data_name='MRM_result.pkl'
MRM_results = pkl.load(open(data_name, 'rb'))

'''
match obs car data segments with ego car data
'''
num_actions = 3
data_segments = []
for segment in all_segs:
    ego_lane_list = []
    ego_car_list = []
    ego_prob_list = []
    obs_prob_list = []
    violation_prob_list = []
    for i, MRM in enumerate(MRM_results):
        if MRM['frame_id'] < 1150: # skip parking lot
            continue
        if MRM['frame_id']+1 in segment[:,0]:
            # read ego car data
            ego_lane_data = ego_lane[MRM['frame_id'],:]
            ego_car_data = ego_data[MRM['frame_id'],:]
            matched_obs_idx = np.where(MRM['obs_id'] == segment[0,1])[0]
            ego_action_prob = np.reshape(MRM['ego_probs'], (num_actions,))
            obs_action_prob = np.reshape(MRM['obs_probs'][:,matched_obs_idx], (num_actions,))#matched_obs_idx*3:(matched_obs_idx+1)*3
            violation_prob = np.reshape(MRM['violation_prob'][:,matched_obs_idx,:], (num_actions, num_actions))

            # save to list
            ego_lane_list.append(ego_lane_data)
            ego_car_list.append(ego_car_data)
            ego_prob_list.append(ego_action_prob)
            obs_prob_list.append(obs_action_prob)
            violation_prob_list.append(violation_prob)
    ego_lane_list = np.array(ego_lane_list)
    ego_car_list = np.array(ego_car_list)
    ego_prob_list = np.array(ego_prob_list)        
    obs_prob_list = np.array(obs_prob_list)
    violation_prob_list = np.array(violation_prob_list)
    
    concat_seg = np.hstack([segment[:,3:7],ego_car_list])
    data_segments.append(concat_seg)

num_train = 250
train_data = data_segments[0:num_train]
test_data = data_segments[num_train:]

length = []
for data in train_data:
    length.append(data.shape[0])
X = np.concatenate(train_data)
# from sklearn.hmm import GMMHMM
# n_components_list = [5,10,15]
# n_mix_list = [2,5,8]
n_components_list = [50,70]
n_mix_list = [1]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    for i, n_mix in enumerate(n_mix_list):
        for j, n_components in enumerate(n_components_list):
            gmm_hmm = hmm.GMMHMM(n_components=n_components, tol=0.0001, n_mix=n_mix, n_iter=10000, verbose=True)
            gmm_hmm.fit(X,length)
            joblib.dump(gmm_hmm, 'GMMHMM_model_'+str(n_components)+'_'+str(n_mix)+'.pkl')
