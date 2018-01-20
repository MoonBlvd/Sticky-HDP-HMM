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
from sklearn.mixture import GaussianMixture
import warnings

time_step = 0.1

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
data_path = "drive_segments_9d/*.csv"
files = glob.glob(data_path)
all_segs = []
for name in files:
    data = read_data(name)#[frame_id, obs_ID, obs_age, x_obs, y_obs, v_obs, a_obs, obs_angle, obs_angle_rate]
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
    
    '''
    Method 1
    Observable states are: dx, dv, obs_angle.
    Hidden states are: daccel, angle_rate
    '''
    dx = np.reshape(segment[:,3], (segment.shape[0],1))
    dv = np.reshape(segment[:,5] - ego_car_list[:,0], (segment.shape[0],1))
    obs_angle = np.reshape(segment[:,7], (segment.shape[0],1))
    daccel = np.reshape(segment[:,6] - ego_car_list[:,1], (segment.shape[0],1))
    obs_angle_rate = np.reshape((segment[1:,7] - segment[0:-1,7]), (segment.shape[0]-1,1))/time_step
    obs_angle_rate = np.vstack([obs_angle_rate, obs_angle_rate[-1]])
    # print(dx.shape)
    # print(dv.shape)
    # print(obs_angle.shape)
    # print(daccel.shape)
    # print(obs_angle_rate.shape)
    concat_seg = np.hstack([dx,dv,obs_angle,daccel,obs_angle_rate])
    # print(concat_seg.shape)
    # input("method 1 is done...")

    # concat_seg = np.hstack([segment[:,3:7],ego_car_list]) # 
    data_segments.append(concat_seg)

num_train = 250
train_data = data_segments[0:num_train]
test_data = data_segments[num_train:]

length = []
for data in train_data:
    length.append(data.shape[0])
X = np.concatenate(train_data)
'''
n_components_list = [5,10,15,20]
n_mix_list = [2,5,8]
# n_components_list = [25]
# n_mix_list = [2]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    for i, n_mix in enumerate(n_mix_list):
        for j, n_components in enumerate(n_components_list):
            gmm_hmm = hmm.GMMHMM(n_components=n_components, tol=0.0001, n_mix=n_mix, n_iter=10000, verbose=True, covariance_type="diag")
            # for i,_ in enumerate(gmm_hmm.gmms_):
            # 	gmm_hmm.gmms_[i].covars_ = np.tile(np.identity(5), (n_mix, 1, 1))
            gmm_hmm.fit(X,length)
            joblib.dump(gmm_hmm, 'GMMHMM_model_'+str(n_components)+'_'+str(n_mix)+'.pkl')
'''
# Trian the GMM
n_components_list = [5,10,15,20,25,30]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    for i, n_mix in enumerate(n_mix_list):
        GMM = GaussianMixture(n_components=n_components,
                              covariance_type='full',
                              tol=0.001,
                              max_iter=1000,
                              init_params='kmeans',
                              verbose=1)
        GMM.fit(X)
        joblib.dump(GMM, 'GMM_model_'+str(n_components)+'.pkl')
