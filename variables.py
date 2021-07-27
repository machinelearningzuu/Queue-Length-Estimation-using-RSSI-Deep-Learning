import os
seed = 1111
n_rsus = 4

#Create CSV data and Data preprocessing
RSUfilename = 'data/text_files/rsu.txt'
NodeFilename = 'data/text_files/nodes.txt'
NodeFeature_names = ['nodeID', 'Xcoor', 'Ycoor', 'endTime']

if n_rsus == 4:
    RSUfeature_names = ['RSU1','RSU2','RSU3','RSU4','power', 'Transmitter', 'Sequence Number']
    FinalFeature_names = ['nodeID','state','RSU1','RSU2','RSU3','RSU4', 'Xcoor', 'Ycoor', 'endTime', 'Transmitter', 'Sequence Number']
else:
    RSUfeature_names = ['RSU1','RSU2','RSU3','RSU4','RSU5','RSU6','RSU7','RSU8','power', 'Transmitter', 'Sequence Number']
    FinalFeature_names = ['nodeID','state','RSU1','RSU2','RSU3','RSU4','RSU5','RSU6','RSU7','RSU8', 'Xcoor', 'Ycoor', 'endTime', 'Transmitter', 'Sequence Number']

preprocessed_rsu_csv = 'data/csv_files/rsu_data.csv'
preprocessed_node_csv = 'data/csv_files/node_data.csv'
final_csv = 'data/csv_files/final_data.csv'
rsu_data_text = 'data/text_files/rsu_data.txt'
transmission_data_text = 'data/text_files/transmission_data.txt'
mlp_weights = 'data/weights/mlp_weights.h5'
mlp_converter = 'data/weights/mlp_converter.tflite'
scalar_weights = 'data/weights/standard_scaler.save'
loss_img = 'data/visualization/loss.png'
acc_img = 'data/visualization/acc.png'
dot_img_file = 'data/visualization/plot_model.png'

#Create State distribution
map_size = 200
common_length = 10
lane_length = map_size // 2
lane_size = lane_length - common_length
lane_width = 7
state_size = 5

#Queue Length parameters
TimeStep = 1
window_size = 4
scale_factor = 1.25
vehicle_mean_threshold = 0.5
queue_length_variation_img = 'data/visualization/queue_length_variation.png'
sides_lanes = ['S1_L1', 'S1_L2', 'S2_L1', 'S2_L2', 'S3_L1', 'S3_L2', 'S4_L1', 'S4_L2']
queue_sides_lanes = ['S1_L2', 'S2_L2', 'S3_L1', 'S4_L1']

#MLP parameters
Ntest = 3000
Nqueue = 3000
learning_rate = 0.0001
validation_split = 0.1
n_features = n_rsus
dense1 = 1024
dense2 = 512
dense3 = 256
keep_prob = 0.4
batch_size= 64
num_epoches = 50

#PCA
n_components = None
if n_components:
    n_features = n_components