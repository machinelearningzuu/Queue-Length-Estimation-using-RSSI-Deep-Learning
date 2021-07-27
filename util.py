import os
import re
import ast
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from variables import*

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(seed)

def seperate_rsu_data():
    rsu_data = open(rsu_data_text,"w+") 
    transmission_data = open(transmission_data_text,"w+") 
    with open(RSUfilename) as f:
        files = f.readlines()
        for i,line in enumerate(files):
            if (i+1) % 3 == 1:
                rsu_data.write(line)
            elif (i+1) % 3 == 2:
                transmission_data.write(line)
    rsu_data.close() 
    transmission_data.close() 

def extract_transmissionId(rsu_data):
    data = re.split('[,\n]', rsu_data)
    data = [d for d in data if d != '']
    for elem in data:
        if 'transmissionId' in elem:
            elem = elem.strip()
            transmissionId = int(elem.split('=')[1].strip())
            return transmissionId

def extract_valid_transmissionIds(rsu_data_array):
    valid_transmissionIds = []
    all_transmissionIds = np.array([extract_transmissionId(rsu_data) for rsu_data in rsu_data_array])
    unique_transmissionIds = list(set(all_transmissionIds))
    for transmissionId in unique_transmissionIds:
        probe_count = (all_transmissionIds==transmissionId).sum()
        if probe_count == n_rsus:
            valid_transmissionIds.append(transmissionId)

    valid_transmissionIds = np.array(valid_transmissionIds)
    return valid_transmissionIds, all_transmissionIds

def preprocess_rsu_text_files():
    rsu_data_array = np.array(open(rsu_data_text,"r").readlines())
    transmission_data_array = np.array(open(transmission_data_text,"r").readlines()) 
    
    valid_transmissionIds, all_transmissionIds = extract_valid_transmissionIds(rsu_data_array)
    valid_indices = np.where(np.in1d(all_transmissionIds, valid_transmissionIds))[0]

    rsu_data_array = rsu_data_array[valid_indices]
    transmission_data_array = transmission_data_array[valid_indices]

    return rsu_data_array, transmission_data_array


def extract_mac(transmission_data):
    data = transmission_data[0]
    seq_number = int(data[-1].strip())
    mac_id = data[2].strip()

    return seq_number, mac_id

def extract_rsu(data):
    rsu_string = data[0]
    rsu_string = rsu_string.split('.')
    for elem in rsu_string:
        if 'RSU' in elem:
            rsu_id = int(elem[-1])
            return rsu_id

def extract_rssi(rsu_data):
    row_values = {feature : [] for feature in RSUfeature_names[:-2] if feature != 'power'}
    for data in rsu_data:
        rsu_id = extract_rsu(data)
        power_elem = data[1]
        f, v = power_elem.split('=')
        v = re.sub('[a-df-zA-Z]', '', v)
        value = float(v.strip())

        value /= 1e-3
        value = 10 * np.log10(value)
        value = round(value, 3)
        for i in range(n_rsus):
            if i+1 == rsu_id:
                row_values[RSUfeature_names[i]].append(value)

    RSUs = []
    for i in range(1, n_rsus+1):
        RSUi = row_values['RSU'+str(i)][0]
        RSUs.append(RSUi)

    return RSUs

def test(rsu_data_array, transmission_data_array):
    rsu_data = open('fuck_rsu.txt',"w+") 
    for line in rsu_data_array:
        rsu_data.write(line)
    rsu_data.close() 

    transmission_data = open('fuck_transmission.txt',"w+") 
    for line in transmission_data_array:
        transmission_data.write(line)
    transmission_data.close()

def create_rsu_csv():
    if not(os.path.exists(rsu_data_text) and os.path.exists(transmission_data_text)):
        seperate_rsu_data()
    
    df_values = {feature : [] for feature in RSUfeature_names if feature != 'power'}
    
    rsu_data_array, transmission_data_array = preprocess_rsu_text_files()

    for i in range(0, len(rsu_data_array), n_rsus):
        rsu_data_lines = rsu_data_array[i:i + n_rsus]
        transmission_data_lines = transmission_data_array[i:i + n_rsus]

        # Extract RSU data from rsu_data.txt and Transmition data from transmission_data.txt
        rsu_data = [[d for d in re.split('[,\n]', rsu_data_line) if d != ''] for rsu_data_line in rsu_data_lines]
        transmission_data = [[d for d in re.split('[:,\n]', transmission_data_line) if d != ''] for  transmission_data_line in transmission_data_lines]
        
        RSUs = extract_rssi(rsu_data)
        seq_number, mac_id = extract_mac(transmission_data)

        for i, RSUi in enumerate(RSUs):
            df_values['RSU'+str(i+1)].append(RSUi)

        df_values['Transmitter'].append(mac_id)
        df_values['Sequence Number'].append(seq_number)

    df = pd.DataFrame(df_values)
    df = df.dropna(axis = 0, how ='any')
    df.to_csv(preprocessed_rsu_csv, index=False)

def extract_nodeId_EndTime(line):
    data = re.split('[,\n]', line)
    data = [d for d in data if d != '']
    node_string = data[0]
    node_string = node_string.split('.')
    for elem in node_string:
        if 'node' in elem:
            node_id = int(elem.replace('node[', '').replace(']', ''))
            break

    for data_elems in data:
        if 'endTime' in data_elems:
            data_elems = data_elems.strip()
            endtime = float(data_elems.split('=')[1].strip())
            break
    return node_id, endtime

def extract_probe_id(line):
    data = re.split('[,\n]', line)
    data = [d for d in data if d != '']
    id_string = data[11].strip()
    id_string = id_string.split('=')
    return int(id_string[1].strip())

def extract_coordinates(line):
    data = line.split('endPosition = ')[1]
    location = data.split(', startOrientation =')[0]
    location = ast.literal_eval(location)

    x = location[0]
    y = location[1]
    return x, y

def preprocess_node_text_file():
    rsu_data_array = np.array(open(rsu_data_text,"r").readlines())
    node_data_array = np.array(open(NodeFilename,"r").readlines())
    
    valid_transmissionIds, _ = extract_valid_transmissionIds(rsu_data_array)
    all_probe_ids = np.array([extract_probe_id(node_data) for node_data in node_data_array])
    valid_indices = np.where(np.in1d(all_probe_ids, valid_transmissionIds))[0]

    node_data_array = node_data_array[valid_indices]
    return node_data_array

def create_node_csv():
    df_values = {feature : [] for feature in NodeFeature_names}
    node_data_array = preprocess_node_text_file()
    for line in node_data_array:
        x, y = extract_coordinates(line)
        node_id, endtime = extract_nodeId_EndTime(line)
        data_row = [node_id, x, y, endtime]
        for d, feature in zip(data_row, NodeFeature_names):
            df_values[feature].append(d)

    df = pd.DataFrame(df_values)
    df = df.dropna(axis = 0, how ='any')
    df.to_csv(preprocessed_node_csv, index=False)

def state_distribution(row):
    x = row['Xcoor']
    y = row['Ycoor']
    
    # Side 1
    if ((lane_length - lane_width) <= y < (lane_length  +  lane_width)) and (0 <= x < (lane_length - common_length)):
        L1bins = np.arange(0,(lane_length - common_length),state_size)[::-1]
        Xstate = np.digitize(x,L1bins,right=False)

        if ((lane_length - lane_width) <= y < lane_length):
            state = 'S1_L1_'+str(Xstate)
        else:
            state = 'S1_L2_'+str(Xstate)

    # Side 2
    elif ((lane_length - lane_width) <= x < (lane_length  +  lane_width)) and ((lane_length  + common_length) <= y < map_size):
        L2bins = np.arange((lane_length  + common_length),map_size,state_size)
        Ystate = np.digitize(y,L2bins,right=True)

        if ((lane_length - lane_width) <= x < lane_length):
            state = 'S2_L1_'+str(Ystate)
        else:
            state = 'S2_L2_'+str(Ystate)

    # Side 3
    elif ((lane_length - lane_width) <= y < (lane_length  +  lane_width)) and ((lane_length  + common_length) <= x < map_size):
        L3bins = np.arange((lane_length  + common_length),map_size,state_size)
        Xstate = np.digitize(x,L3bins,right=True)

        if ((lane_length - lane_width) <= y < lane_length):
            state = 'S3_L1_'+str(Xstate)
        else:
            state = 'S3_L2_'+str(Xstate)

    # Side 4
    elif ((lane_length - lane_width) <= x < (lane_length  +  lane_width)) and (0 <= y < (lane_length - common_length)):
        L4bins = np.arange(0,(lane_length - common_length),state_size)[::-1]
        Ystate = np.digitize(y,L4bins,right=False)

        if ((lane_length - lane_width) <= x < lane_length):
            state = 'S4_L1_'+str(Ystate)
        else:
            state = 'S4_L2_'+str(Ystate)

    else:
        return None

    return state 

def create_final_csv():
    if not os.path.exists(preprocessed_rsu_csv):
        create_rsu_csv()
    if not os.path.exists(preprocessed_node_csv):
        create_node_csv()

    df_node = pd.read_csv(preprocessed_node_csv)
    df_rsu  = pd.read_csv(preprocessed_rsu_csv)
    df_final = pd.concat([df_rsu, df_node], axis=1)

    df_final['state'] = df_final.apply(state_distribution, axis=1)
    df_final[FinalFeature_names[2:6]] = df_final[FinalFeature_names[2:6]].round(3)
    df_final = df_final.dropna(axis = 0, how ='any')
    df_final.to_csv(final_csv , index=False)

def save_scalar(scalar):
    if not os.path.exists(scalar_weights):
        joblib.dump(scalar, scalar_weights) 

def load_scalar():
    scaler = joblib.load(scalar_weights) 
    return scaler


def get_data():
    if not os.path.exists(final_csv):
        print("Preparing CSV data !!!")
        create_final_csv()

    df = pd.read_csv(final_csv)
    df = shuffle(df)
    rsu_names = ['RSU'+str(i) for i in range(1, n_rsus+1)]
    inputs = df[rsu_names].values
    outputs = df['state'].values

    encoder = LabelEncoder()
    encoder.fit(outputs)
    labels = encoder.transform(outputs)

    inputs, labels = shuffle(inputs, labels)
    X, Xtest = inputs[:-Ntest], inputs[-Ntest:]
    Y, Ytest = labels[:-Ntest], labels[-Ntest:]

    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    Xtest = scalar.transform(Xtest)
    save_scalar(scalar)

    if n_components:
        pca = IncrementalPCA(
                n_components=n_components,
                batch_size=batch_size)
        pca.fit(X)
        X = pca.transform(X)
        Xtest = pca.transform(Xtest)

    # X = X.reshape(-1, n_features,1)
    # Xtest = Xtest.reshape(-1, n_features,1)
    return X, Xtest, Y, Ytest, encoder

def get_queue_len_data():
    df = pd.read_csv(final_csv)
    rssis = df[['RSU'+str(i) for i in range(1, n_rsus+1)]].values[-Nqueue:]
    scalar = load_scalar()

    rssis = scalar.transform(rssis)
    Nshift = 3000
    states = df['state'].values[Nshift:Nqueue+Nshift]
    times= df['endTime'].values[Nshift:Nqueue+Nshift]
    return rssis, states, times

def route_accuracy(encoder, model, X, Y):
    ytrue = encoder.inverse_transform(Y).tolist()

    P = model.predict(X)
    ypred = encoder.inverse_transform(P.argmax(axis=-1)).tolist()

    Strue = {
                'S1' : 0,
                'S2' : 0,
                'S3' : 0,
                'S4' : 0
            }
    Spred = {
                'S1' : 0,
                'S2' : 0,
                'S3' : 0,
                'S4' : 0
            }

    for t,p in zip(ytrue, ypred):
        t_side = t.split('_')[0]
        p_side = p.split('_')[0]

        t_lane = t.split('_')[1]
        p_lane = p.split('_')[1]

        Strue[t_side] += 1
        if t == p:
            Spred[p_side] += 1

    # print("\n Simulated distribution of vehicles : {}".format(Strue))
    # print(" Predicted distribution of vehicles : {}\n".format(Spred))
    print("\n")
    for side in Strue.keys():
        side_t = Strue[side]
        side_p = Spred[side]
        print(" Accuracy in Side {} : {}".format(side[1], round(side_p / side_t, 3)))

def state_error_analysis(encoder, model, X, Y):
    State_analysis = {
            'Correct Prediction ' : 0,
            'Single state error ' : 0,
            'Double state error ' : 0,
            'Triple state error ' : 0,
            'Other  state error ' : 0
                    }
    ytrue = encoder.inverse_transform(Y).tolist()

    P = model.predict(X)
    ypred = encoder.inverse_transform(P.argmax(axis=-1)).tolist()

    for t,p in zip(ytrue, ypred):
        if t == p:
            State_analysis['Correct Prediction '] += 1
        else:
            t_side, t_lane, t_state = t.split('_')
            p_side, p_lane, p_state = p.split('_')

            t_state = int(t_state)
            p_state = int(p_state)

            #Single state errors
            if (t_side == p_side) and (t_lane == p_lane) and (abs(t_state - p_state) == 1):
                State_analysis['Single state error '] += 1

            elif (t_side == p_side) and (t_lane != p_lane) and (t_state == p_state):
                State_analysis['Single state error '] += 1

            elif (t_side == p_side) and (t_lane != p_lane) and (abs(t_state - p_state) == 1):
                State_analysis['Single state error '] += 1

            #Double state errors
            elif (t_side == p_side) and (t_lane == p_lane) and (abs(t_state - p_state) == 2):
                State_analysis['Double state error '] += 1

            elif (t_side == p_side) and (t_lane != p_lane) and (abs(t_state - p_state) == 2):
                State_analysis['Double state error '] += 1

            #Triple state errors
            elif (t_side == p_side) and (t_lane == p_lane) and (abs(t_state - p_state) == 3):
                State_analysis['Triple state error '] += 1

            elif (t_side == p_side) and (t_lane != p_lane) and (abs(t_state - p_state) == 3):
                State_analysis['Triple state error '] += 1

            else:
                State_analysis['Other  state error '] += 1

    State_analysis = { k:str(round(v*100/sum(list(State_analysis.values()), 2))) + '%' for k,v in State_analysis.items()}
    print("\n")
    for k, v in State_analysis.items():
        print(" {} percentage : {}".format(k,v))