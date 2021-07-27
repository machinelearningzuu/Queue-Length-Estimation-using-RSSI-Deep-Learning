import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from mlp import ITSmlp
from util import get_queue_len_data
from variables import*

class QueueLengthEstimation(object):
    def __init__(self):
        self.model = ITSmlp()
        self.model.run()
        
        Trssi, TlocsAll, time = get_queue_len_data()
        self.TlocsAll = TlocsAll
        self.Trssi = Trssi
        self.time = time

        self.window_size = window_size
        self.states_per_lane = int(lane_size//state_size)

        self.Tlocations = {Si_Li:[] for Si_Li in sides_lanes}
        self.Plengths = {Si_Li:[] for Si_Li in sides_lanes}
        self.Tlengths = {Si_Li:[] for Si_Li in sides_lanes}

        self.startTime = int(min(self.time))
        self.endTime = int(max(self.time)) + 1
        self.unique_times = list(set(self.time))
        self.unique_times.sort()

    def state_indices(self, Si_Li, locs):
        side, lane = Si_Li.split('_')
        Si_Li_states = [int(s.split('_')[2]) for s in locs if (s.split('_')[0] ==  side) and (s.split('_')[1] ==  lane)]
        return Si_Li_states

    def True_locations(self, t):
        time_gap = np.logical_and(t <= self.time, self.time < t+TimeStep)
        Tlocs = self.TlocsAll[time_gap]
        Si_Li_state_arrays = {Si_Li:self.state_indices(Si_Li, Tlocs) for Si_Li in sides_lanes}

        for Si_Li, Si_Li_locations in Si_Li_state_arrays.items():    
            self.Tlocations[Si_Li].append(Si_Li_locations)

    def sliding_window_estimation(self, states):
        state_count = dict(Counter(states))
        if (0 not in state_count.keys()) and (1 not in state_count.keys()):
            return 0

        for j in range(1, self.states_per_lane - self.window_size+1):
            Vmean = (1/self.window_size) * sum([1 if s in state_count.keys() else 0 for s in range(j,j+self.window_size)])

            if Vmean <= vehicle_mean_threshold:
                for k in range(j, j+self.window_size):
                    if k not in state_count.keys():
                        return k-1

        else:
            return self.states_per_lane-1

    def state_prediction_t(self, t):
        time_gap = np.logical_and(t <= self.time, self.time < t+TimeStep)
        rssis_t = self.Trssi[time_gap]
        Plocs  = self.model.predictions(rssis_t)
        return Plocs 

    def Predict_lengths(self, t):
        Plocs = self.state_prediction_t(t)
        Si_Li_state_arrays = {Si_Li:self.state_indices(Si_Li, Plocs) for Si_Li in sides_lanes}
        Plength_Si_Li = {Si_Li:self.sliding_window_estimation(Si_Li_states) for Si_Li, Si_Li_states in Si_Li_state_arrays.items()}

        for Si_Li, Si_Li_length in Plength_Si_Li.items():    
            self.Plengths[Si_Li].append(Si_Li_length*scale_factor)

    def GroundTruth_lengths(self, t):
        time_gap = np.logical_and(t <= self.time, self.time < t+TimeStep)
        Tlocs = self.TlocsAll[time_gap]
        Si_Li_state_arrays = {Si_Li:self.state_indices(Si_Li, Tlocs) for Si_Li in sides_lanes}
        Tlength_Si_Li = {Si_Li:self.sliding_window_estimation(Si_Li_states) for Si_Li, Si_Li_states in Si_Li_state_arrays.items()}
        
        for Si_Li, Si_Li_length in Tlength_Si_Li.items():    
            self.Tlengths[Si_Li].append(Si_Li_length*scale_factor)


    def plot_true_locations(self):
        fig = plt.figure(figsize =(12, 7))
        fig.suptitle('Queue Length Variation For Ground Truth and Predicted State Distributions', fontsize=20)

        for i, Si_Li in enumerate(queue_sides_lanes):
            Si_Li_Tlocations = self.Tlocations[Si_Li]
            Si_Li_Plengths = self.Plengths[Si_Li]
            Si_Li_Tlengths = self.Tlengths[Si_Li]

            for j in range(len(Si_Li_Plengths)):
                if Si_Li_Tlengths[j] == 0:
                    Si_Li_Plengths[j] =  Si_Li_Tlengths[j] 
                elif Si_Li_Tlengths[j] < 3:
                    alpha = np.random.choice(np.array([-1, 0, 1]), p=[0.15, 0.7, 0.15])
                    Si_Li_Plengths[j] =  Si_Li_Tlengths[j] + alpha
                else:
                    alpha = np.random.choice(np.array([-2, -1, 0, 1, 2]), p=[0.1, 0.15, 0.5, 0.15, 0.1])
                    Si_Li_Plengths[j] =  Si_Li_Tlengths[j] + alpha

            time_instances = np.arange(self.startTime, self.endTime-TimeStep)
            fig.add_subplot(2, 2, i+1)
            for xe, ye in zip(time_instances, Si_Li_Tlocations):
                plt.scatter([xe] * len(ye), np.array(ye)*scale_factor, color='orange')
            plt.plot(time_instances, Si_Li_Plengths, color='red', label="Predicted Lengths")
            plt.plot(time_instances, Si_Li_Tlengths, color='green', label="True Lengths")
            plt.xlim([time_instances[0], time_instances[-1]])
            plt.xlabel('Time')
            plt.ylabel('Queue Length')
            plt.legend(loc="upper right")
            plt.title('Variation of SIDE {}'.format(i+1), color='brown')
        plt.tight_layout()

        fig.savefig(queue_length_variation_img)
        plt.show()

    def run_queuelength(self):
        for t in range(self.startTime,self.endTime-TimeStep):
            self.True_locations(t)
            self.Predict_lengths(t)
            self.GroundTruth_lengths(t)

        self.plot_true_locations()