import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_ctf_2025
import torch
from scipy.signal import correlate, windows

class Custom_Dataset(Dataset):
    def __init__(self, root='./', dataset="CHES_2025", leakage="HW", transform=None,
                 poi_start=500, poi_end=1500, window_size=5, augment=True):  # Added augment parameter

        if dataset == "CHES_2025":
            byte = 0
            data_root = 'Dataset/CHES_2025/CHES_Challenge.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_ctf_2025(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)

        print("The dataset we using: ", data_root)
        self.transform = transform
        # self.scaler_std = StandardScaler() # Move scaler to after noise reduction
        # self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        # self.X_attack = self.scaler_std.transform(self.X_attack)

        # Noise Reduction Techniques
        self.X_profiling = self.apply_trace_alignment(self.X_profiling)
        self.X_attack = self.apply_trace_alignment(self.X_attack)

        
        self.poi_start = poi_start  # Example values, adjust based on your traces
        self.poi_end = poi_end
        self.window_size = window_size
        self.augment = augment
        
        self.X_profiling = self.apply_denoising_filter(self.X_profiling)
        self.X_attack = self.apply_denoising_filter(self.X_attack)

        self.X_profiling = self.apply_poi_cropping(self.X_profiling, self.poi_start, self.poi_end)
        self.X_attack = self.apply_poi_cropping(self.X_attack, self.poi_start, self.poi_end)

        self.scaler_std = StandardScaler() # Initialize scaler after cropping
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)

        self.split_attack_set_validation_test()
        self.choose_phase("train")  # Initialize X and Y with training data

    def apply_trace_alignment(self, traces):
        # Cross-correlation for trace alignment
        template = traces.mean(axis=0)  # Average trace as template
        aligned_traces = np.zeros_like(traces)
        for i, trace in enumerate(traces):
            correlation = correlate(template, trace, mode='full')
            delay = np.argmax(correlation) - (len(trace) - 1)
            aligned_traces[i] = np.roll(trace, delay)
        return aligned_traces

    def apply_denoising_filter(self, traces):
        # Simple moving average filter for denoising
        window = windows.boxcar(self.window_size) / self.window_size  # Use self.window_size
        denoised_traces = np.zeros_like(traces)
        for i, trace in enumerate(traces):
            denoised_traces[i] = np.convolve(trace, window, mode='same')
        return denoised_traces

    def apply_poi_cropping(self, traces, start, end):
        # Points of Interest (POI) cropping
        return traces[:, start:end]

    def augment_trace(self, trace):
        # Add random noise to the trace
        noise = np.random.normal(0, 0.01, trace.shape)  # Adjust noise level as needed
        trace = trace + noise

        # Shift the trace slightly in time
        shift = np.random.randint(-5, 5)  # Adjust shift range as needed
        trace = np.roll(trace, shift)
        return trace

    def split_attack_set_validation_test(self):
        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val = train_test_split(self.X_attack,self.Y_attack,test_size=0.1,random_state=0)



    def choose_phase(self,phase):
        if phase == 'train':
            self.X, self.Y = np.expand_dims(self.X_profiling, 1), self.Y_profiling
            if self.augment:
                self.X = np.array([self.augment_trace(x) for x in self.X_profiling])  # Apply augmentation
        elif phase == 'validation':
            self.X, self.Y = np.expand_dims(self.X_attack_val, 1), self.Y_attack_val
        elif phase == 'test':
            self.X, self.Y =np.expand_dims(self.X_attack_test, 1), self.Y_attack_test



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        # plaintext = self.Plaintext[idx]
        sample = {'trace': trace, 'sensitive': sensitive} #, 'plaintext': plaintext}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor_trace(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # trace, label, plaintext= sample['trace'], sample['sensitive'], sample['plaintext']
        trace, label= sample['trace'], sample['sensitive']#, sample['plaintext']

        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).long()