import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_ctf_2025 
import torch

class Custom_Dataset(Dataset):
    def __init__(self, root = './', dataset = "CHES_2025", leakage = "HW", transform = None,
                 # --- MODIFIED: Added optional custom training/testing ranges for chunking ---
                 custom_train_begin=None, custom_train_end=None,
                 custom_test_begin=None, custom_test_end=None):
                 # --- END MODIFIED ---

        if dataset == "CHES_2025":
            byte = 0
            data_root = 'Dataset/CHES_2025/CHES_Challenge.h5'
            
            # --- MODIFIED: Pass custom ranges to load_ctf_2025, with full dataset defaults ---
            # Default ranges for profiling set
            tb = custom_train_begin if custom_train_begin is not None else 0
            te = custom_train_end if custom_train_end is not None else 500000 # <-- Default to full 500K for profiling
            
            # Default ranges for public attack set
            test_tb = custom_test_begin if custom_test_begin is not None else 0
            test_te = custom_test_end if custom_test_end is not None else 100000 # <-- Default to full 100K for public attack

            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_ctf_2025(
                root + data_root, leakage_model=leakage, byte=byte, 
                train_begin=tb, train_end=te, 
                test_begin=test_tb, test_end=test_te)
            # --- END MODIFIED ---

        print("The dataset we using: ", data_root)
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)
    
    def split_attack_set_validation_test(self):
        # --- MODIFIED: Also split plt_attack ---
        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val, self.plt_attack_test, self.plt_attack_val = train_test_split(
            self.X_attack, self.Y_attack, self.plt_attack, test_size=0.1, random_state=0)
        # --- END MODIFIED ---

    def choose_phase(self,phase):
        # --- MODIFIED: Removed expand_dims (model's forward pass handles it) and added plt ---
        if phase == 'train':
            self.X, self.Y, self.plt = self.X_profiling, self.Y_profiling, self.plt_profiling
        elif phase == 'validation':
            self.X, self.Y, self.plt = self.X_attack_val, self.Y_attack_val, self.plt_attack_val 
        elif phase == 'test':
            self.X, self.Y, self.plt = self.X_attack_test, self.Y_attack_test, self.plt_attack_test
        # --- END MODIFIED ---

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        plaintext = self.plt[idx] # --- MODIFIED: Access plaintext ---
        
        sample = {'trace': trace, 'sensitive': sensitive, 'plaintext': plaintext} # --- MODIFIED: Pass plaintext ---

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor_trace(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        trace, label, plaintext = sample['trace'], sample['sensitive'], sample['plaintext'] # --- MODIFIED: Get plaintext ---

        # The model's forward method should handle unsqueeze for channels
        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).long(), torch.from_numpy(np.array(plaintext)).long() # --- MODIFIED: Return plaintext ---