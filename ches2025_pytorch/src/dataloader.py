import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_ctf_2025
import torch
# Add new imports for signal processing
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter, butter, filtfilt, medfilt
from scipy import stats

class Custom_Dataset(Dataset):
    def __init__(self, root='./', dataset="CHES_2025", leakage="HW", transform=None,
                 clean_data=False,
                 moving_avg_window=5,
                 savgol_window=15,
                 savgol_poly=3,
                 poi_selection=True,
                 num_poi=3000,
                 select_best_traces=False,
                 num_best_traces=35000,
                 apply_median_filter=True,
                 median_window=3,
                 apply_butterworth=True,
                 butterworth_cutoff=0.15,
                 butterworth_order=4):

        if dataset == "CHES_2025":
            byte = 0
            data_root = 'Dataset/CHES_2025/CHES_Challenge.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_ctf_2025(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)

        print("The dataset we using: ", data_root)
        print(f"Original trace length: {self.X_profiling.shape[1]} samples")
        
        # Apply data cleaning if requested
        if clean_data:
            print("Applying trace cleaning techniques...")
            self.X_profiling = self.clean_traces(
                self.X_profiling,
                moving_avg_window=moving_avg_window,
                savgol_window=savgol_window,
                savgol_poly=savgol_poly,
                poi_selection=poi_selection,
                num_poi=num_poi,
                apply_median_filter=apply_median_filter,
                median_window=median_window,
                apply_butterworth=apply_butterworth,
                butterworth_cutoff=butterworth_cutoff,
                butterworth_order=butterworth_order
            )
            self.X_attack = self.clean_traces(
                self.X_attack,
                moving_avg_window=moving_avg_window,
                savgol_window=savgol_window,
                savgol_poly=savgol_poly,
                poi_selection=poi_selection,
                num_poi=num_poi,
                apply_median_filter=apply_median_filter,
                median_window=median_window,
                apply_butterworth=apply_butterworth,
                butterworth_cutoff=butterworth_cutoff,
                butterworth_order=butterworth_order,
                poi_indices=getattr(self, 'poi_indices', None)  # Use same POI indices as profiling
            )
            print(f"Cleaned trace length: {self.X_profiling.shape[1]} samples")
            
        # Select best traces based on noise level if requested
        if select_best_traces and num_best_traces is not None and num_best_traces < len(self.X_profiling):
            print(f"Selecting {num_best_traces} traces with lowest noise...")
            self.X_profiling, self.Y_profiling, self.plt_profiling = self.select_traces_by_noise(
                self.X_profiling, self.Y_profiling, self.plt_profiling, num_best_traces
            )
            print(f"Selected {len(self.X_profiling)} high-quality training traces")
        
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)

    def clean_traces(self, traces, moving_avg_window=5, savgol_window=15, savgol_poly=3,
                    poi_selection=True, num_poi=3000,
                    apply_median_filter=True, median_window=3,
                    apply_butterworth=False, butterworth_cutoff=0.1, butterworth_order=3,
                    poi_indices=None):
        """
        Apply comprehensive signal processing techniques to clean traces with POI selection
        
        Parameters:
        -----------
        traces : np.ndarray
            Raw traces to clean (n_traces, n_samples)
        moving_avg_window : int
            Window size for moving average filter (denoising)
        savgol_window : int
            Window size for Savitzky-Golay filter (advanced smoothing)
        savgol_poly : int
            Polynomial order for Savitzky-Golay filter
        poi_selection : bool
            Whether to apply POI selection based on SNR
        num_poi : int
            Number of top POIs to select (default: 3000)
        apply_median_filter : bool
            Whether to apply median filtering for additional noise reduction
        median_window : int
            Window size for median filtering
        apply_butterworth : bool
            Whether to apply Butterworth low-pass filtering
        butterworth_cutoff : float
            Cutoff frequency for Butterworth filter (0-1)
        butterworth_order : int
            Order of Butterworth filter
        poi_indices : np.ndarray, optional
            Pre-computed POI indices (for attack traces to use same POIs as profiling)
            
        Returns:
        --------
        np.ndarray
            Cleaned traces with POI selection applied
        """
        print("Step 1: Applying Moving Average Filter for denoising...")
        cleaned_traces = traces.copy()
        
        # 1. Moving Average Filter (uniform_filter1d) - Denoise by smoothing fluctuations
        cleaned_traces = np.array([uniform_filter1d(trace, size=moving_avg_window) 
                                 for trace in cleaned_traces])
        
        print("Step 2: Applying Savitzky-Golay Filter for advanced smoothing...")
        # 2. Savitzky-Golay Filter - Advanced smoothing while preserving features
        # Ensure window is odd and greater than polynomial order
        if savgol_window % 2 == 0:
            savgol_window += 1
        if savgol_window <= savgol_poly:
            savgol_window = savgol_poly + 2
        
        cleaned_traces = np.array([savgol_filter(trace, window_length=savgol_window, polyorder=savgol_poly) 
                                 for trace in cleaned_traces])
        
        # 3. Apply Median Filtering for additional noise reduction
        if apply_median_filter:
            print("Step 3: Applying Median Filter for noise reduction...")
            cleaned_traces = np.array([medfilt(trace, kernel_size=median_window) 
                                     for trace in cleaned_traces])
        
        # 4. Apply Butterworth Filtering for frequency-based noise reduction
        if apply_butterworth:
            print("Step 4: Applying Butterworth Filter for frequency-based noise reduction...")
            nyq = 0.5  # Nyquist frequency (assuming normalized frequency)
            normal_cutoff = butterworth_cutoff / nyq
            b, a = butter(butterworth_order, normal_cutoff, btype='low', analog=False)
            cleaned_traces = np.array([filtfilt(b, a, trace) for trace in cleaned_traces])
        
        # 5. SNR-based POI Selection
        if poi_selection:
            print(f"Step 5: Applying SNR-based POI Selection for top {num_poi} points...")
            if poi_indices is None:
                # Calculate POI indices for profiling traces
                poi_indices = self._calculate_poi_snr(cleaned_traces, num_poi)
                # Store POI indices for later use with attack traces
                self.poi_indices = poi_indices
            
            # Apply POI selection
            cleaned_traces = cleaned_traces[:, poi_indices]
            print(f"POI selection completed. New trace length: {cleaned_traces.shape[1]} samples")
        
        return cleaned_traces
    
    def _calculate_poi_snr(self, traces, num_poi):
        """
        Calculate Points of Interest using Signal-to-Noise Ratio (SNR) analysis
        
        Parameters:
        -----------
        traces : np.ndarray
            Cleaned traces to analyze (n_traces, n_samples)
        num_poi : int
            Number of top POIs to select
            
        Returns:
        --------
        np.ndarray
            Indices of the top POIs based on SNR
        """
        print("Calculating SNR for POI selection...")
        
        # Calculate signal (variance across traces) and noise (mean within-trace variance)
        signal_variance = np.var(traces, axis=0)  # Variance across traces for each time point
        noise_variance = np.mean(np.var(traces, axis=1))  # Average within-trace variance
        
        # Calculate SNR for each time point
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        snr = signal_variance / (noise_variance + epsilon)
        
        # Alternative SNR calculation using standard deviation ratio
        signal_std = np.std(traces, axis=0)
        noise_std = np.mean([np.std(trace) for trace in traces])
        snr_alt = signal_std / (noise_std + epsilon)
        
        # Use the standard variance-based SNR
        snr_scores = snr
        
        # Get indices of top POIs
        poi_indices = np.argsort(snr_scores)[-num_poi:]  # Top num_poi indices
        poi_indices = np.sort(poi_indices)  # Sort to maintain temporal order
        
        print(f"SNR analysis completed:")
        print(f"  - Signal variance range: {np.min(signal_variance):.6f} to {np.max(signal_variance):.6f}")
        print(f"  - Average noise variance: {noise_variance:.6f}")
        print(f"  - SNR range: {np.min(snr_scores):.6f} to {np.max(snr_scores):.6f}")
        print(f"  - Selected {len(poi_indices)} POIs with SNR range: {snr_scores[poi_indices[0]]:.6f} to {snr_scores[poi_indices[-1]]:.6f}")
        
        return poi_indices
    
    def select_traces_by_noise(self, traces, labels, plaintexts, num_traces):
        """
        Select traces with the smallest standard deviation (cleanest traces)
        
        Parameters:
        -----------
        traces : np.ndarray
            Traces to select from
        labels : np.ndarray
            Corresponding labels
        plaintexts : np.ndarray
            Corresponding plaintexts
        num_traces : int
            Number of traces to select
            
        Returns:
        --------
        tuple
            Selected traces, labels, and plaintexts with lowest noise levels
        """
        # Calculate noise level (standard deviation) for each trace
        noise_levels = np.std(traces, axis=1)
        
        print(f"Noise levels - Min: {np.min(noise_levels):.4f}, "
              f"Max: {np.max(noise_levels):.4f}, "
              f"Mean: {np.mean(noise_levels):.4f}")
        
        # Get indices of traces with lowest noise (cleanest traces)
        best_indices = np.argsort(noise_levels)[:num_traces]
        
        print(f"Selected traces with noise levels from {noise_levels[best_indices[0]]:.4f} "
              f"to {noise_levels[best_indices[-1]]:.4f}")
        
        return traces[best_indices], labels[best_indices], plaintexts[best_indices]

    def split_attack_set_validation_test(self):
        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val = train_test_split(self.X_attack,self.Y_attack,test_size=0.1,random_state=0)



    def choose_phase(self,phase):
        if phase == 'train':
            self.X, self.Y = np.expand_dims(self.X_profiling, 1), self.Y_profiling
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