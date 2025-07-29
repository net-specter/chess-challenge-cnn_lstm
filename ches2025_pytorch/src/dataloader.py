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
                 roi_extraction=True,
                 roi_percentile=95,
                 roi_width=200,
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
                roi_extraction=roi_extraction,
                roi_percentile=roi_percentile,
                roi_width=roi_width,
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
                roi_extraction=roi_extraction,
                roi_percentile=roi_percentile,
                roi_width=roi_width,
                apply_median_filter=apply_median_filter,
                median_window=median_window,
                apply_butterworth=apply_butterworth,
                butterworth_cutoff=butterworth_cutoff,
                butterworth_order=butterworth_order
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
                    roi_extraction=True, roi_percentile=95, roi_width=100,
                    apply_median_filter=True, median_window=3,
                    apply_butterworth=False, butterworth_cutoff=0.1, butterworth_order=3):
        """
        Apply comprehensive signal processing techniques to clean traces
        
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
        roi_extraction : bool
            Whether to extract regions of interest based on variance
        roi_percentile : float
            Percentile to use for identifying POIs (0-100)
        roi_width : int
            Width of region around POI to extract
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
            
        Returns:
        --------
        np.ndarray
            Cleaned traces
        """
        """Step 1: Applying Moving Average Filter for denoising..."""
        cleaned_traces = traces.copy()
        
        # 1. Moving Average Filter (uniform_filter1d) - Denoise by smoothing fluctuations
        cleaned_traces = np.array([uniform_filter1d(trace, size=moving_avg_window) 
                                 for trace in cleaned_traces])
        
        """Step 2: Applying Savitzky-Golay Filter for advanced smoothing..."""
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
            """Step 3: Applying Median Filter for noise reduction..."""
            cleaned_traces = np.array([medfilt(trace, kernel_size=median_window) 
                                     for trace in cleaned_traces])
        
        # 4. Apply Butterworth Filtering for frequency-based noise reduction
        if apply_butterworth:
            """Step 4: Applying Butterworth Filter for frequency-based noise reduction..."""
            nyq = 0.5  # Nyquist frequency (assuming normalized frequency)
            normal_cutoff = butterworth_cutoff / nyq
            b, a = butter(butterworth_order, normal_cutoff, btype='low', analog=False)
            cleaned_traces = np.array([filtfilt(b, a, trace) for trace in cleaned_traces])
        
        # 5. Standard Deviation Analysis to identify Points of Interest (POI)
        if roi_extraction:
            """Step 5: Analyzing standard deviation to identify Points of Interest..."""
            std_dev = np.std(cleaned_traces, axis=0)
            
            # Find points with highest standard deviation (most informative)
            poi_threshold = np.percentile(std_dev, roi_percentile)
            poi_indices = np.where(std_dev >= poi_threshold)[0]
            
            print(f"Found {len(poi_indices)} points of interest above {roi_percentile}th percentile")
            
            if len(poi_indices) > 0:
                # Find clusters of POIs to identify coherent regions
                poi_clusters = self._find_poi_clusters(poi_indices, roi_width)
                
                # Extract the most informative region
                if poi_clusters:
                    # Select cluster with highest mean standard deviation
                    cluster_std_means = [np.mean(std_dev[cluster]) for cluster in poi_clusters]
                    best_cluster_idx = np.argmax(cluster_std_means)
                    best_cluster = poi_clusters[best_cluster_idx]
                    
                    # Extract region around the center of best cluster
                    center = int(np.mean(best_cluster))
                    start = max(0, center - roi_width//2)
                    end = min(traces.shape[1], center + roi_width//2)
                    
                    print(f"Extracting ROI: samples {start} to {end} (center at {center})")
                    cleaned_traces = cleaned_traces[:, start:end]
                else:
                    print("No coherent POI clusters found, keeping full traces")
            else:
                print("No POIs found above threshold, keeping full traces")
                
        return cleaned_traces
    
    def _find_poi_clusters(self, poi_indices, roi_width):
        """
        Find clusters of Points of Interest
        
        Parameters:
        -----------
        poi_indices : np.ndarray
            Indices of points of interest
        roi_width : int
            Maximum distance between points to be in same cluster
            
        Returns:
        --------
        list
            List of POI clusters
        """
        if len(poi_indices) == 0:
            return []
            
        poi_clusters = []
        current_cluster = [poi_indices[0]]
        
        for i in range(1, len(poi_indices)):
            # If current point is close to previous, add to current cluster
            if poi_indices[i] - poi_indices[i-1] <= roi_width//4:
                current_cluster.append(poi_indices[i])
            else:
                # Start a new cluster
                if len(current_cluster) >= 3:  # Only keep clusters with multiple points
                    poi_clusters.append(current_cluster)
                current_cluster = [poi_indices[i]]
        
        # Add the last cluster if it's significant
        if len(current_cluster) >= 3:
            poi_clusters.append(current_cluster)
            
        return poi_clusters
    
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