# src/dataloader.py

import os # For path manipulation
import numpy as np # For numerical operations
from sklearn.preprocessing import StandardScaler # For Z-score normalization
from torch.utils.data import Dataset # Base class for PyTorch datasets
from sklearn.model_selection import train_test_split # For splitting datasets
from tqdm import tqdm # For progress bars
from src.utils import load_ctf_2025 # Custom utility to load raw data from HDF5
import torch # PyTorch library
from scipy.signal import correlate, convolve # For trace alignment and denoising
from scipy.signal.windows import gaussian # For generating Gaussian filter kernels
import h5py # For reading and writing HDF5 files (for processed data caching)


# --- 1. ToTensor_trace Class ---
class ToTensor_trace(object):
    """Converts NumPy arrays in a sample tuple to PyTorch Tensors."""
    def __call__(self, sample_tuple):
        # Expects sample_tuple to be (trace_ndarray, label_ndarray, plaintext_ndarray)
        trace, label, plaintext = sample_tuple 
        
        return torch.from_numpy(trace).float(), \
               torch.from_numpy(np.array(label)).long(), \
               torch.from_numpy(np.array(plaintext)).long()


# --- 2. Helper Functions for Preprocessing (Static, Callable Outside Class) ---
def _align_traces_cross_correlation_static(traces, reference_trace):
    """
    Aligns a set of traces to a reference trace using a basic cross-correlation peak alignment.
    """
    aligned_traces = np.zeros_like(traces, dtype=traces.dtype) 
    ref_len = len(reference_trace)
    
    ref_peak_idx = np.argmax(reference_trace) 

    for i, trace in tqdm(enumerate(traces), total=len(traces), desc="Aligning Traces"):
        trace_peak_idx = np.argmax(trace)
        shift = ref_peak_idx - trace_peak_idx

        if shift > 0: 
            padded_trace = np.pad(trace, (shift, 0), mode='edge')
            aligned_trace = padded_trace[:ref_len] 
        elif shift < 0: 
            padded_trace = np.pad(trace, (0, -shift), mode='edge')
            aligned_trace = padded_trace[-ref_len:]
        else: 
            aligned_trace = trace
        
        if len(aligned_trace) != ref_len:
            if len(aligned_trace) > ref_len:
                aligned_trace = aligned_trace[:ref_len]
            else: 
                aligned_trace = np.pad(aligned_trace, (0, ref_len - len(aligned_trace)), mode='edge')

        aligned_traces[i] = aligned_trace

    return aligned_traces

def _apply_denoising_filter_static(traces, filter_type, window_size):
    """Applies a specified denoising filter (Gaussian or Moving Average) to traces."""
    denoised_traces = np.zeros_like(traces, dtype=traces.dtype) 
    
    if filter_type == 'gaussian':
        sigma = (window_size - 1) / (2 * 2.355) 
        if sigma <= 0: sigma = 1.0 
        kernel = gaussian(window_size, std=sigma)
        kernel /= np.sum(kernel) 
        
        for i, trace in tqdm(enumerate(traces), total=len(traces), desc=f"Applying Gaussian Filter (Window={window_size})"):
            denoised_traces[i] = convolve(trace, kernel, mode='same') 
    elif filter_type == 'moving_average':
        kernel = np.ones(window_size) / window_size 
        for i, trace in tqdm(enumerate(traces), total=len(traces), desc=f"Applying Moving Average (Window={window_size})"):
            denoised_traces[i] = convolve(trace, kernel, mode='same')
    else:
        print(f"Warning: Unknown denoising filter type '{filter_type}'. No filter applied.")
        return traces 
    return denoised_traces

# --- 3. Data Loading and Preprocessing with Caching Function ---
def _load_and_preprocess_data_once(filename, leakage_model, byte,
                                     train_begin, train_end, test_begin, test_end,
                                     trace_start_point, trace_end_point,
                                     denoising_filter_type, denoising_filter_window,
                                     align_traces, alignment_ref_trace_raw=None, 
                                     processed_data_filepath="processed_ches_data.h5"):
    """
    Orchestrates the data loading and preprocessing pipeline.
    It first attempts to load previously processed data from `processed_data_filepath`.
    If not found or corrupted, it loads raw data, applies alignment, denoising, POI cropping,
    and standardization, then saves the processed data for future runs.
    """
    
    # --- Attempt to load pre-processed data from file ---
    if os.path.exists(processed_data_filepath):
        print(f"Attempting to load pre-processed data from {processed_data_filepath}...")
        try:
            with h5py.File(processed_data_filepath, 'r') as f:
                X_profiling_processed = f['X_profiling_processed'][()]
                X_attack_processed = f['X_attack_processed'][()]
                Y_profiling_raw = f['Y_profiling_raw'][()]
                Y_attack_raw = f['Y_attack_raw'][()]
                P_profiling_raw = f['P_profiling_raw'][()]
                P_attack_raw = f['P_attack_raw'][()]
                correct_key = f['correct_key'][()]
            print("Pre-processed data loaded successfully.")
            return (X_profiling_processed, X_attack_processed), \
                   (Y_profiling_raw, Y_attack_raw), \
                   (P_profiling_raw, P_attack_raw), \
                   correct_key
        except Exception as e:
            print(f"Error loading pre-processed data: {e}. Re-processing data...")
            if os.path.exists(processed_data_filepath): 
                os.remove(processed_data_filepath)

    # --- If pre-processed file not found or corrupted, load raw and process ---
    print("Pre-processed data not found. Starting raw data loading and full preprocessing pipeline...")
    
    (X_profiling_raw, X_attack_raw), \
    (Y_profiling_raw, Y_attack_raw), \
    (P_profiling_raw, P_attack_raw), \
    correct_key = load_ctf_2025(filename, leakage_model, byte,
                                train_begin, train_end, test_begin, test_end)
    
    actual_alignment_ref_trace = alignment_ref_trace_raw
    if align_traces and actual_alignment_ref_trace is None:
        actual_alignment_ref_trace = X_profiling_raw[0].copy()
        print("Using first profiling trace as alignment reference.")

    # --- Preprocessing Pipeline Steps ---

    # 1. Apply Trace Alignment (Cross-Correlation)
    if align_traces:
        print("Performing trace alignment (cross-correlation)...")
        X_profiling_aligned = _align_traces_cross_correlation_static(X_profiling_raw, actual_alignment_ref_trace)
        X_attack_aligned = _align_traces_cross_correlation_static(X_attack_raw, actual_alignment_ref_trace)
        print("Trace alignment completed.")
    else:
        X_profiling_aligned = X_profiling_raw
        X_attack_aligned = X_attack_raw

    # 2. Apply Denoising Filter
    if denoising_filter_type:
        print(f"Applying {denoising_filter_type} denoising with window {denoising_filter_window}...")
        X_profiling_denoised = _apply_denoising_filter_static(X_profiling_aligned, denoising_filter_type, denoising_filter_window)
        X_attack_denoised = _apply_denoising_filter_static(X_attack_aligned, denoising_filter_type, denoising_filter_window)
        print("Denoising completed.")
    else:
        X_profiling_denoised = X_profiling_aligned
        X_attack_denoised = X_attack_aligned
            
    # 3. Apply POI (Points of Interest) Cropping
    current_trace_length = X_profiling_denoised.shape[1]
    effective_end_point = trace_end_point if trace_end_point is not None else current_trace_length
    
    if trace_start_point != 0 or effective_end_point != current_trace_length:
        print(f"Applying POI cropping from index {trace_start_point} to {effective_end_point}...")
        if effective_end_point > current_trace_length or trace_start_point < 0 or trace_start_point >= effective_end_point:
            raise ValueError(f"Invalid POI cropping points: start={trace_start_point}, end={effective_end_point}. "
                             f"Current trace length after denoising/alignment is {current_trace_length}.")
        X_profiling_cropped = X_profiling_denoised[:, trace_start_point:effective_end_point]
        X_attack_cropped = X_attack_denoised[:, trace_start_point:effective_end_point]
        print(f"POI cropping completed. New trace length: {X_profiling_cropped.shape[1]} samples.")
    else:
        X_profiling_cropped = X_profiling_denoised
        X_attack_cropped = X_attack_denoised

    # 4. Apply StandardScaler (Normalization)
    scaler_std = StandardScaler()
    X_profiling_processed = scaler_std.fit_transform(X_profiling_cropped)
    X_attack_processed = scaler_std.transform(X_attack_cropped) 
    
    # --- Save processed data to file ---
    print(f"Saving processed data to {processed_data_filepath}...")
    with h5py.File(processed_data_filepath, 'w') as f:
        f.create_dataset('X_profiling_processed', data=X_profiling_processed)
        f.create_dataset('X_attack_processed', data=X_attack_processed)
        f.create_dataset('Y_profiling_raw', data=Y_profiling_raw)
        f.create_dataset('Y_attack_raw', data=Y_attack_raw)
        f.create_dataset('P_profiling_raw', data=P_profiling_raw)
        f.create_dataset('P_attack_raw', data=P_attack_raw)
        f.create_dataset('correct_key', data=correct_key)
    print("Processed data saved.")

    return (X_profiling_processed, X_attack_processed), \
           (Y_profiling_raw, Y_attack_raw), \
           (P_profiling_raw, P_attack_raw), \
           correct_key


# --- 4. Custom_Dataset Class ---
# This class wraps the processed data (from _load_and_preprocess_data_once)
# and implements PyTorch's `Dataset` interface for use with `DataLoader`.
class Custom_Dataset(Dataset):
    def __init__(self, root='./', dataset="CHES_2025", leakage="HW", transform=None,
                 trace_start_point=0, trace_end_point=None, 
                 denoising_filter_type=None, denoising_filter_window=3,
                 align_traces=False, alignment_ref_trace=None,
                 processed_data_filepath="processed_ches_data.h5",
                 # --- NEW: Parameters to directly provide pre-loaded data ---
                 X_profiling_preloaded=None, X_attack_preloaded=None,
                 Y_profiling_preloaded=None, Y_attack_preloaded=None,
                 P_profiling_preloaded=None, P_attack_preloaded=None,
                 correct_key_preloaded=None):
        """
        Initializes the dataset. It either loads/preprocesses data (and caches it)
        OR directly uses pre-loaded data arrays to avoid redundant loading/processing.

        Args:
            root (str): Base path for dataset files.
            dataset (str): Name of the dataset (e.g., "CHES_2025").
            leakage (str): Leakage model ("ID" or "HW").
            transform (callable, optional): PyTorch transform (e.g., ToTensor_trace) for samples.
            trace_start_point, trace_end_point, denoising_filter_type, denoising_filter_window,
            align_traces, alignment_ref_trace (various types): Parameters for preprocessing,
                                                              passed directly to `_load_and_preprocess_data_once`.
            processed_data_filepath (str): Full path to the file where processed data is cached.
            X_profiling_preloaded, X_attack_preloaded, ...: Optional pre-loaded NumPy arrays.
                                                             If provided, _load_and_preprocess_data_once is skipped.
        """
        self.transform = transform
        self.root = root
        self.dataset_name = dataset
        self.leakage = leakage
        
        # --- Check if data is pre-loaded; if so, use it directly ---
        if X_profiling_preloaded is not None and X_attack_preloaded is not None:
            print("Custom_Dataset initialized with pre-loaded data (skipping full preprocessing).")
            self.X_profiling = X_profiling_preloaded
            self.X_attack = X_attack_preloaded
            self.Y_profiling = Y_profiling_preloaded
            self.Y_attack = Y_attack_preloaded
            self.plt_profiling = P_profiling_preloaded
            self.plt_attack = P_attack_preloaded
            self.correct_key = correct_key_preloaded
        else:
            # --- Otherwise, perform full load and preprocessing pipeline ---
            if dataset == "CHES_2025":
                byte = 0 # Target the first byte (byte 0) as per challenge rules
                data_root_path = os.path.join(root, 'Dataset/CHES_2025/CHES_Challenge.h5')
                
                (self.X_profiling, self.X_attack), \
                (self.Y_profiling, self.Y_attack), \
                (self.plt_profiling, self.plt_attack), \
                self.correct_key = _load_and_preprocess_data_once(
                    data_root_path, self.leakage, byte,
                    train_begin=0, train_end=450000, # Max profiling traces from original dataset
                    test_begin=0, test_end=100000, # Max attack traces from original dataset
                    trace_start_point=trace_start_point, # Pass preprocessing parameters
                    trace_end_point=trace_end_point,
                    denoising_filter_type=denoising_filter_type,
                    denoising_filter_window=denoising_filter_window,
                    align_traces=align_traces,
                    alignment_ref_trace_raw=alignment_ref_trace, 
                    processed_data_filepath=processed_data_filepath 
                )
            else:
                raise ValueError(f"Dataset '{dataset}' not supported.")

        print(f"Custom_Dataset initialized. Profiling traces: {len(self.X_profiling)}, Attack traces: {len(self.X_attack)}")
        print(f"Trace length after all preprocessing: {self.X_profiling.shape[1]} samples.")


    def split_attack_set_validation_test(self, test_size=0.1, random_state=0):
        """
        Splits the *processed* `X_attack` data (and its corresponding labels/plaintexts)
        into validation and test subsets. This split is used for local validation during training
        and for local testing, separate from the primary profiling set.
        """
        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val, \
        self.plt_attack_test, self.plt_attack_val = train_test_split(
            self.X_attack, self.Y_attack, self.plt_attack, test_size=test_size, random_state=random_state
        )
        print(f"Attack set split: Test size {len(self.X_attack_test)}, Validation size {len(self.X_attack_val)}")


    def choose_phase(self, phase):
        """
        Sets the internal pointers (`self.X_current`, etc.) to the data subset 
        corresponding to the specified `phase`. This is how `DataLoader` accesses data.
        """
        if phase == 'train':
            self.X_current = np.expand_dims(self.X_profiling, 1) # Add channel dimension (e.g., (N, 1, L))
            self.Y_current = self.Y_profiling
            self.Plaintext_current = self.plt_profiling
        elif phase == 'validation':
            self.X_current = np.expand_dims(self.X_attack_val, 1)
            self.Y_current = self.Y_attack_val
            self.Plaintext_current = self.plt_attack_val
        elif phase == 'test':
            self.X_current = np.expand_dims(self.X_attack_test, 1)
            self.Y_current = self.Y_attack_test
            self.Plaintext_current = self.plt_attack_test
        else:
            raise ValueError(f"Invalid phase: {phase}. Choose 'train', 'validation', or 'test'.")

        print(f"Dataset active phase: '{phase}' with {len(self.X_current)} samples. Current trace length: {self.X_current.shape[-1]} samples.")

    def __len__(self):
        """Returns the number of samples in the current active phase."""
        return len(self.X_current)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (trace, label, plaintext) for the given index `idx`.
        Applies `self.transform` (ToTensor_trace) if it's set during initialization.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        trace = self.X_current[idx]
        sensitive = self.Y_current[idx]
        plaintext = self.Plaintext_current[idx] 
        
        # Return as a tuple of NumPy arrays. `ToTensor_trace` will then convert these to PyTorch Tensors.
        sample_tuple = (trace, sensitive, plaintext)
        
        if self.transform:
            return self.transform(sample_tuple)
        
        return sample_tuple 