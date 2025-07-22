import os
import numpy as np
from sklearn.preprocessing import StandardScaler # Only StandardScaler is used
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.utils import load_ctf_2025 # Import the data loading utility
import torch
from scipy.signal import correlate, convolve # For trace alignment and denoising
from scipy.signal.windows import gaussian # Fix import for gaussian window

class ToTensor_trace(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # Handle both 2-tuple and 3-tuple returns from __getitem__
        if len(sample) == 3:
            trace, label, plaintext = sample
            return torch.from_numpy(trace).float(), \
                   torch.from_numpy(np.array(label)).long(), \
                   torch.from_numpy(np.array(plaintext)).long()
        else:  # Legacy 2-tuple support
            trace, label = sample
            return torch.from_numpy(trace).float(), \
                   torch.from_numpy(np.array(label)).long()

class Custom_Dataset(Dataset):
    def __init__(self, root='./', dataset="CHES_2025", leakage="HW", transform=None,
                 trace_start_point=0, trace_end_point=None,
                 denoising_filter_type=None, denoising_filter_window=3,
                 align_traces=False, alignment_ref_trace=None):
        """
        Initializes the dataset, loads data, applies SCA preprocessing (POI, Denoising, Alignment),
        and finally StandardScaler.

        Args:
            root (str): Base path for dataset files.
            dataset (str): Name of the dataset (e.g., "CHES_2025").
            leakage (str): Leakage model ("ID" or "HW").
            transform (callable, optional): Optional transform to be applied on a sample.
            trace_start_point (int): Start index for trace cropping (POI).
            trace_end_point (int): End index for trace cropping (POI). If None, uses full length from start_point.
            denoising_filter_type (str, optional): Type of denoising filter ('gaussian', 'moving_average'). None for no denoising.
            denoising_filter_window (int): Window size for denoising filter.
            align_traces (bool): Whether to perform trace alignment.
            alignment_ref_trace (np.ndarray, optional): A reference trace for alignment. If None and align_traces is True,
                                                         the first profiling trace will be used as reference.
        """
        self.transform = transform
        self.root = root
        self.dataset_name = dataset
        self.leakage = leakage
        self.trace_start_point = trace_start_point
        self.trace_end_point = trace_end_point
        self.denoising_filter_type = denoising_filter_type
        self.denoising_filter_window = denoising_filter_window
        self.align_traces = align_traces
        self._alignment_ref_trace = alignment_ref_trace # Store internally

        if dataset == "CHES_2025":
            byte = 0 # Target the first byte (byte 0) as per challenge
            data_root_path = os.path.join(root, 'Dataset/CHES_2025/CHES_Challenge.h5')
            
            # Load all data using the utility function
            (X_profiling_raw, X_attack_raw), \
            (Y_profiling_raw, Y_attack_raw), \
            (P_profiling_raw, P_attack_raw), \
            self.correct_key = load_ctf_2025(
                data_root_path, leakage_model=self.leakage, byte=byte,
                train_begin=0, train_end=45000, # Use more profiling traces if available (e.g., 450K out of 500K)
                test_begin=0, test_end=10000 # Use all 100K public attack traces
            )
        else:
            raise ValueError(f"Dataset '{dataset}' not supported.")

        print(f"Loaded dataset from: {data_root_path}")
        
        # --- Preprocessing Pipeline ---
        
        # 0. Set alignment reference trace if needed
        if self.align_traces and self._alignment_ref_trace is None:
            # Use the first profiling trace as reference if not provided
            self._alignment_ref_trace = X_profiling_raw[0].copy()
            print("Using first profiling trace as alignment reference.")

        # 1. Apply Trace Alignment (Cross-Correlation)
        if self.align_traces:
            print("Performing trace alignment...")
            X_profiling_aligned = self._align_traces_cross_correlation(X_profiling_raw, self._alignment_ref_trace)
            X_attack_aligned = self._align_traces_cross_correlation(X_attack_raw, self._alignment_ref_trace)
            print("Trace alignment completed.")
        else:
            X_profiling_aligned = X_profiling_raw
            X_attack_aligned = X_attack_raw

        # 2. Apply Denoising Filter
        if self.denoising_filter_type:
            print(f"Applying {self.denoising_filter_type} denoising with window {self.denoising_filter_window}...")
            X_profiling_denoised = self._apply_denoising_filter(X_profiling_aligned)
            X_attack_denoised = self._apply_denoising_filter(X_attack_aligned)
            print("Denoising completed.")
        else:
            X_profiling_denoised = X_profiling_aligned
            X_attack_denoised = X_attack_aligned
            
        # 3. Apply POI Cropping
        current_trace_length = X_profiling_denoised.shape[1]
        effective_end_point = self.trace_end_point if self.trace_end_point is not None else current_trace_length
        
        if self.trace_start_point != 0 or effective_end_point != current_trace_length:
            print(f"Applying POI cropping from index {self.trace_start_point} to {effective_end_point}...")
            if effective_end_point > current_trace_length or self.trace_start_point < 0:
                raise ValueError("POI cropping points are out of original trace bounds.")
            X_profiling_cropped = X_profiling_denoised[:, self.trace_start_point:effective_end_point]
            X_attack_cropped = X_attack_denoised[:, self.trace_start_point:effective_end_point]
            print(f"POI cropping completed. New trace length: {X_profiling_cropped.shape[1]}")
        else:
            X_profiling_cropped = X_profiling_denoised
            X_attack_cropped = X_attack_denoised

        # 4. Apply StandardScaler (Normalization)
        self.scaler_std = StandardScaler()
        self.X_profiling = self.scaler_std.fit_transform(X_profiling_cropped)
        self.X_attack = self.scaler_std.transform(X_attack_cropped) # Transform attack data using profiling data's scaler
        
        self.Y_profiling = Y_profiling_raw # Labels already processed by load_ctf_2025 for leakage model
        self.Y_attack = Y_attack_raw
        self.plt_profiling = P_profiling_raw
        self.plt_attack = P_attack_raw

    def _align_traces_cross_correlation(self, traces, reference_trace):
        """Aligns a set of traces to a reference trace using cross-correlation."""
        aligned_traces = np.zeros_like(traces)
        ref_len = len(reference_trace)
        
        # If the reference trace is cropped, apply the same cropping for correlation
        if self.trace_end_point is not None and self.trace_end_point < ref_len:
             ref_for_corr = reference_trace[self.trace_start_point:self.trace_end_point]
        else:
             ref_for_corr = reference_trace[self.trace_start_point:]
        
        if len(ref_for_corr) == 0:
            print("Warning: Reference trace for alignment is empty after cropping. Skipping alignment.")
            return traces # Return original if reference is invalid

        for i, trace in tqdm(enumerate(traces), total=len(traces), desc="Aligning Traces"):
            # Correlate the trace with the reference
            correlation = correlate(trace, ref_for_corr, mode='full')
            
            # Find the lag (shift) that maximizes the correlation
            # The peak of the correlation function indicates the optimal shift.
            # The indices of the full correlation output range from -(ref_len-1) to (trace_len-1).
            # The middle is where trace and ref are perfectly aligned in their start.
            
            # We want the shift relative to the start of the trace.
            # If peak_idx < ref_len -1, trace shifted left, positive alignment_shift
            # If peak_idx > ref_len -1, trace shifted right, negative alignment_shift
            
            # Index where correlation is maximized
            peak_idx = np.argmax(correlation)
            
            # Calculate the shift needed to align the trace to the reference's start point
            # The 'full' mode of correlate produces an array of length N + M - 1.
            # A shift of 0 corresponds to index M-1 in the correlation array if N=M
            # More generally, (len(trace) - 1) is the index for a perfect match if both start at 0
            
            # Offset calculation needs to be precise.
            # The index in `correlation` corresponds to `lag + (len(reference_trace) - 1)`
            # So, `lag = peak_idx - (len(reference_trace) - 1)`
            
            # However, for aligning the *start* of the trace, a simpler approach is to align their
            # highest correlation points.
            
            # Calculate the actual shift needed
            # The offset of the peak in correlation array relative to its center gives the shift.
            # The center of the correlation array is (len(correlation) - 1) / 2
            # shift = peak_idx - (len(trace) - 1)
            
            # To simply align the peak of `trace` to the peak of `reference_trace`:
            # Find peak in current trace
            trace_peak_idx = np.argmax(trace)
            ref_peak_idx = np.argmax(reference_trace) # Assuming ref_trace is already 'aligned' or is the template
            
            # Simpler alignment (aligning peaks):
            shift = ref_peak_idx - trace_peak_idx
            
            # Apply the shift
            if shift > 0: # Shift trace right (pad left)
                aligned_trace = np.pad(trace, (shift, 0), mode='edge')[:ref_len] # Pad left, crop to original length
            elif shift < 0: # Shift trace left (pad right)
                aligned_trace = np.pad(trace, (0, -shift), mode='edge')[-ref_len:] # Pad right, crop to original length
            else:
                aligned_trace = trace
            
            # Ensure output is same length as original input or reference
            aligned_traces[i] = aligned_trace[:ref_len] if len(aligned_trace) > ref_len else np.pad(aligned_trace, (0, ref_len - len(aligned_trace)), mode='edge')


        return aligned_traces

    def _apply_denoising_filter(self, traces):
        """Applies a specified denoising filter to traces."""
        denoised_traces = np.zeros_like(traces)
        if self.denoising_filter_type == 'gaussian':
            # Gaussian filter window is often an odd integer, converted to std dev for scipy.gaussian
            sigma = (self.denoising_filter_window - 1) / (2 * 2.355) # ~ (window-1)/6 for 99.7% of data
            # Create a 1D Gaussian kernel
            kernel = gaussian(self.denoising_filter_window, std=sigma)
            kernel /= np.sum(kernel) # Normalize kernel to sum to 1

            for i, trace in tqdm(enumerate(traces), total=len(traces), desc="Applying Gaussian Filter"):
                denoised_traces[i] = convolve(trace, kernel, mode='same') # 'same' keeps output size same
        elif self.denoising_filter_type == 'moving_average':
            kernel = np.ones(self.denoising_filter_window) / self.denoising_filter_window
            for i, trace in tqdm(enumerate(traces), total=len(traces), desc="Applying Moving Average"):
                denoised_traces[i] = convolve(trace, kernel, mode='same')
        else:
            print(f"Warning: Unknown denoising filter type '{self.denoising_filter_type}'. No filter applied.")
            return traces # Return original if type is not recognized
        return denoised_traces

    def split_attack_set_validation_test(self, test_size=0.1, random_state=0):
        """
        Splits the loaded X_attack and Y_attack into validation and test subsets.
        This uses the *public* attack traces for local validation during training.
        """
        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val, \
        self.plt_attack_test, self.plt_attack_val = train_test_split(
            self.X_attack, self.Y_attack, self.plt_attack, test_size=test_size, random_state=random_state
        )
        print(f"Attack set split: Test size {len(self.X_attack_test)}, Validation size {len(self.X_attack_val)}")


    def choose_phase(self, phase):
        """
        Selects which subset of data (profiling or attack split) the dataset will expose.
        Args:
            phase (str): 'train' (profiling data), 'validation' (attack_val split), or 'test' (attack_test split).
        """
        if phase == 'train':
            self.X_current = np.expand_dims(self.X_profiling, 1) # Add channel dimension
            self.Y_current = self.Y_profiling
            self.Plaintext_current = self.plt_profiling
        elif phase == 'validation':
            self.X_current = np.expand_dims(self.X_attack_val, 1) # Add channel dimension
            self.Y_current = self.Y_attack_val
            self.Plaintext_current = self.plt_attack_val
        elif phase == 'test':
            self.X_current = np.expand_dims(self.X_attack_test, 1) # Add channel dimension
            self.Y_current = self.Y_attack_test
            self.Plaintext_current = self.plt_attack_test
        else:
            raise ValueError(f"Invalid phase: {phase}. Choose 'train', 'validation', or 'test'.")

        print(f"Dataset active phase: '{phase}' with {len(self.X_current)} samples. Current trace length: {self.X_current.shape[-1]}")

    def __len__(self):
        return len(self.X_current)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X_current[idx]
        sensitive = self.Y_current[idx]
        plaintext = self.Plaintext_current[idx] 
        
        return torch.from_numpy(trace).float(), \
               torch.from_numpy(np.array(sensitive)).long(), \
               torch.from_numpy(np.array(plaintext)).long()