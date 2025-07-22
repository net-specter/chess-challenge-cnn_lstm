import math
import random

import h5py
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader # Added for evaluate_fast

# AES S-boxes
AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
AES_Sbox_inv =  np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

# Hamming Weight lookup table (global for efficiency)
HW_lookup_table = np.array([bin(x).count("1") for x in range(256)])

def calculate_HW(data):
    """Calculates Hamming Weight for a list of values."""
    return [HW_lookup_table[int(s)] for s in data]

def load_ctf_2025(filename, leakage_model='HW', byte=0, train_begin=0, train_end=100000, test_begin=0, test_end=50000):
    """
    Loads data from the CHES 2025 HDF5 dataset.
    Generates labels based on the specified leakage model.
    """
    in_file = h5py.File(filename, "r")

    # Load profiling traces and plaintext
    X_profiling_raw = np.array(in_file['Profiling_traces/traces'])
    X_profiling_raw = X_profiling_raw.reshape((X_profiling_raw.shape[0], X_profiling_raw.shape[1]))
    P_profiling_raw = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, byte])
    
    # Generate profiling labels (Y_profiling)
    if byte != 0:
        # If targeting a byte other than 0, key is read per trace
        key_profiling_per_trace = np.array(in_file['Profiling_traces/metadata'][:]['key'][:, byte])
        Y_profiling_raw = np.zeros(P_profiling_raw.shape[0], dtype=int)
        print(f"Generating Y_profiling for byte {byte}...")
        for i in tqdm(range(len(P_profiling_raw)), desc="Generating Profiling Labels"):
            Y_profiling_raw[i] = AES_Sbox[P_profiling_raw[i] ^ key_profiling_per_trace[i]]
    else:
        # For byte 0, labels are directly available (common in ASCAD/similar datasets)
        Y_profiling_raw = np.array(in_file['Profiling_traces/metadata'][:]['labels'], dtype=int)
    
    if leakage_model == 'HW':
        Y_profiling_raw = calculate_HW(Y_profiling_raw)

    # Load attack traces and plaintext
    X_attack_raw = np.array(in_file['Attack_traces/traces'])
    X_attack_raw = X_attack_raw.reshape((X_attack_raw.shape[0], X_attack_raw.shape[1]))
    P_attack_raw = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, byte])
    
    # Get the real key for the attack set (note that attack key is fixed for evaluation)
    attack_key = np.array(in_file['Attack_traces/metadata'][:]['key'][0, byte], dtype=int) 
    # Profiling key (first element) can be useful for debugging/consistency checks
    profiling_key = np.array(in_file['Profiling_traces/metadata'][:]['key'][0, byte], dtype=int) 
    
    print(f"Attack Key for Byte {byte}: {attack_key}")
    print(f"Profiling Key for Byte {byte}: {profiling_key}")

    # For attack traces, labels are usually not known during a real attack.
    # Y_attack_raw here serves as the ground truth for evaluation functions like `evaluate_fast`.
    if byte != 0:
        key_attack_per_trace = np.array(in_file['Attack_traces/metadata'][:]['key'][:, byte])
        Y_attack_raw = np.zeros(P_attack_raw.shape[0], dtype=int)
        print(f"Generating Y_attack for byte {byte}...")
        for i in tqdm(range(len(P_attack_raw)), desc="Generating Attack Labels"):
            Y_attack_raw[i] = AES_Sbox[P_attack_raw[i] ^ key_attack_per_trace[i]]
    else:
        Y_attack_raw = np.array(in_file['Attack_traces/metadata'][:]['labels'], dtype=int)
    
    if leakage_model == 'HW':
        Y_attack_raw = calculate_HW(Y_attack_raw)

    print("Information about the dataset: ")
    print(f"X_profiling raw shape: {X_profiling_raw.shape}")
    print(f"Y_profiling raw shape: {len(Y_profiling_raw) if leakage_model == 'HW' else Y_profiling_raw.shape}")
    print(f"P_profiling raw shape: {P_profiling_raw.shape}")
    print(f"X_attack raw shape: {X_attack_raw.shape}")
    print(f"Y_attack raw shape: {len(Y_attack_raw) if leakage_model == 'HW' else Y_attack_raw.shape}")
    print(f"P_attack raw shape: {P_attack_raw.shape}")
    print()

    # Slice data based on train_begin/end and test_begin/end
    X_profiling_sliced = X_profiling_raw[train_begin:train_end]
    Y_profiling_sliced = np.array(Y_profiling_raw[train_begin:train_end])
    P_profiling_sliced = P_profiling_raw[train_begin:train_end]

    X_attack_sliced = X_attack_raw[test_begin:test_end]
    Y_attack_sliced = np.array(Y_attack_raw[test_begin:test_end])
    P_attack_sliced = P_attack_raw[test_begin:test_end]

    in_file.close() # Close the h5py file

    return (X_profiling_sliced, X_attack_sliced), \
           (Y_profiling_sliced, Y_attack_sliced), \
           (P_profiling_sliced, P_attack_sliced), \
           attack_key

# Objective: GE metric helper function
def rk_key(rank_array, key):
    """Finds the rank of the correct key in a sorted array of likelihoods."""
    key_val = rank_array[key]
    final_rank = np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])
    if math.isnan(float(final_rank)) or math.isinf(float(final_rank)):
        return np.float32(256) # Max rank for 256 possible keys if not found
    else:
        return np.float32(final_rank)

def rank_compute(prediction_log_proba, attack_plaintexts, correct_key, leakage_fn):
    """
    Computes the evolution of key rank over traces.
    Args:
        prediction_log_proba (np.array): Model's output log probabilities (log_softmax) for each trace. Shape: (num_traces, num_classes)
        attack_plaintexts (np.array): Plaintext byte for each trace.
        correct_key (int): The true key byte for evaluation.
        leakage_fn (function): Function to calculate sensitive value based on plaintext and key guess.
    Returns:
        tuple: (rank_evolution_curve, final_key_log_probabilities)
    """
    nb_traces, nb_hyp = prediction_log_proba.shape # nb_hyp is num_classes (9 or 256)

    key_log_prob = np.zeros(256, dtype=np.float64) # Accumulates log likelihoods for each possible key
    rank_evol = np.full(nb_traces, 255, dtype=np.float32) # Stores rank evolution (default max rank)

    for i in range(nb_traces):
        for k_guess in range(256): # Iterate through all 256 possible key byte guesses
            # Calculate the hypothetical sensitive value based on guessed key and plaintext
            sensitive_value = leakage_fn(attack_plaintexts[i], k_guess)
            
            # Add the log probability of this sensitive value from the model's prediction for current trace
            key_log_prob[k_guess] += prediction_log_proba[i, sensitive_value]
            
        # Update rank evolution at this trace count
        rank_evol[i] = rk_key(key_log_prob, correct_key)

    return rank_evol, key_log_prob

def perform_attacks(nb_traces_to_use, predictions_log_proba, plaintexts_attack, correct_key, leakage_fn, nb_attacks=1, shuffle=True):
    """
    Performs multiple attack experiments to get an average rank evolution.
    Args:
        nb_traces_to_use (int): Number of traces to use for each attack experiment.
        predictions_log_proba (np.array): Model's output log probabilities for all attack traces.
        plaintexts_attack (np.array): Plaintexts for all attack traces.
        correct_key (int): The true key byte.
        leakage_fn (function): Function to calculate sensitive value.
        nb_attacks (int): Number of attack experiments to average over (for Guessing Entropy).
        shuffle (bool): Whether to shuffle traces before each experiment.
    Returns:
        tuple: (average_rank_evolution, final_accumulated_key_log_probabilities_from_last_run)
    """
    all_rk_evol = np.zeros((nb_attacks, nb_traces_to_use), dtype=np.float32)

    for i in tqdm(range(nb_attacks), desc="Performing Attacks"):
        if shuffle:
            # Create shuffled indices
            shuffled_indices = np.random.permutation(len(predictions_log_proba))
            
            # Apply shuffle and select subset of traces
            current_predictions = predictions_log_proba[shuffled_indices[:nb_traces_to_use]]
            current_plaintexts = plaintexts_attack[shuffled_indices[:nb_traces_to_use]]
        else:
            # Use data sequentially
            current_predictions = predictions_log_proba[:nb_traces_to_use]
            current_plaintexts = plaintexts_attack[:nb_traces_to_use]
            
        rank_evol, key_log_prob = rank_compute(current_predictions, current_plaintexts, correct_key, leakage_fn=leakage_fn)
        all_rk_evol[i] = rank_evol
        
        # We return key_log_prob from the LAST run for debugging/inspection, but GE is avg of ranks.
        # This `key_log_prob` is not used in the final `evaluate_fast` GE calculation,
        # which uses a vectorized approach.
    
    return np.mean(all_rk_evol, axis=0), key_log_prob 

def NTGE_fn(GE_curve):
    """
    Calculates NTGE: the minimum number of traces for GE to consistently be 0.
    Iterates backward from the end of the GE curve.
    """
    NTGE = float('inf')
    # Loop backwards. NTGE is the index (number of traces) of the *first* 0 from the end
    # GE_curve might not be strictly monotonically non-increasing due to averaging over limited attacks.
    # To be "consistently 0", it must be 0 from that point to the end.
    
    # Find the last point where GE is 0
    last_zero_idx = -1
    for i in range(len(GE_curve)):
        if GE_curve[i] == 0:
            last_zero_idx = i
        else:
            last_zero_idx = -1 # Reset if it becomes non-zero
            
    if last_zero_idx != -1:
        # If it's consistently zero from `last_zero_idx` to the end
        if np.all(GE_curve[last_zero_idx:] == 0):
            NTGE = last_zero_idx + 1 # Convert to 1-indexed trace count
    
    return NTGE


def evaluate_fast(device, model,
                  X_attack, plt_attack, correct_key,
                  leakage_fn,
                  nb_attacks=100,
                  total_nb_traces_attacks=100000,
                  nb_traces_attacks=100000,
                  batch_size=512):
    """
    A faster, vectorized version for computing GE and NTGE.
    Assumes model returns logits directly (no softmax).
    """
    model.eval() # Set model to evaluation mode

    # ---------- 1. Forward pass in batches to get log probabilities ------------------- #
    ds = TensorDataset(torch.from_numpy(X_attack[:total_nb_traces_attacks]))
    # No shuffle needed for evaluation
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0) # num_workers=0 for Windows

    logp_chunks = []
    with torch.no_grad():
        # Use AMP for inference as well if the model was trained with it
        with torch.amp.autocast('cuda'):
            for (batch_traces,) in tqdm(loader, desc="Model Inference on Attack Traces"):
                batch_traces = batch_traces.to(device).unsqueeze(1).float() # Add channel dim
                logits = model(batch_traces)
                # If model returns a dict (like the original CNN_LSTM_SCA), extract 'output'
                if isinstance(logits, dict) and 'output' in logits:
                    logits = logits['output']
                logp_chunks.append(F.log_softmax(logits, dim=1).cpu())
    logp = torch.cat(logp_chunks).numpy() # (Total_attack_traces, num_classes)
    num_classes = logp.shape[1] # 256 or 9

    # ---------- 2. Prepare containers for attack experiments -------------------------- #
    # Stores accumulated log-likelihood for each of the 256 key candidates, across 'nb_attacks' runs
    key_probs_runs = np.zeros((nb_attacks, 256), dtype=np.float64) 
    # Stores the mean rank of the correct key across 'nb_attacks' runs for each trace count
    GE_curve = np.empty(nb_traces_attacks, dtype=np.float32)

    # Generate shuffled indices once for all attack experiments
    shuffles = [np.random.permutation(total_nb_traces_attacks) for _ in range(nb_attacks)]

    # ---------- 3. Vectorized rank update --------------------------------------------- #
    # Iterate over the number of traces used for attack, one by one
    for t in tqdm(range(nb_traces_attacks), desc="Calculating GE Evolution"):
        # Get the index of the trace to use for each attack run at current 't'
        # idxs: (nb_attacks,) -> e.g., [trace_idx_for_run1, trace_idx_for_run2, ...]
        idxs = np.array([s[t] for s in shuffles])    
        
        # Get the log probabilities and plaintext for these specific traces
        lp_slice = logp[idxs]                            # (nb_attacks, num_classes)
        pt_slice = plt_attack[idxs]                      # (nb_attacks,)

        # Calculate the sensitive value (label) for each possible key guess (0-256)
        # for each trace in the current slice, for each attack run.
        # `xor`: (nb_attacks, 256) where each row is [pt_i ^ 0, pt_i ^ 1, ..., pt_i ^ 255]
        xor_results = pt_slice[:, None] ^ np.arange(256)      
        
        if num_classes == 256: # ID leakage
            # `indices`: (nb_attacks, 256) same as xor_results, directly points to S-box output
            indices = AES_Sbox[xor_results]
        else: # HW leakage (num_classes == 9)
            # `indices`: (nb_attacks, 256) mapping S-box output to HW value (0-8)
            indices = HW_lookup_table[AES_Sbox[xor_results]] 

        # Gather the log probabilities corresponding to the 'indices' (sensitive values)
        # from the model's predictions (`lp_slice`).
        # `aligned`: (nb_attacks, 256) log-prob of (Sbox(pt ^ k_guess)) for each trace/run
        aligned = np.take_along_axis(lp_slice, indices, axis=1)
        
        # Accumulate log likelihoods for each key candidate across all attack runs
        key_probs_runs += aligned

        # Calculate ranks: For each attack run, sort the key_probs_runs for that run
        # argsort(-...) gives indices sorted in descending order of likelihood
        # argsort(argsort(-...)) gives ranks (0 to 255) directly
        ranks = np.argsort(np.argsort(-key_probs_runs, axis=1), axis=1)
        
        # Get the mean rank of the *correct* key (`correct_key`) across all `nb_attacks` runs
        GE_curve[t] = ranks[:, correct_key].mean()

    NTGE = NTGE_fn(GE_curve)
    print(f'GE: {GE_curve}')
    print(f"GE (final): {GE_curve[-1]:.2f}  |  NTGE: {NTGE:.0f}")

    return GE_curve, NTGE