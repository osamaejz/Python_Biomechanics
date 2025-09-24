import numpy as np

def extract_emg_features(emg_envelope, fs, win_ms=150, hop_ms=75, wamp_threshold=0.01):
    """
    Extract IEMG, MAV, RMS, WL, WAMP features from multi-channel EMG envelope.
    
    Parameters:
        emg_envelope : np.ndarray
            Shape: (n_channels, n_samples) -> envelope of EMG after preprocessing
        fs : int
            Sampling rate of the envelope (Hz)
        win_ms : int
            Window length in milliseconds
        hop_ms : int
            Step size between windows in milliseconds
        wamp_threshold : float
            Threshold for WAMP calculation (absolute difference)
            
    Returns:
        features : np.ndarray
            Shape: (n_windows, n_channels * 5)
        feature_times : np.ndarray
            Center time (in samples) of each window
    """
    n_ch, n_samples = emg_envelope.shape
    win_samps = int(win_ms / 1000 * fs)
    hop_samps = int(hop_ms / 1000 * fs)
    
    features = []
    feature_times = []
    
    for start in range(0, n_samples - win_samps + 1, hop_samps):
        seg = emg_envelope[:, start:start+win_samps]  # window segment
        
        IEMG = np.sum(seg, axis=1)
        MAV  = np.mean(seg, axis=1)
        RMS  = np.sqrt(np.mean(seg**2, axis=1))
        WL   = np.sum(np.abs(np.diff(seg, axis=1)), axis=1)
        WAMP = np.sum(np.abs(np.diff(seg, axis=1)) > wamp_threshold, axis=1)
        
        # Concatenate all features for all channels
        feat_vector = np.concatenate([IEMG, MAV, RMS, WL, WAMP])
        features.append(feat_vector)
        
        # Store window center time
        feature_times.append(start + win_samps // 2)
    
    if len(features) == 0:
        return np.empty((0, n_ch*5)), np.empty((0,))
    
    return np.vstack(features), np.array(feature_times)
