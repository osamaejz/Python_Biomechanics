import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
import math
from typing import Tuple, Dict, List

# -------------------------
# Parameters (tweak here)
# -------------------------
fs_emg = 2000        # EMG original sampling rate (Hz)
fs_kin = 100         # Kinematics sampling rate (Hz) — target
bp_low, bp_high = 20, 450   # bandpass for EMG (Hz)
env_cut = 10         # envelope low-pass cutoff (Hz) — 5-10 typical
notch_freq = 50      # set None or 50 or 60 depending on mains
notch_Q = 30         # notch quality factor
bp_order = 4
env_order = 4

# -------------------------
# Helper: design filters
# -------------------------
def design_filters(fs_emg: int,
                   bp: Tuple[float,float]=(20,450),
                   env_cut: float=10.0,
                   notch_freq: float=None,
                   notch_Q: float=30,
                   bp_order: int=4,
                   env_order: int=4) -> Dict:
    ny = fs_emg / 2.0
    b_bp, a_bp = butter(bp_order, [bp[0]/ny, bp[1]/ny], btype='band')
    b_env, a_env = butter(env_order, env_cut/ny, btype='low')
    if notch_freq is not None:
        # normalized freq for scipy iirnotch: w0 in (0,1) = f0/(fs/2)
        w0 = notch_freq / ny
        b_notch, a_notch = iirnotch(w0, notch_Q)
    else:
        b_notch, a_notch = None, None
    return {'b_bp': b_bp, 'a_bp': a_bp,
            'b_env': b_env, 'a_env': a_env,
            'b_notch': b_notch, 'a_notch': a_notch}

# -------------------------
# Helper: detect time row presence
# -------------------------
def detect_time_row(arr: np.ndarray, expected_fs: int, tol_rel: float=0.05):
    """
    If the first row looks like a time vector sampled at expected_fs (within tol_rel),
    returns (has_time, time_vector, data_rows).
    Otherwise returns (False, None, arr).
    """
    if arr.ndim != 2:
        raise ValueError("Array must be 2D (rows x samples).")
    first_row = arr[0, :]
    if first_row.size < 2:
        return False, None, arr
    dt = np.median(np.diff(first_row))
    if dt <= 0:
        return False, None, arr
    expected_dt = 1.0 / expected_fs
    if abs(dt - expected_dt) <= tol_rel * expected_dt:
        return True, first_row, arr[1:, :]
    else:
        return False, None, arr

# -------------------------
# Core: process one gait cycle
# -------------------------
def process_cycle(emg_arr: np.ndarray,
                  kin_arr: np.ndarray,
                  fs_emg: int,
                  fs_kin: int,
                  filters: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input:
      emg_arr: (17, N_emg) or (nrows, N_emg) where row0 may be time and rows 1.. are channels
      kin_arr: (25, N_kin) or (nrows, N_kin) where row0 may be time (detected automatically)
    Returns:
      emg_ds_aligned: (n_channels, N_time_aligned)  -- envelope downsampled & aligned to kin time
      kin_data_trim: (n_kin_vars, N_time_aligned)  -- kinematic variables (time row removed)
      t_common: (N_time_aligned,)                   -- common time vector (seconds)
    """
    # 1) detect time rows
    emg_has_time, t_emg, emg_sig = detect_time_row(emg_arr, fs_emg)
    kin_has_time, t_kin, kin_data = detect_time_row(kin_arr, fs_kin)  # if kin has time row detect it
    
    # If emg had no explicit time row, build it (assume uniform sampling)
    if not emg_has_time:
        N_emg = emg_sig.shape[1]
        t_emg = np.arange(N_emg) / fs_emg
    
    # If kin had no time row, create one aligned to start time of emg (best-effort)
    if not kin_has_time:
        N_kin = kin_data.shape[1]
        # assume kin starts at first emg time and is sampled at fs_kin
        t_kin = t_emg[0] + np.arange(N_kin) / fs_kin
    
    # Sanity: ensure emg_sig shape is (n_ch, N)
    # If emg_sig came as (channels, samples) that's fine. If it's (samples, channels) transpose:
    if emg_sig.shape[0] < emg_sig.shape[1] and emg_sig.shape[0] <= 16:
        # usually channels x samples, keep as-is
        pass
    else:
        # fallback: if channels appear as columns, transpose
        if emg_sig.shape[0] > emg_sig.shape[1]:
            emg_sig = emg_sig.T
    
    # apply filters (axis=1 = along samples)
    # notch
    if filters['b_notch'] is not None:
        emg_sig = filtfilt(filters['b_notch'], filters['a_notch'], emg_sig, axis=1)
    # bandpass
    emg_bp = filtfilt(filters['b_bp'], filters['a_bp'], emg_sig, axis=1)
    # rectify
    emg_rect = np.abs(emg_bp)
    # envelope
    emg_env = filtfilt(filters['b_env'], filters['a_env'], emg_rect, axis=1)
    
    # downsample envelope to kin sampling using rational resampling (resample_poly)
    g = math.gcd(fs_emg, fs_kin)
    up = fs_kin // g
    down = fs_emg // g
    # note: resample_poly operates along axis; we have shape (channels, samples) => axis=1
    emg_env_ds = resample_poly(emg_env, up, down, axis=1)
    # downsample the emg time vector similarly
    t_emg_ds = resample_poly(t_emg, up, down)
    
    # Now align emg_env_ds to kinematics using interpolation onto kin times (best)
    # We'll interpolate EMG envelope channels at t_kin (which is at kin sampling instants).
    # Ensure t_emg_ds is strictly increasing
    # If t_kin extends beyond emg time bounds we trim to overlap
    start_t = max(t_emg_ds[0], t_kin[0])
    end_t   = min(t_emg_ds[-1], t_kin[-1])
    if end_t <= start_t:
        # no overlap — return empty
        return np.empty((emg_env_ds.shape[0], 0)), np.empty((kin_data.shape[0], 0)), np.array([])
    
    # select kin times in overlap
    kin_mask = (t_kin >= start_t) & (t_kin <= end_t)
    t_kin_trim = t_kin[kin_mask]
    kin_trim = kin_data[:, kin_mask]
    
    # interpolate each EMG channel onto t_kin_trim
    n_ch = emg_env_ds.shape[0]
    emg_interp = np.zeros((n_ch, t_kin_trim.size), dtype=float)
    for ch in range(n_ch):
        emg_interp[ch, :] = np.interp(t_kin_trim, t_emg_ds, emg_env_ds[ch, :])
    
    return emg_interp, kin_trim, t_kin_trim

# -------------------------
# Feature extraction (on downsampled envelope)
# -------------------------
def sliding_window_features(emg_ds: np.ndarray,
                            fs: int,
                            win_ms: int = 150,
                            hop_ms: int = 75,
                            wamp_threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    emg_ds: (n_channels, n_samples) -- expected at fs (e.g., 100 Hz)
    returns:
      features: (n_windows, n_channels * 5) [IEMG, MAV, RMS, WL, WAMP concatenated per channel]
      times:   (n_windows,) center time index (samples) of window (integers)
    """
    win_samps = max(1, int(round(win_ms/1000 * fs)))
    hop_samps = max(1, int(round(hop_ms/1000 * fs)))
    n_ch, n_samples = emg_ds.shape
    feats = []
    times = []
    for start in range(0, n_samples - win_samps + 1, hop_samps):
        seg = emg_ds[:, start:start+win_samps]  # shape (ch, win)
        IEMG = np.sum(np.abs(seg), axis=1)              # integrated EMG over window
        MAV  = np.mean(np.abs(seg), axis=1)
        RMS  = np.sqrt(np.mean(seg**2, axis=1))
        WL   = np.sum(np.abs(np.diff(seg, axis=1)), axis=1)
        WAMP = np.sum(np.abs(np.diff(seg, axis=1)) > wamp_threshold, axis=1)
        feat_vec = np.concatenate([IEMG, MAV, RMS, WL, WAMP])  # length n_ch*5
        feats.append(feat_vec)
        times.append(start + win_samps//2)
    if len(feats) == 0:
        return np.empty((0, n_ch*5)), np.empty((0,), dtype=int)
    return np.vstack(feats), np.array(times, dtype=int)

# -------------------------
# Batch processing for all cycles
# -------------------------
def process_all_cycles(emg_list: List[np.ndarray],
                       kin_list: List[np.ndarray],
                       fs_emg: int = fs_emg,
                       fs_kin: int = fs_kin,
                       filters: Dict = None):
    if filters is None:
        filters = design_filters(fs_emg, (bp_low, bp_high), env_cut, notch_freq, notch_Q, bp_order, env_order)
    n = min(len(emg_list), len(kin_list))
    emg_aligned_list = []
    kin_aligned_list = []
    t_common_list = []
    for i in range(n):
        try:
            emg_a, kin_a, t_common = process_cycle(emg_list[i], kin_list[i], fs_emg, fs_kin, filters)
            emg_aligned_list.append(emg_a)
            kin_aligned_list.append(kin_a)
            t_common_list.append(t_common)
        except Exception as e:
            # catch and continue; log/print if you want
            print(f"Warning: cycle {i} processing failed: {e}")
            emg_aligned_list.append(np.empty((0,0)))
            kin_aligned_list.append(np.empty((0,0)))
            t_common_list.append(np.array([]))
    return emg_aligned_list, kin_aligned_list, t_common_list

# -------------------------
# Example usage
# -------------------------
# 1) design filters once
filters = design_filters(fs_emg, (bp_low, bp_high), env_cut, notch_freq, notch_Q, bp_order, env_order)

# 2) run over all cycles (this will produce lists of aligned (channels x samples) arrays)
# emg_list and kin_list must exist in your workspace
# emg_aligned_list, kin_aligned_list, t_common_list = process_all_cycles(emg_list, kin_list, fs_emg, fs_kin, filters)

# 3) feature extraction example for cycle i (after you have emg_aligned_list)
# feats_i, feat_times_i = sliding_window_features(emg_aligned_list[i], fs_kin, win_ms=150, hop_ms=75, wamp_threshold=0.01)
