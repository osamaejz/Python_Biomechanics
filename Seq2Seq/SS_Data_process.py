# Data_process.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
import scipy.io

class Process:
    @staticmethod
    def load_data(EMG_File, Kin_File):
        EMG = scipy.io.loadmat(EMG_File)
        Kin = scipy.io.loadmat(Kin_File)

        EMG_data = EMG[EMG_File[10:-4]]
        Kin_data = Kin[Kin_File[10:-4]]

        EMG_data = [np.array(cell) for cell in EMG_data[0]]
        Kin_data = [np.array(cell) for cell in Kin_data[0]]

        return EMG_data, Kin_data

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        return butter(order, [low, high], btype="band")

    @staticmethod
    def butter_lowpass(cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype="low")

    @staticmethod
    def notch_filter(fs, freq=50.0, Q=30.0):
        # returns (b, a) coefficients
        return iirnotch(w0=freq/(fs/2), Q=Q)

    @staticmethod
    def preprocess_emg(emg_signal, fs=2000, target_fs=100):
        """
        Returns: envelope (downsampled to target_fs), emg_filt, emg_notch, emg_rect
        Note: rectification done AFTER notch filtering.
        """
        # bandpass
        b, a = Process.butter_bandpass(10, 400, fs)
        emg_filt = filtfilt(b, a, emg_signal)

        # notch
        b, a = Process.notch_filter(fs, freq=50.0)
        emg_notch = filtfilt(b, a, emg_filt)

        # rectify
        emg_rect = np.abs(emg_notch)

        # low-pass for envelope
        b, a = Process.butter_lowpass(6, fs)
        envelope = filtfilt(b, a, emg_rect)

        # downsample to target_fs using polyphase (avoid alias)
        if target_fs < fs:
            gcd = np.gcd(fs, target_fs)
            up = target_fs // gcd
            down = fs // gcd
            envelope = resample_poly(envelope, up, down)

        return envelope, emg_filt, emg_notch, emg_rect

    @staticmethod
    def extract_time_features(emg_window):
        """
        emg_window: ndarray shape (n_channels, win_len)
        returns: 1D array of length n_channels * 5 (IEMG, MAV, RMS, WL, WAMP per channel)
        """
        n_ch, win = emg_window.shape
        feats = np.zeros(n_ch * 5, dtype=float)
        for ch in range(n_ch):
            sig = emg_window[ch]
            iemg = np.sum(np.abs(sig))
            mav = np.mean(np.abs(sig))
            rms = np.sqrt(np.mean(sig**2))
            wl = np.sum(np.abs(np.diff(sig)))
            # WAMP threshold: a small fraction of channel max (avoid zero)
            thr = 0.02 * (np.max(np.abs(sig)) + 1e-8)
            wamp = np.sum(np.abs(np.diff(sig)) > thr)
            feats[ch*5:(ch+1)*5] = [iemg, mav, rms, wl, wamp]
        return feats

    @staticmethod
    def window_signals(emg, kin, fs, window_size, step, return_sequences=True, add_features=False):
        """
        emg: (n_channels, n_samples)
        kin: (n_samples,)  -- kinematic scalar per sample
        fs, window_size, step: in SAMPLES (not ms). Eg window_size=30 means 30 samples.
        return_sequences: True -> kin windows are sequences shape (n_windows, window_size)
                          False -> kin windows are single center values
        add_features: if True, compute 5 time-domain features per channel for each window,
                      tile those features across timesteps and append them to per-timestep features.
        Returns:
            emg_windows: ndarray (n_windows, window_size, n_features)
            kin_windows: ndarray (n_windows, window_size)  OR (n_windows,) if return_sequences=False
        """
        emg_windows = []
        kin_windows = []
        n_channels, n_samples = emg.shape

        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            emg_w = emg[:, start:end]               # (n_channels, window_size)
            emg_w_t = emg_w.T                       # (window_size, n_channels)
            # print("EMG size after slice and transpose: ", np.shape(emg_w_t))
            if add_features:
                feats = Process.extract_time_features(emg_w)          # (n_channels*5,)
                feats_tiled = np.tile(feats, (window_size, 1))       # (window_size, n_channels*5)
                emg_w_t = np.concatenate([emg_w_t, feats_tiled], axis=1)

            # print("EMG after feature: ", np.shape(emg_w_t))
            emg_windows.append(emg_w_t)

            if return_sequences:
                kin_windows.append(kin[start:end])
            else:
                center = start + window_size // 2
                kin_windows.append( kin[center] if center < len(kin) else 0.0 )

        emg_windows = np.array(emg_windows)     # (n_windows, window_size, n_features)
        # print("EMG window Size in Dataprocess: ", np.shape(emg_windows))
        kin_windows = np.array(kin_windows)     # (n_windows, window_size) or (n_windows,)
        return emg_windows, kin_windows
