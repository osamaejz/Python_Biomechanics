import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import resample, resample_poly

class Process():

    def __init__(self):
        pass

    @staticmethod
    def load_data(EMG_File, Kin_File):
        EMG = scipy.io.loadmat(EMG_File)
        Kin = scipy.io.loadmat(Kin_File)

        EMG_data = EMG[EMG_File[0:-4]]
        Kin_data = Kin[Kin_File[0:-4]]

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
        return iirnotch(w0=freq/(fs/2), Q=Q)

    @staticmethod
    def preprocess_emg(emg_signal, fs=2000, target_fs=100):
        # print("Starting filtration")
        b, a = Process.butter_bandpass(10, 400, fs)
        emg_filt = filtfilt(b, a, emg_signal)
        # print("filtration done")

        b, a = Process.notch_filter(fs, freq=50.0)
        emg_notch = filtfilt(b, a, emg_filt)
        # print("Notch done")

        emg_rect = np.abs(emg_notch)
        # print("Rectification done")

        b, a = Process.butter_lowpass(6, fs)
        envelope = filtfilt(b, a, emg_rect)
        # print("Envelope done")

        # --- Downsample envelope to match kinematics ---
        if target_fs < fs:
            gcd = np.gcd(fs, target_fs)
            up, down = target_fs // gcd, fs // gcd
            envelope = resample_poly(envelope, up, down)
            
        return envelope, emg_filt, emg_notch, emg_rect

    @staticmethod
    def plot_all_spectra(signals, fs, titles):
        n = len(signals)
        plt.figure(figsize=(10, 2*n))  # one column, n rows
        
        for i, (sig, title) in enumerate(zip(signals, titles), 1):
            N = len(sig)
            freqs = fftfreq(N, 1/fs)[:N//2]
            spectrum = np.abs(fft(sig))[:N//2]

            plt.subplot(n, 1, i)
            plt.plot(freqs, spectrum)
            plt.title(title)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
        
        plt.show()

    # @staticmethod
    # def window_emg(emg_env, fs=100, win_ms=150, hop_ms=75):
    #     """
    #     Splits EMG envelope into overlapping windows.
    #     emg_env: np.ndarray, shape (n_channels, n_samples) at fs
    #     Returns:
    #     windows: list of np.ndarray, each (n_channels, win_samps)
    #     centers: np.ndarray, sample indices of window centers
    #     """
    #     if emg_env.size == 0:
    #         return [], []

    #     n_ch, n_samples = emg_env.shape
    #     win_samps = max(1, int(round(win_ms/1000 * fs)))
    #     hop_samps = max(1, int(round(hop_ms/1000 * fs)))

    #     windows = []
    #     centers = []

    #     for start in range(0, n_samples - win_samps + 1, hop_samps):
    #         seg = emg_env[:, start:start+win_samps]
    #         windows.append(seg)
    #         centers.append(start + win_samps // 2)

    #     return windows, np.array(centers, dtype=int)
    
    # @staticmethod
    # def window_signals(emg_env, kin_signal=None, fs=100, win_ms=150, hop_ms=75, kin_mode="center"):
    #     """
    #     Window EMG envelope (multi-channel) and optionally a kinematic signal.

    #     Parameters
    #     ----------
    #     emg_env : np.ndarray
    #         Shape (n_channels, n_samples)
    #     kin_signal : np.ndarray or None
    #         Shape (n_samples,) if provided
    #     fs : int
    #         Sampling frequency (Hz)
    #     win_ms : int
    #         Window length (ms)
    #     hop_ms : int
    #         Hop length (ms)
    #     kin_mode : str
    #         "center" → take center sample of window
    #         "mean"   → take mean value of window
    #         "seq"    → keep full sequence of window

    #     Returns
    #     -------
    #     emg_windows : np.ndarray
    #         Shape (n_windows, n_channels, win_samps)
    #     kin_windows : np.ndarray or None
    #         If kin_signal is given:
    #         - mode "center"/"mean" → shape (n_windows,)
    #         - mode "seq"           → shape (n_windows, win_samps, 1)
    #     centers : np.ndarray
    #         Indices of window centers
    #     """
    #     n_ch, n_samples = emg_env.shape
    #     win_samps = max(1, int(round(win_ms/1000 * fs)))
    #     hop_samps = max(1, int(round(hop_ms/1000 * fs)))

    #     emg_windows = []
    #     kin_windows = []
    #     centers = []

    #     for start in range(0, n_samples - win_samps + 1, hop_samps):
    #         stop = start + win_samps
    #         seg_emg = emg_env[:, start:stop]  # (n_ch, win_samps)
    #         emg_windows.append(seg_emg)
    #         centers.append(start + win_samps // 2)

    #         if kin_signal is not None:
    #             seg_kin = kin_signal[start:stop]
    #             if kin_mode == "center":
    #                 kin_windows.append(seg_kin[len(seg_kin)//2])
    #             elif kin_mode == "mean":
    #                 kin_windows.append(np.mean(seg_kin))
    #             elif kin_mode == "seq":
    #                 kin_windows.append(seg_kin[:, np.newaxis])  # (win_samps, 1)

    #     emg_windows = np.stack(emg_windows, axis=0)  # (n_windows, n_ch, win_samps)

    #     if kin_signal is None:
    #         return emg_windows, None, np.array(centers, dtype=int)

    #     if kin_mode in ["center", "mean"]:
    #         kin_windows = np.array(kin_windows)  # (n_windows,)
    #     elif kin_mode == "seq":
    #         kin_windows = np.stack(kin_windows, axis=0)  # (n_windows, win_samps, 1)

    #     return emg_windows, kin_windows, np.array(centers, dtype=int)

    @staticmethod
    def window_signals(emg, kin, fs, window_size, step):
        emg_windows = []
        kin_windows = []
        emg_centers = []
        
        window_size = int(window_size * fs)
        step = int(step * fs)

        for start in range(0, emg.shape[1] - window_size + 1, step):
            end = start + window_size
            emg_windows.append(emg[:, start:end])

            # Center index of the current window
            center = start + window_size // 2
            emg_centers.append(center)

            # Get kin value at the same center (if lengths align)
            if center < len(kin):
                kin_windows.append(kin[center])

        return emg_windows, kin_windows, emg_centers




    @staticmethod
    def extract_windowed_features_from_envelope(emg_env, fs=100,
                                                win_ms=150, hop_ms=75,
                                                wamp_threshold=0.01):
        """
        emg_env: np.ndarray, shape (n_channels, n_samples) -- envelope at fs (e.g., 100 Hz)
        Returns:
        features: np.ndarray, shape (n_windows, n_channels*5)
        centers:  np.ndarray, shape (n_windows,) indices (sample indices of window center)
        """
        if emg_env.size == 0:
            return np.empty((0,0)), np.empty((0,), dtype=int)

        n_ch, n_samples = emg_env.shape
        win_samps = max(1, int(round(win_ms/1000 * fs)))
        hop_samps = max(1, int(round(hop_ms/1000 * fs)))
        windows = []
        centers = []

        for start in range(0, n_samples - win_samps + 1, hop_samps):
            seg = emg_env[:, start:start+win_samps]   # shape (n_ch, win_samps)

            # Features (per channel)
            IEMG = np.sum(seg, axis=1)                       # integrated EMG
            MAV  = np.mean(seg, axis=1)                      # mean absolute value (envelope already positive)
            RMS  = np.sqrt(np.mean(seg**2, axis=1))
            WL   = np.sum(np.abs(np.diff(seg, axis=1)), axis=1)
            WAMP = np.sum((np.abs(np.diff(seg, axis=1)) > wamp_threshold).astype(int), axis=1)

            feat_vec = np.concatenate([IEMG, MAV, RMS, WL, WAMP])  # length n_ch*5
            windows.append(feat_vec)
            centers.append(start + win_samps // 2)

        if len(windows) == 0:
            return np.empty((0, n_ch*5)), np.empty((0,), dtype=int)

        return np.vstack(windows), np.array(centers, dtype=int)
