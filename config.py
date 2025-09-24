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