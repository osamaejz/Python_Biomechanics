# main.py (relevant parts)
from Seq2Seq.SS_Data_process import Process
import numpy as np
from Seq2Seq.SS_lstm_model import ModelTrainer
import matplotlib.pyplot as plt

# --- load data (unchanged)
EMG_Data, Kin_Data = Process.load_data("Mat_Data//R_UH_GC_EMG.mat", "Mat_Data//R_UH_GC_kin.mat")

all_emg_windows = []
all_kin_windows = []


# user choices
fs = 100                   # kinematics / envelope sampling rate after downsampling
window_size = 30           # samples (30 -> 300 ms at 100 Hz)
step = 15                  # overlap 50%
use_features = True        # if True, add IEMG/MAV/RMS/WL/WAMP features tiled across timesteps
left_leg_only = True       # if you want to keep only left-leg channels (adjust indices below)

for idx, (curr_EMG, curr_Kin) in enumerate(zip(EMG_Data, Kin_Data)):
    curr_EMG = np.transpose(curr_EMG)   # shape (17, N)
    curr_Kin = np.transpose(curr_Kin)   # shape (25, M)
    curr_Kin = curr_Kin[5, :]           # sagittal ankle angle -> (M,)

    envelope = []
    rect_emg = []
    ## preprocess channels 1..16 (skip timestamps index 0)
    for flt in range(1, 17):
        env, filt, notch, rect = Process.preprocess_emg(curr_EMG[flt], fs=2000, target_fs=100)
        envelope.append(env)
        rect_emg.append(rect)

    emg_envelop = np.vstack(envelope)   # (16, n_samples)

    left_leg_channels = emg_envelop[1::2, :]

    print("EMG Envelope: ", np.shape(left_leg_channels))
    ## windowing -> returns (n_windows, window_size, n_features) and kin windows (n_windows, window_size)
    emg_win, kin_win = Process.window_signals(left_leg_channels, curr_Kin, fs=fs,
                                              window_size=window_size, step=step,
                                              return_sequences=True, add_features=use_features)
    print("EMG Window Size: ", np.shape(emg_win))
    if emg_win.size > 0:
        all_emg_windows.append(emg_win)
        all_kin_windows.append(kin_win)
 
    print(f"Processed gait cycle {idx+1}/{len(EMG_Data)}")
    

# concatenate across gait cycles
all_emg_windows = np.vstack(all_emg_windows)   # (total_windows, window_size, n_features)
all_kin_windows = np.vstack(all_kin_windows)   # (total_windows, window_size)

print("EMG windows shape:", all_emg_windows.shape)
print("Kin windows shape:", all_kin_windows.shape)

# # optionally select left-leg channels only (if you know the channel mapping)
# if left_leg_only and not use_features:
#     left_leg_indices = np.arange(0, 16, 2)   # example: 0-based even indices for left leg
#     all_emg_windows = all_emg_windows[:, :, left_leg_indices]
#     print("Selected left leg -> new EMG shape:", all_emg_windows.shape)

# X and y
X = all_emg_windows.astype(np.float32)   # (n_samples, seq_len, n_features)
y = all_kin_windows.astype(np.float32)   # (n_samples, seq_len)

# initialize trainer
trainer = ModelTrainer(input_size=X.shape[2],
                    #    hidden_size=128,    # increased
                    #    num_layers=2,       # deeper model
                    #    lr=1e-3,
                    #    batch_size=32,
                        num_epochs=200,
                       loss_fn="mse")

# prepare, build and train
trainer.prepare_data(X, y, test_split=0.2)
trainer.build_model()
trainer.train()

# evaluate
preds, trues = trainer.evaluate()

# compute metrics (flattened arrays)
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(trues, preds))
r2 = r2_score(trues, preds)
corr = np.corrcoef(trues, preds)[0,1]
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Correlation: {corr:.4f}")

# quick plot (first 400 flattened samples)
plt.figure(figsize=(10,4))
plt.plot(trues[:], label='True')
plt.plot(preds[:], label='Pred')
plt.xlabel("No. of Samples")
plt.ylabel("Angle (Degrees)")
plt.legend()
plt.show()
