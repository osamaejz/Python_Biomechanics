# from Data_process import Process
# import numpy as np
# import matplotlib.pyplot as plt

# EMG_Data, Kin_Data = Process.load_data("R_UH_GC_EMG.mat", "R_UH_GC_kin.mat")
# curr_EMG = EMG_Data[326]
# curr_Kin = Kin_Data[326]

# curr_EMG = np.transpose(curr_EMG)
# curr_Kin = np.transpose(curr_Kin)
# curr_Kin = curr_Kin[5,:] # 5 for angle in saggital plane

# print("EMG Shape is: ", np.shape(curr_EMG))
# print("Kin Shape is: ", np.shape(curr_Kin))

# #EMG Shape is:  (17, 2140)
# #Kin Shape is:  (25, 108)

# envelope = []
# emg_filt = []
# emg_notch = []
# emg_rect = []

# for flt in range(1, 17):
#     env, filt, notch, rect = Process.preprocess_emg(curr_EMG[flt])
#     envelope.append(env)
#     emg_filt.append(filt)
#     emg_notch.append(notch)
#     emg_rect.append(rect)

# # convert to numpy arrays (ragged because downsample changes length)
# envelope = np.array(envelope, dtype=object)
# emg_filt = np.array(emg_filt, dtype=object)
# emg_notch = np.array(emg_notch, dtype=object)
# emg_rect = np.array(emg_rect, dtype=object)

# emg_envelop = np.vstack(envelope)   # shape: (17, ~107)

# print("EMG Envelope Shape is: ", np.shape(emg_envelop))
# print("Kin Shape is: ", np.shape(curr_Kin))

# emg_win, Kin_win, center_1 = Process.window_signals(emg_envelop, curr_Kin, 100, 150, 75)
# emg_win = np.array(emg_win)
# Kin_win = np.array(Kin_win)

# print("EMG window size: ", np.shape(emg_win))
# print("Kin Window size: ", np.shape(Kin_win))




from Data_process import Process
import numpy as np
import matplotlib.pyplot as plt

# Load all gait cycles
EMG_Data, Kin_Data = Process.load_data("R_UH_GC_EMG.mat", "R_UH_GC_kin.mat")

all_emg_windows = []
all_kin_windows = []

# Loop through all gait cycles
for idx, (curr_EMG, curr_Kin) in enumerate(zip(EMG_Data, Kin_Data)):
    curr_EMG = np.transpose(curr_EMG)   # shape: (17, N)
    curr_Kin = np.transpose(curr_Kin)   # shape: (25, M)

    # pick one kinematic variable (e.g., sagittal plane angle)
    curr_Kin = curr_Kin[5, :]  

    # preprocess all EMG channels
    envelope = []
    for flt in range(1, 17):   # skip row 0 (timestamps)
        env, filt, notch, rect = Process.preprocess_emg(curr_EMG[flt])
        envelope.append(env)

    # convert envelope to array
    emg_envelop = np.vstack(envelope)   # shape: (16, ~len)

    # windowing EMG and Kin together
    emg_win, kin_win, emg_centers = Process.window_signals(emg_envelop, curr_Kin, 100, 0.3, 0.15)
   
    # store results
    if len(emg_win) > 0:
        all_emg_windows.append(np.array(emg_win))
        all_kin_windows.append(np.array(kin_win))

    print(f"Processed gait cycle {idx+1}/{len(EMG_Data)}")

# Concatenate across gait cycles
all_emg_windows = np.vstack(all_emg_windows)   # shape: (total_windows, 16, win_size)
all_kin_windows = np.concatenate(all_kin_windows)  # shape: (total_windows,)

print("Final EMG window dataset:", all_emg_windows.shape)
print("Final Kin window dataset:", all_kin_windows.shape)

# Keep only even index channels (left leg)
left_leg_indices = np.arange(1, 16, 2)   # [1,3,5,...,15]

all_emg_windows_left = all_emg_windows[:, left_leg_indices, :]

print("Original EMG windows:", all_emg_windows.shape)
print("Left leg EMG windows:", all_emg_windows_left.shape)



# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from lstm_model import LSTMTrainer

# # Assuming you already have:
# # all_emg_windows: (5809, seq_len, channels)
# # all_kin_windows: (5809,)

# # Convert to torch tensors
# X = torch.tensor(all_emg_windows, dtype=torch.float32)
# y = torch.tensor(all_kin_windows, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

# # Train/test split
# train_size = int(0.8 * len(X))
# test_size = len(X) - train_size
# train_X, test_X = torch.utils.data.random_split(X, [train_size, test_size])
# train_y, test_y = torch.utils.data.random_split(y, [train_size, test_size])

# # Wrap in dataset
# train_dataset = TensorDataset(train_X.dataset[train_X.indices], train_y.dataset[train_y.indices])
# test_dataset = TensorDataset(test_X.dataset[test_X.indices], test_y.dataset[test_y.indices])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Define trainer
# input_size = X.shape[2]   # number of channels
# trainer = LSTMTrainer(input_size=input_size, hidden_size=64, num_layers=2, output_size=1, lr=0.001)

# # Train
# trainer.train(train_loader, num_epochs=30, device="cpu")

# # Evaluate
# trainer.evaluate(test_loader, device="cpu")


import numpy as np
from lstm_model import ModelTrainer

X = np.transpose(all_emg_windows_left, (0, 2, 1))  
# now X.shape = (5809, 15, 16) → (samples, time, features)

# X = all_emg_windows
y = all_kin_windows

# Initialize trainer
trainer = ModelTrainer(input_size=X.shape[2],
                       hidden_size=128,
                       num_layers=2,
                       lr=1e-3,
                       batch_size=32,
                       num_epochs=50,
                       loss_fn="mse")

# Prepare data
trainer.prepare_data(X, y, test_split=0.2)

# Build and train model
trainer.build_model()
trainer.train()

# Evaluate model
preds, trues = trainer.evaluate()

print("Predictions shape:", preds.shape)
print("Ground truth shape:", trues.shape)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(trues, preds))
r2 = r2_score(trues, preds)
corr = np.corrcoef(trues, preds)[0,1]

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Correlation: {corr:.4f}")



for i in range(10):
    print(f"True: {trues[i]:.2f}, Pred: {preds[i]:.2f}")

import matplotlib.pyplot as plt
plt.plot(trues[:200], label="True")
plt.plot(preds[:200], label="Pred")
plt.legend()
plt.show()
