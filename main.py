from Data_process import Process
import numpy as np
import matplotlib.pyplot as plt

EMG_Data, Kin_Data = Process.load_data("R_UH_GC_EMG.mat", "R_UH_GC_kin.mat")
curr_EMG = EMG_Data[326]
curr_Kin = Kin_Data[326]

curr_EMG = np.transpose(curr_EMG)
curr_Kin = np.transpose(curr_Kin)
curr_Kin = curr_Kin[5,:] # 5 for angle in saggital plane

print("EMG Shape is: ", np.shape(curr_EMG))
print("Kin Shape is: ", np.shape(curr_Kin))

#EMG Shape is:  (17, 2140)
#Kin Shape is:  (25, 108)

envelope = []
emg_filt = []
emg_notch = []
emg_rect = []

for flt in range(1, 17):
    env, filt, notch, rect = Process.preprocess_emg(curr_EMG[flt])
    envelope.append(env)
    emg_filt.append(filt)
    emg_notch.append(notch)
    emg_rect.append(rect)

# convert to numpy arrays (ragged because downsample changes length)
envelope = np.array(envelope, dtype=object)
emg_filt = np.array(emg_filt, dtype=object)
emg_notch = np.array(emg_notch, dtype=object)
emg_rect = np.array(emg_rect, dtype=object)

emg_envelop = np.vstack(envelope)   # shape: (17, ~107)

print("EMG Envelope Shape is: ", np.shape(emg_envelop))
print("Kin Shape is: ", np.shape(curr_Kin))


win, center = Process.window_emg(emg_envelop, 100, 150, 75)
windows = np.array(win)
print("Window size: ", np.shape(windows))



features, centers = Process.extract_windowed_features_from_envelope(
    envelopes_mat, fs=100,   # because envelope was downsampled to 100 Hz
    win_ms=150, hop_ms=75,   # 150 ms window, 75 ms hop
    wamp_threshold=0.01
)
feat, centre = Process.extract_windowed_features_from_envelope(curr_EMG[1])


plt.subplot(2,2,1)
plt.plot(curr_EMG[5])
plt.title("Raw")

plt.subplot(2,2,2)
plt.plot(emg_filt[5])
plt.title("Filtered")

plt.subplot(2,2,3)
plt.plot(emg_rect[5])
plt.title("Rectified")

plt.subplot(2,2,4)
plt.plot(envelope[5])
plt.title("Envelope")
plt.show()

# fs = 2000
# Process.plot_all_spectra(
#     [curr_EMG[5], emg_filt[5], emg_notch[5], emg_rect[5], envelope[5]],
#     fs=2000,
#     titles=["Raw EMG", "Band-pass", "Notch Filtered", "Rectified", "Envelope"]
# )

