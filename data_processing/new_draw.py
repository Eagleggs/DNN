import numpy as np
import matplotlib.pyplot as plt

# Read the .pcm file
with open('./temp/recording_new_1.pcm', 'rb') as f:
    pcm_data = f.read()

# Convert the raw PCM data to a NumPy array
data = np.frombuffer(pcm_data, dtype='int16')
# for element in data:
#     print(element)

# Set up audio recording parameters
sample_rate = 63333

# Plot the spectrogram
plt.specgram(data, Fs=sample_rate, cmap='jet',  vmin=-60, vmax=60)
plt.xlabel('Time [sec]')
# plt.ylim([12000, 12500])
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram of PCM file')
plt.colorbar()
plt.show()