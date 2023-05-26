import matplotlib.pyplot as plt  # 画图包
import numpy as np
import contextlib
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wavfile
import wave
from scipy import signal

from get_data import highpassfilter

#	1.设定文件的格式为小端，16bit有符号整型，小端存储
dt = np.dtype('<h')

# path1 = 'samples/recording_1684330722347_13.pcm'
path1 = '../all/recording_1684330725318_13.pcm'
path2 = '../0517_data/recording_1684330759387_13.pcm'

y = np.fromfile(path1, dtype=dt, sep='', offset=0)
hi = highpassfilter();
y = hi.butter_highpass_filter(y, 18000, 63333)
abs_array = np.abs(y)
index_f = np.argmax(abs_array > 1000)
max = np.max(abs_array[index_f:index_f + 3000])
indices = np.where(abs_array > max / 4)[0]
indices = indices[indices < index_f + 3000]
id = np.max(indices)
# cut signal
y = y[id:id+1500]
cnt = len(y)

# Time Domain
time = np.linspace(0, len(y - 1) / 63333, len(y - 1)) * 1000
plt.plot(time, y / len(y))  # plot in seconds
plt.title("Voice Signal")
plt.xlabel("Time [seconds]")
plt.ylabel("Voice amplitude")
plt.ylim(-0.01, 0.01)
# plt.xlim(200, 300)
plt.show()

fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis')
cmap.set_under(color='k', alpha=None)

# Spectrogram 1
powerSpectrum, frequenciesFound, time, imageAxis = ax.specgram(y, Fs=63333, NFFT= 256, mode="magnitude", noverlap= 128)
fig.colorbar(imageAxis)
plt.ylim(19000,22000)
# plt.xlim(0.10,0.15)
# plt.xlim(0.1, 0.2)
plt.show()

# Spectrogram 2
freq, t, stft = signal.spectrogram(y, fs=63333, mode='complex')
#Sxx, freq, t = plt.specgram(Voice, Fs=Fs, mode='magnitude')
plt.pcolormesh(t, freq, abs(stft), shading='gouraud')
plt.title('Spectrogramm using STFT amplitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [seconds]')
plt.show()


# Freq Domain
# Fourier transform
F = fft(y) / cnt
#f = np.linspace(0, Fs - Fs / N, N)
f = fftfreq(cnt, 1 / 63333)[:cnt // 2]
#f = np.linspace(0, 4000, N//2)
plt.plot(f, abs(F[0:cnt // 2]))
plt.title("FFT of the signal")
plt.xlabel('Frequency')
plt.ylabel('Power of Frequency')
plt.show()


# plt.semilogy(xf[1:cnt//2], 2.0/N * np.abs(ywf[1:cnt//2]), '-r')
# plt.legend(['FFT', 'FFT w. window'])

# powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(yf, Fs=63333)

# old fft
# cnt = len(y)
# # sp = np.fft.fft(y)
# # freq = np.fft.fftfreq(cnt, 1 / 63333)
# # plt.plot(freq, sp.real, freq, sp.imag)
# # plt.show()
# yf = fft(y)
# # ywf = fft(y*w)
# xf = fftfreq(cnt, 1 / 63333)[:cnt // 2]
# plt.semilogy(xf[1:cnt // 2], 2.0 / cnt * np.abs(yf[1:cnt // 2]), '-b')
# plt.grid()
# plt.show()

# with open(path1, 'rb') as pcmfile:
#     pcmdata = pcmfile.read()
# with wave.open('get.wav', 'wb') as wavfile1:
#     wavfile1.setparams((1, 2, 63333, 0, 'NONE', 'NONE'))
#     wavfile1.writeframes(pcmdata)

# Fs, aud = wavfile.read('get.wav')
# select left channel only
# aud = aud[:,0]



