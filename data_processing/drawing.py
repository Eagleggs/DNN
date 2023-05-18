import matplotlib.pyplot as plt  # 画图包
import numpy as np
import contextlib
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wavfile
import wave
import argparse

parser = argparse.ArgumentParser(
                    prog='PcmDraw',
                    description='FFT and spectrogram',
                    epilog='Text at the bottom of help')

parser.add_argument('pcmName')
parser.add_argument('sampleRate')

args = parser.parse_args()

dt = np.dtype('<h')
pcmPath = args.pcmName
sampleRates = int(args.sampleRate)
y = np.fromfile(pcmPath, dtype=dt, sep='', offset=0)

cnt = len(y)
yf = fft(y)
xf = fftfreq(cnt, 1 / sampleRates)[:cnt // 2]



# plt.semilogy(xf[1:cnt//2], 2.0/N * np.abs(ywf[1:cnt//2]), '-r')
# plt.legend(['FFT', 'FFT w. window'])
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(yf, Fs=63333)
plt.grid()
plt.show()

with open(pcmPath, 'rb') as pcmfile:
    pcmdata = pcmfile.read()
with wave.open('get.wav', 'wb') as wavfile1:
    wavfile1.setparams((1, 2, sampleRates, 0, 'NONE', 'NONE'))
    wavfile1.writeframes(pcmdata)


Fs, aud = wavfile.read('get.wav')
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs, NFFT=1024, noverlap=512)
plt.ylim(20500,21500)
plt.show()

