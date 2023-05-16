import sys
import wave
import argparse
import contextlib

parser = argparse.ArgumentParser(
                    prog='Pcm2Wav',
                    description='Transform Pcm2Wav',
                    epilog='Text at the bottom of help')

parser.add_argument('pcmName')
parser.add_argument('sampleRate')
parser.add_argument('wavName')

args = parser.parse_args()


with open(args.pcmName, 'rb') as pcmfile:
    pcmdata = pcmfile.read()
with wave.open(args.wavName, 'wb') as wavfile:
    wavfile.setparams((1, 2, int(args.sampleRate), 0, 'NONE', 'NONE'))
    wavfile.writeframes(pcmdata)

# fname = 'get.wav'
# print(librosa.get_duration(path=fname))

with contextlib.closing(wave.open(args.wavName, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    print("Audio duratin:", duration)
