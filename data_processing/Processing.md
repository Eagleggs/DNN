# Processing Raw PCM data
- `drawing.py`
  - draw the FFT and spectrogram of the PCM file
  - usage: `python drawing.py pcmName sampleRate`
  - example: `python drawing.py ./samples/recording_21.pcm 63333`
- `pcm2wav.py`
  - transform PCM to WAV, get duration of PCM(WAV)
  - usage: `python pcm2wav.py pcmName sampleRate wavName`
  - example: `python pcm2wav.py ./samples/recording_21.pcm 63333 new.wav`