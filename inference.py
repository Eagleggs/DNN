import argparse

import numpy as np
aimport torch
import torch.nn as nn

from get_data import highpassfilter


def load_file(pcm_path):
    with open(pcm_path, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=np.int16)
        # print(pcm_data)
    waveform = torch.from_numpy(np.copy(pcm_data)).float()
    waveform = waveform.unsqueeze(1)
    time_index = torch.arange(waveform.shape[0]).unsqueeze(1)
    waveform = torch.cat((waveform, time_index), dim=1)
    waveform = torch.unsqueeze(waveform,0)
    return waveform

def infer():
    parser = argparse.ArgumentParser(prog='Path', epilog='Text at the bottom of help')
    parser.add_argument('pcmName')
    args = parser.parse_args()

    # Load the model (specify the map_location parameter to load on CPU)
    model = torch.load('model_best_26_3.pt', map_location=torch.device('cpu'))

    device = torch.device('cpu')
    model = model.to(device)

    # Load and process the PCM file
    with open(args.pcmName, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=np.int16)
    hi = highpassfilter();
    pcm_data = hi.butter_highpass_filter(pcm_data, 18000, 63333)
    abs_array = np.abs(pcm_data)

    index_f = np.argmax(abs_array > 1000)
    max = np.max(abs_array[index_f:index_f + 3000])
    indices = np.where(abs_array > max / 4)[0]
    indices = indices[indices < index_f + 3000]
    id = np.max(indices)
    waveform = torch.from_numpy(pcm_data.copy()[id:id + 1500]).float()  # 5ms = 320 samples
    waveform = waveform.unsqueeze(1)
    time_index = torch.arange(waveform.shape[0]).unsqueeze(1)
    waveform = torch.cat((waveform, time_index), dim=1)
    waveform = torch.unsqueeze(waveform,0)
    # if pcmfile.size(1) > MAX_LENGTH:
    #     pcmfile = pcmfile[:, 10000:10000 + MAX_LENGTH, :]
    waveform = waveform.to(device)

    # Perform the inference
    output = model(waveform)
    print(f"The prediction of this room is: Room {(torch.argmax(output, dim=1) + 1)}")
    print(output)

infer()
