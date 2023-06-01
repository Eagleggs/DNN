import argparse

import numpy as np
import torch
import torch.nn as nn

from get_data import highpassfilter
from transformerLite import TransformerLite


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
    model = TransformerLite(t=1500, k=2, heads=8)
    # Load the model (specify the map_location parameter to load on CPU)
    model.load_state_dict(torch.load('model_best_30_test.pt', map_location=torch.device('cpu')))

    device = torch.device('cpu')
    model = model.to(device)

    # Load and process the PCM file
    with open(args.pcmName, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=np.int16)
    hi = highpassfilter();
    pcm_data = hi.butter_highpass_filter(pcm_data, 14000, 63333)
    abs_array = np.abs(pcm_data)

    index_f = 0
    amplitude = 3000
    while index_f > 16000 or index_f < 8000:
        index_f = np.argmax(pcm_data > amplitude)
        amplitude += 100
        if amplitude > 4000:
            break
    max = np.max(pcm_data[index_f:index_f + 3000])
    if np.max(pcm_data) != max:
        print("1")
    indices = np.where(pcm_data > max / 5)[0]
    indices = indices[indices < index_f + 3000]
    id = np.max(indices)
    waveform = torch.from_numpy(pcm_data.copy()[id:id + 1500]).float()
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
