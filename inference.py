import argparse

import numpy as np
from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn
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

def infer(MAX_LENGTH=1000):
    parser = argparse.ArgumentParser(prog='Path', epilog='Text at the bottom of help')
    parser.add_argument('pcmName')
    args = parser.parse_args()

    # Load the model (specify the map_location parameter to load on CPU)
    model = torch.load('model.pt', map_location=torch.device('cpu'))

    device = torch.device('cpu')
    model = model.to(device)

    # Load and process the PCM file
    pcmfile = load_file(args.pcmName)
    if pcmfile.size(1) > MAX_LENGTH:
        pcmfile = pcmfile[:, 10000:10000 + MAX_LENGTH, :]
    pcmfile = pcmfile.to(device)

    # Perform the inference
    output = model(pcmfile)
    print(f"The prediction of this room is: Room {(output)}")

infer()
