import os
import torch
from torch.utils.data import Dataset, DataLoader

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

count = [0,0,0,0]
class PCMDataSet(Dataset):
    def __init__(self, folder_path):
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        self.labels = [self.get_label(file) for file in os.listdir(folder_path)]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        pcm_path = self.file_paths[index]
        label = self.labels[index]
        with open(pcm_path, 'rb') as f:
            pcm_data = np.frombuffer(f.read(), dtype=np.int16)
            # print(pcm_data)
        waveform = torch.from_numpy(pcm_data).float()
        waveform = waveform.unsqueeze(1)
        time_index = torch.arange(waveform.shape[0]).unsqueeze(1)
        waveform = torch.cat((waveform,time_index),dim = 1)
        return waveform,label

    def get_label(self, file):
        match file.split('_')[-1]:
            case "13.pcm":
                return torch.Tensor([1,0,0,0])
            case "8.pcm":
                return torch.Tensor([0,1,0,0])
            case "2.pcm":
                return torch.Tensor([0,0,1,0])
            case "66.pcm":
                return torch.Tensor([0,0,0,1])
            case other:
                return torch.Tensor([0,0,0,0])

dataset = PCMDataSet("./0517_data")
batch_size = 2  # Set your desired batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)