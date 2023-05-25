import os
import torch
from torch.utils.data import Dataset, DataLoader

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal

class highpassfilter():
    def __init__(self):
        self.frequency = 63333;

    def butter_highpass(self,cutoff,fs, order=5):
        nyq = 0.5 *fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self,data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

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
        hi = highpassfilter();
        pcm_data = hi.butter_highpass_filter(pcm_data,18000,63333)
        abs_array = np.abs(pcm_data)
        index = np.argmax(abs_array > 1000) + 320
        waveform = torch.from_numpy(pcm_data.copy()[index:index+1500]).float() #5ms = 320 samples
        waveform = waveform.unsqueeze(1)
        time_index = torch.arange(waveform.shape[0]).unsqueeze(1)

        waveform = torch.cat((waveform,time_index),dim = 1)
        # print(waveform)
        return waveform,label

    def get_label(self, file):
        match file.split('_')[-1]:
            case "1.pcm":
                return torch.Tensor([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            case "2.pcm":
                return torch.Tensor([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            case "3.pcm":
                return torch.Tensor([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            case "4.pcm":
                return torch.Tensor([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            case "5.pcm":
                return torch.Tensor([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            case "6.pcm":
                return torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            case "7.pcm":
                return torch.Tensor([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
            case "8.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
            case "9.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
            case "10.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
            case "11.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
            case "12.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
            case "13.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
            case "14.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
            case "15.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
            case "16.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
            case "17.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
            case "18.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
            case "19.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
            case "20.pcm":
                return torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])



#
# dataset = PCMDataSet("./0524_data")
# batch_size = 2  # Set your desired batch size
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)