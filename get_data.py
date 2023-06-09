import os
import torch
from torch.utils.data import Dataset, DataLoader

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import torch.nn.functional as F


class highpassfilter():
    def __init__(self):
        self.frequency = 63333;

    def butter_highpass_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        high = 10000 / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        c, d = signal.butter(order, high, btype='low', analog=False)
        return b, a, c, d

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a, c, d = self.butter_highpass_lowpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        y = signal.filtfilt(c, d, y)
        return y


class PCMDataSet(Dataset):
    def __init__(self, folder_path):
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        self.labels = [self.get_label(file) for file in os.listdir(folder_path)]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        npz_path = self.file_paths[index]
        label = self.labels[index]
        with open(npz_path, 'rb') as f:
            stft_data = np.load(npz_path)
        #
        # hi = highpassfilter();
        # pcm_data = hi.butter_highpass_filter(pcm_data, 2000, 63333) #filter the human speaking noise
        # index_f = 0
        # amplitude = 10000
        # while index_f > 16000 or index_f < 7000:
        #     index_f = np.argmax(pcm_data > amplitude)
        #     amplitude += 100
        #     if amplitude > 30000:
        #         break
        # max = np.max(pcm_data[index_f:index_f + 3000])
        # if np.max(pcm_data) != max:
        #     print(np.max(pcm_data) )
        # indices = np.where(pcm_data > max - 5000)[0]
        # indices = indices[indices < index_f + 3000]
        # id = np.max(indices)
        # waveform = torch.from_numpy(pcm_data.copy()[id:id + 3000]).float()
        # freq, t, stft = signal.spectrogram(waveform, fs=63333, mode='magnitude',nperseg=10,noverlap=1,nfft = 400)
        t = stft_data['arr_0'].T
        stft = torch.from_numpy(t).double()
        # print(stft.shape)

        # waveform = F.normalize(waveform, p=2.0, dim=0, eps=1e-12, out=None)
        # waveform = waveform.unsqueeze(1)
        # time_index = torch.arange(waveform.shape[0]).unsqueeze(1)
        # waveform = torch.cat((waveform, time_index), dim=1)
        # print(waveform)
        return stft, label

    def get_label(self, file):
        match file.split('_')[-1]:
            # case "5.npz":
            #     return torch.Tensor([1,0,0,0])
            # case "6.pcm":
            #     return torch.Tensor([0,1,0,0])
            # case "7.pcm":
            #     return torch.Tensor([0,0,1,0])
            # case "8.pcm":
            #     return torch.Tensor([0,0,0,1])
            case "1.npz":
                return torch.Tensor([0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "2.npz":
                return torch.Tensor([0.2, 0.6, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "3.npz":
                return torch.Tensor([0.05, 0.1, 0.6, 0, 0, 0, 0.1, 0.05, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "4.npz":
                return torch.Tensor([0, 0, 0, 0.8, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "5.npz":
                return torch.Tensor([0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "6.npz":
                return torch.Tensor([0, 0, 0, 0.1, 0.1, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "7.npz":
                return torch.Tensor([0, 0, 0.1, 0, 0, 0, 0.7, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "8.npz":
                return torch.Tensor([0, 0, 0.1, 0, 0, 0, 0.1, 0.7, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "9.npz":
                return torch.Tensor([0, 0, 0.1, 0, 0, 0, 0.1, 0.1, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "10.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            case "11.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0])
            case "12.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0.7, 0.15, 0, 0, 0, 0, 0, 0, 0])
            case "13.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0.7, 0.15, 0, 0, 0, 0, 0, 0])
            case "14.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.15, 0.6, 0.15, 0, 0, 0, 0, 0])
            case "15.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.6, 0.2, 0, 0, 0, 0])
            case "16.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.7, 0, 0, 0, 0])
            case "17.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.1, 0.2, 0])
            case "18.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.7, 0.2, 0])
            case "19.npz":
                return torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0.15, 0.7, 0])

#
# dataset = PCMDataSet("./0524_data")
# batch_size = 2  # Set your desired batch size
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
