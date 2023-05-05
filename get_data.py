import os
import torch


def get_data(batch_size, len):
    data_path = "/home/lu/Desktop/SMS_data_sample/data_sample.txt"
    recordinds_path ="/home/lu/Desktop/SMSS_recordings"
    recordinds_iterator = (os.path.join(recordinds_path, file_name) for file_name in os.listdir(recordinds_path) if
                      os.path.isfile(os.path.join(recordinds_path, file_name)))
    for path in recordinds_iterator:
        with open(path,'rb') as rec:
            binaray_data = rec.read(len)
            print(binaray_data)
            with open(data_path,'wb') as data:
                data.write(binaray_data)


get_data(0, pow(2,15))
