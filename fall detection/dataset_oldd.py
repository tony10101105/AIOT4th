import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from sklearn.decomposition import PCA

class ACC(Dataset):
    def __init__(self, transform=None, mode = 'train'):
        self.transform = transform
        if mode == 'train':
            DATA_PATH = './data/fall_train.txt'
        else:
            DATA_PATH = './data/fall_test.txt'
        self.final = []
        temp = []
        with open(DATA_PATH, newline='\n') as file:
            for row in file.readlines():
                row = row.split(' ')
                try:
                    row = [float(i) for i in row]
                    row = row[:3]
                except:
                    print('row:', row)
                    raise Exception('format error')
                if row == [1,1,1] or row == [0,0,0]:
                    self.final.append((temp, row))
                    temp = []
                else:
                    temp.append(row)
            '''
            print(len(input_data[1]))
            pca = PCA(2, copy = True)
            pca_data = pca.fit_transform(input_data)

            plt.figure()
            plt.plot(pca_data)
            plt.show()
            '''
            
    def __getitem__(self, index):
        data, label = self.final[index]
        if label == [1,1,1]:
            label = torch.tensor(1.0)
        elif label == [0,0,0]:
            label = torch.tensor(0.0)
        else:
            raise Exception('label assignment error')
        
        if len(data) == 100:
            del data[0]
        
        data = np.array(data)
        data = self.transform(data)
        
        return data, label
        
    def __len__(self):
        return len(self.final)
