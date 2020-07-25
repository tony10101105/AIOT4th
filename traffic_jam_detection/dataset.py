import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from utils import get_road_info_dict
from time import sleep

class tj(Dataset):
    def __init__(self, transform=None, mode = 'train'):        
        self.max_obj = 4
        self.transform = transform
        self.images = []
        self.objects = []
        self.hw = []
        self.center_point = []
        
        if mode == 'train':
            DATA_IMAGE_PATH = './data/tj_train_image'
            DATA_TXT_PATH = './data/tj_train_txt'
        elif mode == 'test':
            DATA_IMAGE_PATH = './data/tj_test_image'
            DATA_TXT_PATH = './data/tj_test_txt'
            
        if mode == 'train':
            num = '0'
        elif mode == 'test':
            num = '510'
            
        c = 0
        while True:
            try:
                while len(num) < 4:
                    num = '0' + num
                IMAGE_PATH = os.path.join(DATA_IMAGE_PATH, num+'.jpg')
                TXT_PATH = os.path.join(DATA_TXT_PATH, num+'.txt')

                with Image.open(IMAGE_PATH) as img:
                    img = img.resize((64, 64), Image.ANTIALIAS)
                    img = self.transform(img)
                    self.images.append(img)
            
                if c % 10 == 0:
                    with open(TXT_PATH, newline='\n') as txtfile:
                        objects = []
                        count = 1
                        for row in txtfile.readlines():
                            for i in range(len(row)):
                                if row[i] == '1':
                                    objects.append(count)
                                count += 1
            
                    for i in range(self.max_obj):
                        if len(objects) < self.max_obj:
                            objects.append(0)

                    self.objects.append(objects)
                else:
                    self.objects.append(self.objects[-1])
                c += 1
                num = str(int(num)+1)
            except:
                break
        road_info_dict = get_road_info_dict()
        for i in self.objects:
            hw = []
            center_point = []
            for j in i:
                if j != 0:
                    road_endpoint_location = road_info_dict[j]
                    h = abs(road_endpoint_location[3] - road_endpoint_location[1])
                    w = abs(road_endpoint_location[2] - road_endpoint_location[0])
                    hw.append([h, w])
                    center_point.append([(road_endpoint_location[3] + road_endpoint_location[1])/2, (road_endpoint_location[2] + road_endpoint_location[0])/2])
                elif j == 0:
                    hw.append([0, 0])
                    center_point.append([0, 0])

            self.hw.append(hw)
            self.center_point.append(center_point)
            

            
    def __getitem__(self, index):
        img = self.images[index]
        img = img.float()
        obj = self.objects[index]
        obj = torch.tensor(obj)
        obj = obj.float()
        hw = self.hw[index]
        hw = torch.tensor(hw)
        hw = hw.float()
        cp = self.center_point[index]
        cp = torch.tensor(cp)
        cp = cp.float()
        
        return img, obj, hw, cp
        
    def __len__(self):
        assert len(self.objects) == len(self.hw) == len(self.center_point), 'data length error:{0}, {1}, {2}'.format(len(self.objects), len(self.hw), len(self.center_point))
        return len(self.images)
