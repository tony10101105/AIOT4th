import torch
import torch.nn as nn

class tj(nn.Module):
    def __init__(self):
        super(tj, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(3),
                                         nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16),
                                         nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32))
        self.cp_head =  nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels = 16, out_channels = 2, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True))
        self.cp_fc = nn.Linear(128, 8)
        self.hw_head =  nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels = 16, out_channels = 2, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True))
        self.hw_fc = nn.Linear(128, 8)
        self.obj = nn.Sequential(nn.Linear(16*16*32, 16*16), nn.ReLU(True), nn.Linear(16*16, 4), nn.Sigmoid())

    def init_weights(self):
        for _, m in self.conv1.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
                
        for _, m in self.conv2.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for _, m in self.cp_head.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for _, m in self.hw_head.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        pred_cp = self.cp_head(x)
        p_cp = pred_cp.view(x.shape[0], -1)
        p_cp = self.cp_fc(p_cp)
        pred_cp = p_cp.reshape(x.shape[0], 4, 2)

        pred_hw = self.hw_head(x)
        p_hw = pred_hw.view(x.shape[0], -1)
        p_hw = self.hw_fc(p_hw)
        pred_hw = p_hw.reshape(x.shape[0], 4, 2)

        x = x.view(x.shape[0], -1)
        pred_obj = self.obj(x)
        return pred_cp, pred_hw, pred_obj
        
        
