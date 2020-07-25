import torch
import torch.nn as nn

class fallCNN(nn.Module):
    def __init__(self):
        super(fallCNN, self).__init__()
        # 3*99 to 2*50
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(3),
                                         nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16))
        # 2*49 to 1*25
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16),
                                         nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32))
        self.fc_layer = nn.Sequential(nn.Linear(1600, 1), nn.Sigmoid())

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

        for _, m in self.fc_layer.named_modules():
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
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc_layer(x)
        return x
        
        
