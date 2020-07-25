import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import dataset
from models import tj


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 4)
parser.add_argument("--n_epochs", type = int, default = 100)
parser.add_argument("--lr", type = float, default = 2e-3)
parser.add_argument("--init_weights", type = bool, default = False)
parser.add_argument("--GPU", type = bool, default = False)
args = parser.parse_args()
print(args)

#for reproducibility
torch.manual_seed(0)

#load dataset
print('loading dataset...')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
train_data = dataset.tj(transform = transform, mode = 'train')
print('number of data points:', len(train_data))
trainloader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True)
print('dataset laoding finished')
print('data length: ', len(trainloader))

#create model
tj = tj()
if args.init_weights:
    tj.init_weights()

optimizer = torch.optim.Adam(tj.parameters(), lr = args.lr, betas = [0.5, 0.999])
current_epoch = 1

if args.GPU and torch.cuda.is_available():
    print('using GPU...')
    tj = tj.cuda()

#loss function settings
criterionL1 = nn.L1Loss()
criterionBCE = nn.BCELoss()

if current_epoch >= args.n_epochs:
    raise Exception('training already finished!')
else:
    print('start training!')

#start training
for epoch in range(current_epoch, args.n_epochs+1):
    print('current_epoch:', epoch)
    for i, (img, obj, hw, cp) in enumerate(trainloader, 1):
        '''print('obj before:', obj)
        print('obj after:', obj > 0)'''
        obj = (obj > 0).float()
        mask = obj.unsqueeze(2).expand_as(hw)
        if args.GPU and torch.cuda.is_available():
            img = img.cuda()
            obj = obj.cuda()
            hw = hw.cuda()
            cp = cp.cuda()
            mask = mask.cuda()

        #train tjCNN
        pred_cp, pred_hw, pred_obj = tj(img)
        #print('pred_obj:', pred_obj)
        cp_loss = criterionL1(pred_cp * mask, cp * mask)
        hw_loss = criterionL1(pred_hw * mask, hw * mask)
        obj_loss = criterionBCE(pred_obj, obj)
        loss = (0.1*cp_loss + 0.1*hw_loss + obj_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            print('loss:', loss.mean())


    torch.save(tj.state_dict(), './checkpoints/tj_{}.pth'.format(epoch))
    print('model at {}th epoch is saved'.format(epoch))


