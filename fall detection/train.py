import torch
import torch.nn as nn
ffrom torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import dataset
from models import fallCNN


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 4)
parser.add_argument("--n_epochs", type = int, default = 50)
parser.add_argument("--lr", type = float, default = 1e-2)
parser.add_argument("--init_weights", type = bool, default = False)
parser.add_argument("--GPU", type = bool, default = False)
args = parser.parse_args()
print(args)

#for reproducibility
torch.manual_seed(0)

#load dataset
print('loading dataset...')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_data = dataset.ACC(transform = transform)
print('number of data points:', len(train_data))
trainloader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True)
print('dataset laoding finished')

#create model
fallCNN = fallCNN()
if args.init_weights:
    fallCNN.init_weights()
optimizer = torch.optim.Adam(fallCNN.parameters(), lr = args.lr, betas = [0.5, 0.999])
current_epoch = 0

if args.GPU and torch.cuda.is_available():
    print('using GPU...')
    fallCNN = fallCNN.cuda()

#loss function settings
criterionL1 = nn.L1Loss()
criterionBCE = nn.BCELoss()

if current_epoch >= args.n_epochs:
    raise Exception('training already finished!')
else:
    print('start training!')

#start training
for epoch in range(current_epoch, args.n_epochs):
    print('current_epoch:', epoch+1)
    for i, (data, label) in enumerate(trainloader):
        label = label.unsqueeze(1)
        data = data.float()
        if args.GPU and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        output = fallCNN(data)
        #print('output:', output)
        loss = criterionBCE(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:', loss.mean())

    torch.save(fallCNN.state_dict(), './checkpoints/fallCNN_{}.pth'.format(epoch+1))
    print('model at {}th epoch is saved'.format(epoch+1))


