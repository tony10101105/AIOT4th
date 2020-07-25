import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import dataset
from models import fallCNN
from utils import load_checkpoint, get_MCC, get_accuracy, get_precision, get_recall


parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type = float, default = 0.35)
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--GPU", type = bool, default = False)
args = parser.parse_args()
print(args)

#for reproducibility
torch.manual_seed(0)

#load dataset
print('loading dataset...')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
test_data = dataset.ACC(transform = transform, mode = 'test')
print('number of data points:', len(test_data))
testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = True)
print('dataset laoding finished')
#create and load model
model = fallCNN()
try:
    MODEL_PATH = './fallCNN_4.pth'
    fallCNN = load_checkpoint(MODEL_PATH, model)
    fallCNN.eval()
except:
    raise Exception('no model')

if args.GPU and torch.cuda.is_available():
    print('using GPU...')
    fallCNN = fallCNN.cuda()

TP, FP, FN, TN = 0, 0, 0, 0
#start training
for i, (data, label) in enumerate(testloader):
    label = label.unsqueeze(1)
    data = data.float()
    if args.GPU and torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()
    output = fallCNN(data)
    print('label:', label.data, end = ' ')
    print('output:', output.data)
    #record the confusion matrix
    if (output > args.threshold and label == 1):
        TP += 1
    elif (output < args.threshold and label == 0):
        TN += 1
    elif (output > args.threshold and label == 0):
        FP += 1
    elif (output < args.threshold and label == 1):
        FN += 1

print('two classification accuracy: {:.4f}'.format(get_accuracy(TP, FP, FN, TN)))
print('MCC: {:.4f}'.format(get_MCC(TP, FP, FN, TN)))
print('precision: {:.4f}'.format(get_precision(TP, FP, FN, TN)))
print('recall: {:.4f}'.format(get_recall(TP, FP, FN, TN)))


    
    



