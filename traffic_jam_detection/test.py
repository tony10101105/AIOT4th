import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import dataset
from models import tj
from utils import get_road_info_dict, load_checkpoint, get_MCC, get_accuracy, get_precision, get_recall, get_euclidean_distance


parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type = float, default = 0.5)
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--GPU", type = bool, default = False)
args = parser.parse_args()
print(args)

#for reproducibility
torch.manual_seed(0)

#load dataset
print('loading dataset...')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
test_data = dataset.tj(transform = transform, mode = 'test')
print('number of data points:', len(test_data))
testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = True)
print('dataset laoding finished')
#create and load model
tj = tj()
try:
    MODEL_PATH = './checkpoints/tj_40.pth'
    tj = load_checkpoint(MODEL_PATH, tj)
    tj.eval()
except:
    raise Exception('no model')

if args.GPU and torch.cuda.is_available():
    print('using GPU...')
    tj = tj.cuda()

total_count = 0
correct_count = 0
#start training
for i, (img, obj, hw, cp) in enumerate(testloader):
    print('obj:', obj[0].tolist())
    ob = (obj > 0).float()
    total_count += torch.sum(ob)
    mask = ob.unsqueeze(2).expand_as(hw)
    if args.GPU and torch.cuda.is_available():
        img = img.cuda()
        obj = obj.cuda()
        hw = hw.cuda()
        cp = cp.cuda()
        mask = mask.cuda()

    pred_cp, pred_hw, pred_obj = tj(img)
    pred_obj = (pred_obj > args.threshold).float()
    pred_mask = pred_obj.unsqueeze(2).expand_as(hw)
    hw = (hw * pred_mask).squeeze(0)
    pred_cp = (pred_cp * pred_mask).squeeze(0)
    road_info = get_road_info_dict()
    final_pred = []
    for i in range(len(pred_cp)):
        if (pred_cp[i][0] != 0) or (pred_cp[i][1] != 0):
            this_obj = 0
            min_distance = 100000000
            for key in road_info.keys():
                e_distance = get_euclidean_distance(pred_cp[i], road_info[key][-1])
                if e_distance <= min_distance:
                    min_distance = e_distance
                    this_obj = key
            final_pred.append(this_obj)
    
    print('final_pred:', final_pred)
    obj_list = obj[0].tolist()
    for i in range(len(obj_list)):
        if obj_list[i] in final_pred:
            correct_count += 1

    #print('accuracy:', correct_count / total_count)

'''print('two classification accuracy: {:.4f}'.format(get_accuracy(TP, FP, FN, TN)))
print('MCC: {:.4f}'.format(get_MCC(TP, FP, FN, TN)))
print('precision: {:.4f}'.format(get_precision(TP, FP, FN, TN)))
print('recall: {:.4f}'.format(get_recall(TP, FP, FN, TN)))'''


    
    



