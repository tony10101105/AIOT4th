import torch
import torchvision
import torch.nn as nn
from models import tj

def get_road_info_dict():
    road_info_path = './edge.txt'
    road_info_dict = {}
    road_info = open(road_info_path, 'r')
    for i, row in enumerate(road_info.readlines(), 1):
        row = row.split(' ')
        row[0] = i
        row = [float(j) for j in row]
        row.append([(row[4]+row[2])/2, (row[3]+row[1])/2])#add h and w
        road_info_dict[int(row[0])] = row[1:]
    road_info.close()
    return road_info_dict

def get_euclidean_distance(a, b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**(1/2)


def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model

def get_MCC(TP, FP, FN, TN):
    MCC = (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(1/2)
    return MCC

def get_accuracy(TP, FP, FN, TN):
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    return accuracy

def get_precision(TP, FP, FN, TN):
    precision = TP/(TP+FP)
    return precision

def get_recall(TP, FP, FN, TN):
    recall = TP/(TP+FN)
    return recall
