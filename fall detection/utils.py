import torch
import torchvision
import torch.nn as nn
from models import fallCNN


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
