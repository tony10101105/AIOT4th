import serial
import numpy as np
from time import sleep
import sys
import csv
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
from utils import load_checkpoint
from models import fallCNN

parser = argparse.ArgumentParser()
parser.add_argument("--GPU", type = bool, default = False)
args = parser.parse_args()
print(args)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

path = 'fallingtest.csv'
COM_PORT = '/dev/tty.HC-05-DevB'  # 請自行修改序列埠名稱
BAUD_RATES = 115200
#ser = serial.Serial(COM_PORT, BAUD_RATES, timeout=None)
ser = serial.Serial(COM_PORT, BAUD_RATES, bytesize=8, parity='N', stopbits=1, timeout=None, xonxoff=False, rtscts=False)

final = []
#create and load model
model = fallCNN()
try:
    MODEL_PATH = './checkpoints/fallCNN_4.pth'
    fallCNN = load_checkpoint(MODEL_PATH, model)
    fallCNN.eval()
except:
    raise Exception('no model')

if args.GPU and torch.cuda.is_available():
    print('using GPU...')
    fallCNN = fallCNN.cuda()
'''with open(path,'w',newline='') as f:
	csv_write = csv.writer(f,delimiter=' ')
	csv_head = ["AccX","AccY","AccZ","GyrX","GyrY","GyrZ"]
	csv_write.writerow(csv_head)'''
waste = ser.readline()
waste = 0
try:
	while True:
		start = 1
		mcu_feedback = ser.readline()  # 接收回應訊息並解碼
		xyz = str(mcu_feedback[:-2])
		#print(xyz)
		data = xyz.split(',')
		for i in range(len(data)):
			if i == 0:
				try:
					#print(data)
					data[i] = float(data[i][2:-1])
				except:
					data[i] = float(data[i][2:])
			elif i == len(data)-1:
				data[i] = float(data[i][:-1])
			else:
				data[i] = float(data[i])
		final.append(data)
		if len(final)==99:
			for i in range(len(final)):
				if len(final[i]) != 6:
					del final[i]
					start = 0
					break
			if start == 1:
				inp = np.array(final)
				inp = transform(inp)
				inp = inp.unsqueeze(0)
				inp = inp.float()
				pred = fallCNN(inp)
				if pred > 0.6:
					print('FALLLLLLLLLLL')
				else:
					print('WALKINGGGGGGG')
				final = final[20:]
				start = 0

			#ser.close()
			#print('再見！')



except KeyboardInterrupt:
	with open(path,'a+',newline='') as result_file:
		wr = csv.writer(result_file, delimiter=' ')
		wr.writerows(final)
	ser.close()
	print('再見！')
