#!/usr/bin/python
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_arch import *
import csv, sys, pdb

torch.manual_seed(777)

####################################################################################################
#Datasets
####################################################################################################
#file_acc_x = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/squeeze_release_X.csv"
#file_acc_y = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/squeeze_release_Y.csv"
#file_press = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/squeeze_release_P.csv"
#file_acc_x = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/ed_data_x.csv"
#file_acc_y = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/ed_data_y.csv"
#file_press = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/ed_pressure.csv"
file_acc_x = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/squeeze_X.csv"
file_acc_y = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/squeeze_Y.csv"
file_press = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/squeeze_P.csv"
#file_acc_x = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/fft_squeezeX.csv"
#file_acc_y = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/fft_squeezeY.csv"
#file_press = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/fft_squeezeP.csv"
file_label = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/labels.csv"
file_bilabel = "/home/ykhassanov/projects/tactile_sensors/lump_detection/dataset/binary_label.csv"

df_acc_x = pd.read_csv(file_acc_x, sep=',', header=None)
df_acc_y = pd.read_csv(file_acc_y, sep=',', header=None)
df_press = pd.read_csv(file_press, sep=',', header=None)
df_label = pd.read_csv(file_label, sep=',', header=None)
df_bilabel = pd.read_csv(file_bilabel, sep=',', header=None)
df_newlabel = df_bilabel.astype(str) + "_" + df_label.astype(str)

#df_all = pd.concat([df_acc_x, df_acc_y, df_press, df_label], axis=1, sort=False)
#df_all = pd.concat([df_press, df_label], axis=1, sort=False)
df_all = pd.concat([df_acc_x, df_bilabel, df_label, df_newlabel], axis=1, sort=False)
#df_all = pd.concat([df_acc_y, df_label], axis=1, sort=False)
#df_all = pd.concat([df_press, df_label], axis=1, sort=False)
#df_all = pd.concat([df_acc_x, df_press, df_label], axis=1, sort=False)
#df_all = pd.concat([df_acc_x, df_acc_y, df_label], axis=1, sort=False)

df_train, df_eval = train_test_split(df_all, test_size=1.0/3, random_state=777, stratify=df_newlabel.values)
df_dev, df_test   = train_test_split(df_eval, test_size=0.5, random_state=777, stratify=df_eval.values[:,-1])

#save data splitting
tmp = df_train.iloc[:,:-2]
tmp.to_csv(path_or_buf="./dataset/df_train1.csv", index=False)
tmp = df_dev.iloc[:,:-2]
tmp.to_csv(path_or_buf="./dataset/df_dev1.csv", index=False)
tmp = df_test.iloc[:,:-2]
tmp.to_csv(path_or_buf="./dataset/df_test1.csv", index=False)

if False:
  # Use FFT based features
  #fft_size = 512
  fft_size = df_train.shape[1]-3
  train_x = np.fft.fft(df_train.values[:,:-3], fft_size, 1)
  train_x = np.abs(train_x)
  train_x	= torch.FloatTensor(train_x)

  dev_x = np.fft.fft(df_dev.values[:,:-3], fft_size, 1)
  dev_x = np.abs(dev_x)
  dev_x = torch.FloatTensor(dev_x)

  test_x = np.fft.fft(df_test.values[:,:-3], fft_size, 1)
  test_x = np.abs(test_x)
  test_x = torch.FloatTensor(test_x)
else:
  # Use raw features
	train_x	= torch.FloatTensor(df_train.values[:,:-3].astype('float64'))
	dev_x = torch.FloatTensor(df_dev.values[:,:-3].astype('float64'))
	test_x = torch.FloatTensor(df_test.values[:,:-3].astype('float64'))

train_y = torch.FloatTensor(df_train.values[:,-3].astype('float64'))
dev_y = torch.FloatTensor(df_dev.values[:,-3].astype('float64'))
test_y = torch.FloatTensor(df_test.values[:,-3].astype('float64'))

####################################################################################################
#Model
####################################################################################################
dtype = torch.float
#device = torch.device("cpu") # Uncomment this to run on CPU
if len(sys.argv) > 1:
	gpu_id=sys.argv[1]
	device = torch.device("cuda:"+gpu_id)
else:
	device = torch.device("cuda:0")
print(device)

epochs = 10000
N = 16 #batch size
D_in = train_x.shape[1]
D_out = 1
drop = 0.4
#FFN Model
H = 512 #hidden size
#model = TwoLayerNet(D_in, H, D_out, drop).to(device)
#model_path = "./model/biclass_ffn_l2_h"+str(H)+"_bs"+str(N)+"_drop"+str(drop)+"_epoch"+str(epochs)+"_accX"
#model_path = "./model/biclass_ffn_l3_h"+str(H)+"_bs"+str(N)+"_drop"+str(drop)+"_epoch"+str(epochs)+"_fftX"

#CNN Model
ch_out = 4
kernel_w = 16
stride = 2
W_in = train_x.shape[1]
model = ConvNet(1, W_in, D_out, ch_out, H, kernel_w, stride, drop).to(device)
#model_path = "./model/biclass_cnn_l1_h"+str(H)+"_bs"+str(N)+"_c"+str(ch_out)+"_k"+str(kernel_w)+"_s"+str(stride)+"_drop"+str(drop)+"_epoch"+str(epochs)+"_accX"
model_path = "./model/biclass_cnn_l1_h"+str(H)+"_bs"+str(N)+"_c"+str(ch_out)+"_k"+str(kernel_w)+"_s"+str(stride)+"_drop"+str(drop)+"_epoch"+str(epochs)+"_fttX"
print(model)

loss_fn = torch.nn.BCELoss()

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

####################################################################################################
#Training
####################################################################################################
dev_acc_max=0
epoch_max=0
for e in range(epochs):
	total_loss = 0
	#for x, y in zip(train_x, train_y):
	for i in range(0, train_x.shape[0], N):
		if i+N >= train_x.shape[0]:
			x = train_x[i:]
			y = train_y[i:]
		else:
			x = train_x[i:i+N]
			y = train_y[i:i+N]
		y_pred = model(x.to(device))
		loss = loss_fn(y_pred, y.view(-1,1).to(device))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	print("Epoch {}/{}, Train Loss: {:.3f}".format(e+1, epochs, total_loss))
	#pdb.set_trace()
	#Loss and accuracy on dev data
	with torch.no_grad():
		model.eval()
		output = model(dev_x.to(device))
		output = (output>0.5).float()
		acc    = accuracy_score(dev_y, output.cpu())
		if acc > dev_acc_max:
			dev_acc_max=acc
			epoch_max=e+1
			torch.save(model, model_path)
      #correct = (output == dev_y.to(device)).float().sum()
		model.train()
	print("Dev Accuracy: {:.3f}".format(acc))

print("##############################################")
print("Best dev accuracy is {:.3f} at epoch {}".format(dev_acc_max, epoch_max))
print("Number of class 1 samples: ", (dev_y>0.5).float().sum().item())
print("##############################################")

####################################################################################################
#Evaluation
####################################################################################################
with torch.no_grad():
  pdb.set_trace()
  model = torch.load(model_path)
  model.eval().to(device)
  output = model(test_x.to(device))
  output = (output>0.5).float()
  acc    = accuracy_score(test_y, output.cpu())
  print("Test Accuracy: {:.3f}".format(acc))
  print(model)

exit()
