import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
import random
import pandas as pd
import numpy as np
sys.path.append(".")
from utils import *
from model import *

#setting
#dataframe_WAFF-2019-07.pkl
seq_length = 13
tahun = 2019
pattern_file = 'dataframe_WAFF-%s-%s.pkl'
input_channels = 1
hidden_layer = 10
hidden_neuron = 20
n_classes = 1
batch_size = 32
#seq_length = 20
epochs = 100
len_feature = 10
total_train = 5000
total_test = 500
epochs = 20
dropout = 0.05
kernel_size = 5
lr = 4e-3
test_data = 500

def train(epoch):
	global lr
	model.train()
	batch_idx = 1
	total_loss = 0
	for i in range(0, training_param_norm.size(0), batch_size):
		if i + batch_size > training_param_norm.size(0):
			x, y = training_param_norm[i:], training_label_norm[i:]
		else:
			x, y = training_param_norm[i:(i+batch_size)], training_label_norm[i:(i+batch_size)]
		optimizer.zero_grad()
		output = model(x)
		loss = F.mse_loss(output, y)
		loss.backward()
		optimizer.step()
		batch_idx += 1
		total_loss += loss.item()

		if batch_idx % 100 == 0:
			cur_loss = total_loss / 100
			processed = min(i+batch_size, training_param_norm.size(0))
			print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
				epoch, processed, training_param_norm.size(0), 100.*processed/training_param_norm.size(0), lr, cur_loss))
			total_loss = 0

def evaluate():
	model.eval()
	with torch.no_grad():
		output = model(test_param_norm)
		test_loss = F.mse_loss(output, test_label_norm)
		print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
		return test_loss.item()

#get data
all_kec_angin = []
for i in range(1,13):
	if i < 10:
		bulan = '0'+str(i)
	else:
		bulan = str(i)
	file_access = pattern_file%(tahun, bulan)
	try:
		file_access_open = pd.read_pickle(file_access)
		kec_angin = file_access_open['kecepatan angin']
		if i == 1:
			all_kec_angin = kec_angin
		else:
			all_kec_angin = np.hstack((all_kec_angin, kec_angin))
	except:
		continue

print(all_kec_angin, np.shape(all_kec_angin))

all_param = []
all_label = []
for i in range(len(all_kec_angin)-seq_length):
	seq_data = all_kec_angin[i:i+seq_length]
	if np.isnan(seq_data).any():
		print('there is nan in seq data, skip...')
		continue
	all_label.append(seq_data[-1])
	all_param.append(seq_data[:-1])

all_param = np.array(all_param)
all_label = np.array(all_label)

all_param_train = all_param[:-test_data]
all_label_train = all_label[:-test_data]

all_param_test = all_param[-test_data:]
all_label_test = all_label[-test_data:]

#print('shape train', np.shape(all_param_train))

#normalization
rata_param_train = np.mean(all_param_train)
rata_label_train = np.mean(all_label_train)

std_param_train = np.std(all_param_train)
std_label_train = np.std(all_label_train)

training_param_norm = ((all_param_train - rata_param_train) / std_param_train)
training_label_norm = ((all_label_train - rata_label_train) / std_label_train)

test_param_norm = ((all_param_test - rata_param_train) / std_param_train)
test_label_norm = ((all_label_test - rata_label_train) / std_label_train)


#convert to torch
training_param_norm = torch.from_numpy(training_param_norm).double()
training_label_norm = torch.from_numpy(training_label_norm).double()
test_param_norm = torch.from_numpy(test_param_norm).double()
test_label_norm = torch.from_numpy(test_label_norm).double()


training_param_norm = training_param_norm.cuda()
training_label_norm = training_label_norm.cuda()
test_param_norm = test_param_norm.cuda()
test_label_norm = test_label_norm.cuda()

#reshape
training_param_norm = torch.reshape(training_param_norm, (-1, 1, seq_length-1))
training_label_norm = torch.reshape(training_label_norm, (-1, 1))
test_param_norm = torch.reshape(test_param_norm, (-1, 1, seq_length-1))
test_label_norm = torch.reshape(test_label_norm, (-1, 1))

#model section
channel_sizes = [hidden_neuron]*hidden_layer
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout).double()

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for ep in range(1, epochs+1):
	train(ep)
	tloss = evaluate()

y_pred = model(test_param_norm)
y_pred = y_pred.cpu().detach().numpy()
y_pred = np.reshape(y_pred, (-1))
y_pred = y_pred * std_label_train + rata_label_train
print(np.shape(y_pred), y_pred)

test_label_norm = torch.reshape(test_label_norm, (-1,))
test_label_norm = test_label_norm.cpu().detach().numpy()
test_label = test_label_norm * std_label_train + rata_label_train
dataframe = {'label': test_label, 'prediksi': y_pred}
df = pd.DataFrame(data=dataframe)
df.to_excel("wind.xlsx") 
