import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# parameters
NUM = 0
repeating = 10
LR1 = 0.05
LR2 = 0.05
D = 0.5
DROPOUT = 0.5
EPOCH = 500
HIDDEN1 = 1000
HIDDEN2 = 500
HIDDEN3 = 200
HIDDEN4 = 100
alpha = 1
w = 0

predictions=[]

# load data
data = np.loadtxt('train_data.txt')

train_time_points = 50

train1 = data[1:train_time_points-1,:]
train2 = data[0:train_time_points-2,:]


INPUT_SIZE = train1.shape[1]

label = np.loadtxt('label.txt')
label = label[2+int(NUM):,]
label = label.reshape(train_time_points-2,1)

#.cuda().to(device)
Train1 = torch.tensor(train1,dtype=torch.float64)
Train2 = torch.tensor(train2,dtype=torch.float64)

Label = torch.tensor(label,dtype=torch.float64)
Pred = data[train_time_points-2:train_time_points,:]
Pred = torch.tensor(Pred)

# define model
class NN(nn.Module):
	def __init__(self, INPUT_SIZE, HIDDNE1, HIDDEN2, HIDDEN3, HIDDEN4):
		super(NN, self).__init__()
		self.d1 = nn.Dropout(p = D)
		self.d = nn.Dropout(p = DROPOUT)
		self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1)
		self.fc1_bn = nn.BatchNorm1d(HIDDEN1)
		self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
		self.fc2_bn = nn.BatchNorm1d(HIDDEN2)
		self.fc3 = nn.Linear(HIDDEN2, HIDDEN3)
		self.fc3_bn = nn.BatchNorm1d(HIDDEN3)
		self.fc4 = nn.Linear(HIDDEN3, HIDDEN4)
		self.fc4_bn = nn.BatchNorm1d(HIDDEN4)
		self.output = nn.Linear(HIDDEN4, 1)
		self.relu = nn.ReLU()
		
	def forward(self, x):
		input_data = self.d1(x)
		out = self.fc1(input_data)
		out = self.fc1_bn(out)
		out = self.relu(out)
		out = self.d(out)
		out = self.fc2(out)
		out = self.fc2_bn(out)
		out = self.relu(out)
		out = self.d(out)
		out = self.fc3(out)
		out = self.fc3_bn(out)
		out = self.relu(out)
		out = self.d(out)
		out = self.fc4(out)
		out = self.fc4_bn(out)
		out = self.relu(out)
		out = self.d(out)
		out = self.output(out)
		return out
for index in range(repeating):
	model1 = NN(INPUT_SIZE, HIDDEN1, HIDDEN2, HIDDEN3, HIDDEN4)
	model1 = model1.double()
	optimizer1 = torch.optim.Adam(model1.parameters(), lr = LR1, weight_decay = w)


	model2 = NN(INPUT_SIZE, HIDDEN1, HIDDEN2, HIDDEN3, HIDDEN4)
	model2 = model2.double()
	optimizer2 = torch.optim.Adam(model2.parameters(), lr = LR2, weight_decay = w)

	loss_function = nn.MSELoss()

	for epoch in range(EPOCH):
		model1.train()
		model2.train()
		output1 = model1(Train1)
		output2 = model2(Train2)
		# loss function
		loss1 = loss_function(output1, Label) + alpha *  (output1 - output2).pow(2).sum().item()/output1.shape[0]
		loss2 = loss_function(output2, Label) + alpha * (output1 - output2).pow(2).sum().item()/output1.shape[0]


		optimizer1.zero_grad()
		loss1.backward()
		optimizer1.step()

		optimizer2.zero_grad()
		loss2.backward()
		optimizer2.step()
		
		model1.eval()
		model2.eval()
		pred1 = model1(Pred)
		pred2 = model2(Pred)
		print('---------------------')
		print(epoch)
		#print((pred1[1].item()+pred2[0].item())/2.0)
		del loss1
		del loss2

	result = (pred1[1].item()+pred2[0].item())/2.0
	print('Prediction result: '+str(result))
	predictions.append(result)
	#np.savetxt('1/result_'+str(index+1)+'.txt',result)

print('Prediction result(repeat 10 times): '+str(sum(predictions)/len(predictions)))

del Train1
del Train2
del Label
del Pred
