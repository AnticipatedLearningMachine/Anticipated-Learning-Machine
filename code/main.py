import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 47
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(SEED)


# initial parameters
LR1 = 0.05
LR2 = 0.05
EPOCH = 300
alpha = 1
w = 0
D = 0.5
DROPOUT = 0.5
activate = nn.Tanh()
'''
period --- used when variables need to be divided into several segmants in prediction
e.g. 
a variable of n time points is divided into 2 segmants with n/2 time points to predict
in the prediction of the first segmants, period=0, and we get the prediction results(called P1)

in the prediction of the second segmants, replace the corresponding part in 'label.txt' with the prediction results of the first segmants
let period = n/2, and we get the prediction results of the second segmants(called P2)

'''
period = 0
# length of training time points
train_time_points = 30


'''
 load training data
 rows --- time points
 columns --- variables
'''
data = np.loadtxt('train.txt')

'''
Train1 --- train data of model1
Train2 --- train data of model2
'''
train1 = data[1:train_time_points-1,:]
train2 = data[0:train_time_points-2,:]
Train1 = torch.tensor(train1,dtype=torch.float64)
Train2 = torch.tensor(train2,dtype=torch.float64)
INPUT_SIZE = train1.shape[1]

# load label
Label = np.loadtxt('label.txt')

Pred = data[train_time_points - 2:train_time_points, ]
Pred = torch.tensor(Pred).to(device)

# Sampling operation
def newdropout(data, subspace):
    result = np.zeros_like(data)
    shuffled_indices=np.random.permutation(data.shape[1])
    indices =shuffled_indices[:subspace]
    result[:, indices] = data[:, indices]
    return result

# Framework of neural networks
class NN(nn.Module):
    def __init__(self, units):
        super(NN, self).__init__()
        self.layers = self._make_layer(units)

    def forward(self, x):
        predict = self.layers(x)
        return predict

    def _make_layer(self, units):
        layers = []
#         layers.append(nn.Dropout(p=D))
        for i in range(len(units)-1):
            layers += [nn.Linear(units[i], units[i+1]), nn.BatchNorm1d(units[i+1]), activate, nn.Dropout(p=DROPOUT)]
        layers += [nn.Linear(units[-1], 1)]
        return nn.Sequential(*layers)

units = [INPUT_SIZE]+[200, 150, 100, 50, 25]
net = NN(units)
#print(net)


import matplotlib.pyplot as plt
# plot some curves 
def show_multi_curve(ys, title, legends, xxlabel, yylabel, if_point = False):
    x = np.array(range(len(ys[0])))
    for i in range(len(ys)):
        if if_point:
            plt.plot(x, ys[i], label = legends[i], marker = 'o')
        else:
            plt.plot(x, ys[i], label = legends[i])   
    plt.axis()
    plt.title(title)
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.legend()
    plt.show()

import math
def MAE(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

def RMSE(y_true, y_pred):
    n = len(y_true)
    arr = y_true - y_pred
    mse = 0
    for each in arr:
        mse = mse + math.pow(each, 2)
    mse = mse / n
    return math.sqrt(mse)


def evaluate(output1, output2, Label, loss_function):
    loss1 = loss_function(output1, Label)
    loss2 = loss_function(output1, output2)
    return loss1, loss2, loss1+alpha*loss2

# train model

def train(model1, model2, Label, loss_function, optimizer1,
          optimizer2):
    error = 0
    model1 = model1.double()
    model2 = model2.double()
    
    losses1 = []
    losses2 = []
    losses11 = []
    losses12 = []
    losses21 = []
    losses22 = []
    modelpreds=[]

#     evals1 = []
#     evals2 = []
    for epoch in range(EPOCH):
        model1.train()
        model2.train()
        train1_dropout_data = torch.tensor(newdropout(Train1.numpy(), int(INPUT_SIZE*D)),dtype=torch.float64)
        train2_dropout_data = torch.tensor(newdropout(Train2.numpy(), int(INPUT_SIZE*D)),dtype=torch.float64)
        train1_dropout_data = train1_dropout_data.to(device)
        train2_dropout_data = train2_dropout_data.to(device)
        output1 = model1(train1_dropout_data)
        output2 = model2(train2_dropout_data)
        # loss function
        loss11, loss12, loss1 = evaluate(output1, output2, Label, loss_function)
        optimizer1.zero_grad()
        loss1.backward()

        optimizer1.step()
        output1 = model1(train1_dropout_data)
        output2 = model2(train2_dropout_data)
        
        loss21, loss22, loss2 = evaluate(output2, output1, Label, loss_function)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        losses1.append(loss1.item())
        losses11.append(loss11.item())
        losses12.append(loss12.item())
        losses2.append(loss2.item())
        losses21.append(loss21.item())
        losses22.append(loss22.item())
        if epoch == EPOCH-1:
            error = torch.mean((abs(output1-Label)+abs(output2-Label))/2).item()
        model1.eval()
        model2.eval()
        pred1 = model1(Pred)
        pred2 = model2(Pred)
        modelpreds.append((pred1[1].item() + pred2[0].item()) / 2.0)

    return error, losses11, losses12, losses1, losses21, losses22, losses2,modelpreds

from scipy.stats import pearsonr

def fit(steps, repeating = 10):
    final_predicts = np.array([0.0]*steps)
    for j in range(repeating):
        label = Label[2+period:train_time_points+period, ]
        label = list(label)
        targets = []
        predicts = []
        errors = []
        for index in range(steps):
            LABEL = np.array(label).reshape(train_time_points - 2, 1)
            LABEL = torch.tensor(LABEL, dtype=torch.float64).to(device)
            model1 = NN(units)
            model1 = model1.to(device)
            optimizer1 = torch.optim.Adam(model1.parameters(),
                                          lr=LR1,
                                          weight_decay=w)
            model2 = NN(units)
            model2 = model2.to(device)
            optimizer2 = torch.optim.Adam(model2.parameters(),
                                          lr=LR2,
                                          weight_decay=w)
            loss_function = nn.MSELoss()
            error, losses11, losses12, losses1, losses21, losses22, losses2,modelpreds = train(model1, model2,
                                                 LABEL, loss_function, optimizer1,
                                                 optimizer2)
            
            show_multi_curve([losses11, losses12,  losses21, losses22],
                             "losses for the " + str(index + 1) + " th step",
                             ["losses11", "losses12",  "losses21", "losses22"], "EPOCH", "Value")
            show_multi_curve([losses1, losses2],
                             "train and test losses for the " + str(index + 1) + " th step",
                             ["train_losses1", "train_losses2"], "EPOCH", "Value")
            target = Label[train_time_points + index + period]
            targets.append(target)
            predicts.append(modelpreds[-1])
            errors.append(error)

            label.append(predicts[-1])
            label = label[1:]
            
        
        show_multi_curve([predicts, targets],
                         "predictions from the 1 th to the " + str(steps) + " th steps",
                         ["predictions", "targets"], "STEP", "Value", True)
        show_multi_curve([[abs(x-y) for x,y in zip(predicts, targets)], errors],
                         "errors from the 1 th to the " + str(steps) + " th steps",
                         ["prediction_errors", "train_errors"], "STEP", "Value", True)
        
        print(predicts, targets)
        predicts, target = np.array(predicts), np.array(targets)
        print('test MAE', MAE(predicts, targets))
        print('test RMSE', RMSE(predicts, targets))
        print('test pearsonr', pearsonr(predicts, targets))
        final_predicts += predicts
    final_predicts /= repeating
    print(final_predicts)
    np.savetxt('prediction.txt',final_predicts)
    print('test MAE', MAE(final_predicts, targets))
    print('test RMSE', RMSE(final_predicts, targets))
    print('test pearsonr', pearsonr(final_predicts, targets))

    print('EPOCH ---- ', EPOCH)
    print('LR1 ---- ', LR1)
    print('LR2 ---- ', LR2)
    print('D ---- ', D)
    print('DROPOUT ---- ', DROPOUT)
    print('alpha ---- ', alpha)
    print('w ---- ', w)
    print('activate ---- ', activate)


fit(10,10)
