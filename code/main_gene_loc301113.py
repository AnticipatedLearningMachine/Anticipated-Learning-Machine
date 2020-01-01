import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
# fix random number to ensore the reliability of our results
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 47
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(SEED)

#initial parameters
# LR1,LR2 --- 0.05
# D --- 0.8~0.9
# DROPOUT --- 0~0.1
# EPOCH --- 100
# alpha --- 1~10
LR1 = 0.05
LR2 = 0.05
D = 0.8
DROPOUT = 0.
EPOCH = 100
alpha = 10
w = 0
activate = nn.ReLU()

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
train_time_points = 11


data = np.loadtxt('LOC301113.txt')
data=data.transpose()

'''
Train1 --- train data of model1
Train2 --- train data of model2
'''
train1 = data[1:train_time_points-1,:]
train2 = data[0:train_time_points-2,:]
Train1 = torch.tensor(train1,dtype=torch.float64)
Train2 = torch.tensor(train2,dtype=torch.float64)
INPUT_SIZE = train1.shape[1]
Label = np.loadtxt('label.txt')

# used as the input of modlels to get final result
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
        for i in range(len(units)-1):
            layers += [nn.Linear(units[i], units[i+1]), nn.BatchNorm1d(units[i+1]), activate, nn.Dropout(p=DROPOUT)]
        layers += [nn.Linear(units[-1], 1)]
        return nn.Sequential(*layers)

units = [INPUT_SIZE]+[500,200,150,50]

# plot function
import matplotlib.pyplot as plt
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

# measures of results
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

# evaluate results and compute loss
def evaluate(output1, output2, Label, loss_function):
    loss1 = loss_function(output1, Label)
#     loss2 = (output1 - output2).pow(2).sum().item()/output1.shape[0]
    loss2 = loss_function(output1, output2)
    return loss1, loss2, loss1+alpha*loss2


# train model bu pairwise scheme
def train(model1, model2, Label, loss_function, optimizer1,
          optimizer2,j,index):
    error = 0
    model1 = model1.double()
    model2 = model2.double()
    
    losses1 = []
    losses2 = []
    losses11 = []
    losses12 = []
    losses21 = []
    losses22 = []
    modelpreds = []
    preds1 = []
    preds2 = []
    grad0 = []
    grad1 = []
    grad2 = []
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
        
        grad0.append(torch.mean(model1.layers[0].weight.grad.data).item())
        grad1.append(torch.mean(model1.layers[4].weight.grad.data).item())
        grad2.append(torch.mean(model1.layers[-1].weight.grad.data).item())
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
        preds1.append(pred1[1].item())
        preds2.append(pred2[0].item())
    # save the models
    save_path = './models/'+str(j)+'_'+str(index+period)+'.pt'
    torch.save(model1.state_dict(), save_path)
    save_path = './models/'+str(j)+'_'+str(index+1+period)+'.pt'
    torch.save(model1.state_dict(), save_path)
    return error, losses11, losses12, losses1, losses21, losses22, losses2, modelpreds, preds1, preds2, [
        grad0, grad1, grad2
    ]
    return error, losses11, losses12, losses1, losses21, losses22, losses2, modelpreds, preds1, preds2, [grad0, grad1, grad2]

from scipy.stats import pearsonr

# predict results at #steps time points with #repeating pairwise training
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
            error, losses11, losses12, losses1, losses21, losses22, losses2, modelpreds, preds1, preds2, GRADS = train(model1, model2,
                                                 LABEL, loss_function, optimizer1,
                                                 optimizer2,j,index)
            
            show_multi_curve([losses11, losses12,  losses21, losses22],
                             "losses for the " + str(index + 1) + " th step",
                             ["losses11", "losses12",  "losses21", "losses22"], "EPOCH", "Value")
            show_multi_curve([losses1, losses2],
                             "train and test losses for the " + str(index + 1) + " th step",
                             ["train_losses1", "train_losses2"], "EPOCH", "Value")
            show_multi_curve(GRADS, "train grads for the " + str(index + 1) + " th step",
                             ["grad"+str(i+1) for i in range(len(GRADS))], "EPOCH", "Value")
            
            target = Label[train_time_points + index + period]
            targets.append(target)
            predicts.append(modelpreds[-1])
            errors.append(error)

            label.append(predicts[-1])
            label = label[1:]

            del model1
            del model2
            
            show_multi_curve([preds1, preds2, modelpreds, [target] * EPOCH],
                             "predictions for the " + str(index + 1) + " th step",
                             [str(index + 1) + " th prediction", str(index + 2) + " th prediction","final prediction", "targets"], "EPOCH", "Value")
            
        
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
        final_predicts += predicts
    final_predicts /= repeating
    print(final_predicts)
    np.savetxt('prediction.txt',final_predicts)
    print('test MAE', MAE(final_predicts, targets))
    print('test RMSE', RMSE(final_predicts, targets))

    print('EPOCH ---- ', EPOCH)
    print('LR1 ---- ', LR1)
    print('LR2 ---- ', LR2)
    print('D ---- ', D)
    print('DROPOUT ---- ', DROPOUT)
    print('alpha ---- ', alpha)
    print('w ---- ', w)
    print('activate ---- ', activate)

'''
the pairwise-training ends, if the result predicted here is good enough, there is no need to run the 
following codes; else you can use the models saved by the above codes to initial input of the following
consistent-training process
'''
fit(5,10)

'''
train the model by the consistent-training scheme. 
'''

from tqdm import tqdm_notebook
def show_multi_curve_2(ys, title, legends, xxlabel, yylabel, start_point = 0, if_point = False):
    x = np.array(range(len(ys[0])))
    for i in range(len(ys)):
        if if_point:
            plt.plot(x[start_point:], ys[i][start_point:], label = legends[i], marker = 'o')
        else:
            plt.plot(x[start_point:], ys[i][start_point:], label = legends[i])   
    plt.axis()
    plt.title(title)
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.legend()
    plt.show()
# steps of prediction
step = 5
repeat = 10

# NEWINPUTS[i] ---- training inputs of φi 
NEWINPUTS = []
for i in range(step):
    NEWINPUTS.append(data[step-i-1:train_time_points-i-1,:])

# NEWLABELS --- training outputs of φi 
NEWLABELS = Label[step:train_time_points].reshape(-1,1)
NEWLABELS = torch.tensor(NEWLABELS, dtype=torch.float64).to(device)
def alldropout(data, subspace):
    result = [np.zeros_like(d) for d in data]
    shuffled_indices=np.random.permutation(data[0].shape[1])
    indices =shuffled_indices[:subspace]
    for i in range(len(result)):
        result[i][:, indices] = data[i][:, indices]
    return result

# output multi-step predictions of all learnt models
def predict(models):
    for model in models:
        model.eval()
    predicts = np.array([model(Pred[1:2,:])[0].item() for model in models])
    targets = Label[train_time_points:train_time_points+step]
    print(predicts)
    show_multi_curve_2([predicts, targets],
                     "predictions from the 1 th to the " +
                     str(step) + " th steps",
                     ["predictions", "targets"], "STEP", "Value", 0, True)
    print('test MAE', MAE(predicts, targets))
    print('test RMSE', RMSE(predicts, targets))
    print('test pearsonr', pearsonr(predicts, targets))
    return predicts

# fold([1,2,3,4,5,6],2)   =  ([1,2]+[3,4]+[5,6])/3  = [3,4]
def fold(l, period):
    k = np.array(l[0:period])
    total = 1
    for i in range(period, len(l), period):
        total += 1
        k += np.array(l[i:i+period])
    return k/total

# one round of training on all models
# cycle --- training rounds  D --- sampling rate  alpha --- weight between two parts of losses
def mini_train(cycle, EPOCH, D, alpha, models, optimizers, loss_function):
    for i in range(cycle):
        # train --- φ2-φn
        for modelindex in range(1, step):
            models[modelindex].train()
            losses = []
            losses1 = []
            losses2 = []
            for epoch in tqdm_notebook(range(EPOCH)):
                DR = alldropout(NEWINPUTS, int(INPUT_SIZE * D))
                output1 = models[modelindex](torch.tensor(DR[modelindex], dtype=torch.float64).to(device))
                loss1 = loss_function(output1, NEWLABELS)
                models[0].eval()
                output2 = models[0](torch.tensor(DR[0], dtype=torch.float64).to(device))
                for premodelindex in range(1,modelindex):
                    models[premodelindex].eval()
                    output2 += models[premodelindex](torch.tensor(DR[premodelindex], dtype=torch.float64).to(device))
                output2 /= modelindex
                loss2 = loss_function(output1, output2)
                loss = loss1 + alpha*loss2
                losses.append(loss.item())
                losses1.append(loss1.item())
                losses2.append(loss2.item())
                optimizers[modelindex].zero_grad()
                loss.backward()
                optimizers[modelindex].step()
            models[modelindex].eval()
        # φ1
        modelindex = 0
        models[modelindex].train()
        for epoch in tqdm_notebook(range(EPOCH)):
            DR = alldropout(NEWINPUTS, int(INPUT_SIZE * D))
            output1 = models[modelindex](torch.tensor(DR[modelindex], dtype=torch.float64).to(device))
            loss1 = loss_function(output1, NEWLABELS)
            models[1].eval()
            output2 = models[1](torch.tensor(DR[1], dtype=torch.float64).to(device))
            for aftermodelindex in range(2,step):
                models[aftermodelindex].eval()
                output2 += models[aftermodelindex](torch.tensor(DR[aftermodelindex], dtype=torch.float64).to(device))
            output2 /= (step-1)
            loss2 = loss_function(output1, output2)
            loss = loss1 + alpha*loss2
            losses.append(loss.item())
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            optimizers[modelindex].zero_grad()
            loss.backward()
            optimizers[modelindex].step()
        models[modelindex].eval()
    # Output the results after each round of training
    show_multi_curve_2([fold(x, EPOCH) for x in [losses1, losses2, losses]],
                             "losses for the " + str(i + 1) + " th cycle",
                             ["losses1", "losses1", "losses"],
                             "EPOCH", "Value")
    p = predict(models)
    return p
 
# inital parameters
cycle = 3
EPOCH = 5
D = 0.1
DROPOUT = 0.1
alpha = 10
LR = 1e-5
w = 0

# multi-step predicted results at #step time points with #repeat consistent-training
loss_function = nn.MSELoss()
from scipy.stats import spearmanr
final_predicts = np.array([0.0] * step)
for r in range(repeat):
    MODELS = [NN(units).to(device).double() for i in range(step)]
    for i in range(step):
        saved_parametes = torch.load('./models/'+str(r)+'_'+str(i)+'.pt')
        MODELS[i].load_state_dict(saved_parametes)
        optimizers = [torch.optim.Adam(model.parameters(),lr=LR,weight_decay=w) for model in MODELS]
    final_predicts = final_predicts + mini_train(cycle, EPOCH, D, alpha, MODELS, optimizers, loss_function)
final_predicts /= repeat
targets = Label[train_time_points:train_time_points+step]
print(final_predicts)
show_multi_curve_2([final_predicts, targets],
                     "final predictions from the 1 th to the " +
                     str(step) + " th steps",
                     ["predictions", "targets"], "STEP", "Value", 0, True)
print('test MAE', MAE(final_predicts, targets))
print('test RMSE', RMSE(final_predicts, targets))
print('test pearsonr', pearsonr(final_predicts, targets))
print('test spearmanr',spearmanr(final_predicts,targets))