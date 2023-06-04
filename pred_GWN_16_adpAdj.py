import sys
import os
import shutil
import numpy as np
from scipy.fft import dct, idct
import pandas as pd
from datetime import datetime
import time
import torch
import torch.nn as nn
import Metrics
# import Utils
from GWN_SCPT_14_adpAdj import *
import unseen_nodes

class StandardScaler:
    def __init__(self):
        self.u = None
        self.z = None
    def fit_transform(self, x):
        self.u = x.mean()
        self.z = x.std()
        return (x-self.u)/self.z
    def inverse_transform(self, x):
        return x * self.z + self.u

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * P.TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - P.TIMESTEP_OUT - P.TIMESTEP_IN + 1):
            x = data[i:i+P.TIMESTEP_IN, :]
            y = data[i+P.TIMESTEP_IN:i+P.TIMESTEP_IN+P.TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - P.TIMESTEP_IN,  data.shape[0] - P.TIMESTEP_OUT - P.TIMESTEP_IN + 1):
            x = data[i:i+P.TIMESTEP_IN, :]
            y = data[i+P.TIMESTEP_IN:i+P.TIMESTEP_IN+P.TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def setups():
    # make save folder
    if not os.path.exists(P.PATH):
        os.makedirs(P.PATH)
    # seed
    if P.seed_SS == -1:
        P.seed_SS = P.seed
    torch.manual_seed(P.seed)
    torch.cuda.manual_seed(P.seed)
    np.random.seed(P.seed)
    # epoch
    if P.IS_EPOCH_1:
        P.EPOCH = 1
        P.PRETRN_EPOCH = 1
    print(P.KEYWORD, 'data splits', time.ctime())
    # test split temporal
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    testXS, testYS = getXSYS(data, 'TEST')
    if P.IS_DESEASONED:
        trainXS_ds, trainYS = getXSYS(data_ds, 'TRAIN') # all the Y are de-seasoned
        testXS_ds, testYS = getXSYS(data_ds, 'TEST') # all the Y are de-seasoned
        trainXS = np.concatenate((trainXS, trainXS_ds), axis=1) # the Xs are combined between normal and de-seasoned
        testXS = np.concatenate((testXS, testXS_ds), axis=1) # the Xs are combined between normal and de-seasoned
    # trn val split
    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (1-P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[:P.train_size,:,:,:]
    YS_torch_trn = trainYS[:P.train_size,:,:,:]
    XS_torch_val = trainXS[P.train_size:P.trainval_size,:,:,:]
    YS_torch_val = trainYS[P.train_size:P.trainval_size,:,:,:]
    # spatial split
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=.1, r_tst=.2, seed=P.seed_SS)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=min(1.0,P.R_TRN*8/7), r_tst=1.0, seed=P.seed_SS)
    print('spatialSplit_unseen', spatialSplit_unseen)
    print(spatialSplit_unseen.i_trn)
    print(spatialSplit_unseen.i_val)
    print(spatialSplit_unseen.i_tst)
    print('spatialSplit_allNod', spatialSplit_allNod)
    print(spatialSplit_allNod.i_trn)
    print(spatialSplit_allNod.i_val)
    print(spatialSplit_allNod.i_tst)
    XS_torch_train = torch.Tensor(XS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    YS_torch_train = torch.Tensor(YS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    XS_torch_val_u = torch.Tensor(XS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    XS_torch_val_a = torch.Tensor(XS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    YS_torch_val_a = torch.Tensor(YS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    XS_torch_tst_u = torch.Tensor(testXS[:,:,spatialSplit_unseen.i_tst,:])
    YS_torch_tst_u = torch.Tensor(testYS[:,:,spatialSplit_unseen.i_tst,:])
    XS_torch_tst_a = torch.Tensor(testXS[:,:,spatialSplit_allNod.i_tst,:])
    YS_torch_tst_a = torch.Tensor(testYS[:,:,spatialSplit_allNod.i_tst,:])
    print('train.shape', XS_torch_train.shape, YS_torch_train.shape)
    print('val_u.shape', XS_torch_val_u.shape, YS_torch_val_u.shape)
    print('val_a.shape', XS_torch_val_a.shape, YS_torch_val_a.shape)
    print('tst_u.shape', XS_torch_tst_u.shape, YS_torch_tst_u.shape)
    print('tst_a.shape', XS_torch_tst_a.shape, YS_torch_tst_a.shape)
    # torch dataset
    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train)
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u)
    val_a_data = torch.utils.data.TensorDataset(XS_torch_val_a, YS_torch_val_a)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u)
    tst_a_data = torch.utils.data.TensorDataset(XS_torch_tst_a, YS_torch_tst_a)
    # torch dataloader
    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)
    # adj matrix spatial split
    adj_mx = load_adj(P.ADJPATH, P.ADJTYPE, P.DATANAME)
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn,:][:,spatialSplit_unseen.i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val,:][:,spatialSplit_unseen.i_val]).to(device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val,:][:,spatialSplit_allNod.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[spatialSplit_unseen.i_tst,:][:,spatialSplit_unseen.i_tst]).to(device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst,:][:,spatialSplit_allNod.i_tst]).to(device) for i in adj_mx]
    print('adj_train', len(adj_train), adj_train[0].shape, adj_train[1].shape)
    print('adj_val_u', len(adj_val_u), adj_val_u[0].shape, adj_val_u[1].shape)
    print('adj_val_a', len(adj_val_a), adj_val_a[0].shape, adj_val_a[1].shape)
    print('adj_tst_u', len(adj_tst_u), adj_tst_u[0].shape, adj_tst_u[1].shape)
    print('adj_tst_a', len(adj_tst_a), adj_tst_a[0].shape, adj_tst_a[1].shape)
    # PRETRAIN data loader
    pretrn_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            XS_torch_train[:,-1,:,0].T), P.BATCHSIZE, shuffle=True)
    preval_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(trainYS[:,-1,spatialSplit_unseen.i_val,0]).T.float()),
        P.BATCHSIZE, shuffle=False)
    print('pretrn_iter.dataset.tensors[0].shape', pretrn_iter.dataset.tensors[0].shape)
    print('preval_iter.dataset.tensors[0].shape', preval_iter.dataset.tensors[0].shape)
    # print
    for k, v in vars(P).items():
        print(k,v)
    return pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a

def pre_evaluateModel(model, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x in data_iter:
            l = model.contrast(x[0].to(device))
            l_sum += l.item() * x[0].shape[0]
            n += x[0].shape[0]
        return l_sum / n

def pretrainModel(name, mode, pretrain_iter, preval_iter):
    print('pretrainModel Started ...', time.ctime())
    model = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    for epoch in range(P.PRETRN_EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x in pretrain_iter:
            optimizer.zero_grad()
            loss = model.contrast(x[0].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x[0].shape[0]
            n += x[0].shape[0]
        train_loss = loss_sum / n
        val_loss = pre_evaluateModel(model, preval_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '.pt')
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    print('pretrainModel Ended ...', time.ctime())

def getModel(name, device):
    model = gwnet(device, num_nodes=P.N_NODE, in_dim=P.CHANNEL, adp_adj=P.adp_adj, sga=P.is_SGA).to(device)
    return model

def evaluateModel(model, criterion, data_iter, adj, embed):
    model.eval()
    torch.cuda.empty_cache()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x.to(device), adj, embed)
            l = criterion(y_pred, y.to(device))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
    return l_sum / n

def predictModel(model, data_iter, adj, embed):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x.to(device), adj, embed)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode,
        train_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod):
    print('trainModel Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', P.TIMESTEP_IN, P.TIMESTEP_OUT)
    model = getModel(name, device)
    min_val_u_loss = np.inf
    min_val_a_loss = np.inf
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    print('Model Training Started ...', s_time)
    if P.IS_PRETRN:
        encoder = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
        encoder.eval()
        with torch.no_grad():
            encoder.load_state_dict(torch.load(P.PATH+ '/' + 'encoder' + '.pt'))
            train_embed = encoder(train_iter.dataset.tensors[0][:,-1,:,0].T.to(device)).T.detach()
            if P.IS_DESEASONED:
                val_u_embed = encoder(torch.Tensor(data_ds[:P.train_size,spatialSplit_unseen.i_val]).to(device).float().T).T.detach()
                val_a_embed = encoder(torch.Tensor(data_ds[:P.train_size,spatialSplit_allNod.i_val]).to(device).float().T).T.detach()
            else:
                val_u_embed = encoder(torch.Tensor(data[:P.train_size,spatialSplit_unseen.i_val]).to(device).float().T).T.detach()
                val_a_embed = encoder(torch.Tensor(data[:P.train_size,spatialSplit_allNod.i_val]).to(device).float().T).T.detach()
    else:
        train_embed = torch.zeros(32, train_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_u_embed = torch.zeros(32, val_u_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_a_embed = torch.zeros(32, val_a_iter.dataset.tensors[0].shape[2]).to(device).detach()
    print('train_embed', train_embed.shape, train_embed.mean(), train_embed.std())
    print('val_u_embed', val_u_embed.shape, val_u_embed.mean(), val_u_embed.std())
    print('val_a_embed', val_a_embed.shape, val_a_embed.mean(), val_a_embed.std())
    for epoch in range(P.EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x.to(device), adj_train, train_embed)
            loss = criterion(y_pred, y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
            # print('n', n)
        train_loss = loss_sum / n
        val_u_loss = evaluateModel(model, criterion, val_u_iter, adj_val_u, val_u_embed)
        val_a_loss = evaluateModel(model, criterion, val_a_iter, adj_val_a, val_a_embed)
        if val_u_loss < min_val_u_loss:
            min_val_u_loss = val_u_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '_u.pt')
        if val_a_loss < min_val_a_loss:
            min_val_a_loss = val_a_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '_a.pt')
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch,
            "time used:",epoch_time," seconds ",
            "train loss:", train_loss,
            "validation unseen nodes loss:", val_u_loss,
            "validation all nodes loss:", val_a_loss)
        with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f, %s, %.10f\n" % \
                ("epoch", epoch,
                 "time used:",epoch_time," seconds ",
                 "train loss:", train_loss,
                 "validation unseen nodes loss:", val_u_loss,
                 "validation all nodes loss:", val_a_loss))
    e_time = datetime.now()
    print('MODEL TRAINING DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    torch_score = evaluateModel(model, criterion, train_iter, adj_train, train_embed)
    with open(P.PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, %s, %.10e, %.10f\n" % (name, mode, 'MAE on train', torch_score, torch_score))
    print('*' * 40)
    print("%s, %s, %s, %.10e, %.10f" % (name, mode, 'MAE on train', torch_score, torch_score))
    print('min_val_u_loss', min_val_u_loss)
    print('min_val_a_loss', min_val_a_loss)
    print('trainModel Ended ...', time.ctime())

def testModel(name, mode, test_iter, adj_tst, spatialsplit):
    criterion = nn.L1Loss()
    print('Model Testing', mode, 'Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', P.TIMESTEP_IN, P.TIMESTEP_OUT)
    if P.IS_PRETRN:
        encoder = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
        encoder.load_state_dict(torch.load(P.PATH+ '/' + 'encoder' + '.pt'))
        encoder.eval()
    model = getModel(name, device)
    model.load_state_dict(torch.load(P.PATH+ '/' + name +mode[-2:]+ '.pt'))
    s_time = datetime.now()
    
    print('Model Infer Start ...', s_time)
    tst_embed = torch.zeros(32, test_iter.dataset.tensors[0].shape[2]).to(device).detach()
    n_node_remaining = P.N_NODE
    if P.IS_PRETRN:
        data_= data_ds if P.IS_DESEASONED else data
        with torch.no_grad():
            i=0
            while n_node_remaining > 2000:
                tst_embed[:, i*2000:(i+1)*2000] = encoder(torch.Tensor(data_[:P.trainval_size,spatialsplit.i_tst[i*2000:(i+1)*2000]]).to(device).float().T).T.detach()
                i+=1
                n_node_remaining -= 2000
            tst_embed[:, -n_node_remaining:] = encoder(torch.Tensor(data_[:P.trainval_size,spatialsplit.i_tst[-n_node_remaining:]]).to(device).float().T).T.detach()
    torch_score = evaluateModel(model, criterion, test_iter, adj_tst, tst_embed)
    e_time = datetime.now()
    print('Model Infer End ...', e_time)
    
    print('MODEL INFER DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    YS_pred = predictModel(model, test_iter, adj_tst, tst_embed)
    YS = test_iter.dataset.tensors[1].cpu().numpy()
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    original_shape = np.squeeze(YS).shape
    YS = scaler.inverse_transform(np.squeeze(YS).reshape(-1, YS.shape[2])).reshape(original_shape)
    YS_pred  = scaler.inverse_transform(np.squeeze(YS_pred).reshape(-1, YS_pred.shape[2])).reshape(original_shape)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(P.PATH + '/' + P.MODELNAME + '_' + mode + '_' + name +'_prediction.npy', YS_pred)
    np.save(P.PATH + '/' + P.MODELNAME + '_' + mode + '_' + name +'_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(P.PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(P.TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())

################# Parameter Setting #######################
P = type('Parameters', (object,), {})()
P.TIMESTEP_IN = 12
P.TIMESTEP_OUT = 12
P.CHANNEL = 1
P.BATCHSIZE = 64 # 64
P.LEARN = 0.001
P.PRETRN_EPOCH = 100
P.EPOCH = 100 # 100
P.TRAINRATIO = 0.8 # TRAIN + VAL
P.TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
P.ADJTYPE = 'doubletransition'
P.MODELNAME = 'GraphWaveNet'

data = None
data_ds = None
scaler = None
###########################################################
def get_argv():
    ''' # ARGV
    0: .py file
    1: IS_PRETRN
    2: R_TRN
    3: IS_EPOCH_1
    4: seed
    5: TEMPERATURE
    6: dataset
    7: seed_ss # spatial split
    8: IS_DESEASONED
    9: weight_decay
    10: adp_adj
    11: is_SGA
    '''
    print('sys.argv', sys.argv)
    P.IS_PRETRN = bool(int(sys.argv[1])) if len(sys.argv) >= 2 else True
    P.R_TRN = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.7
    P.IS_EPOCH_1 = bool(int(sys.argv[3])) if len(sys.argv) >= 4 else False
    P.seed = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    P.TEMPERATURE = float(sys.argv[5]) if len(sys.argv) >= 6 else 1.0
    P.DATANAME = sys.argv[6] if len(sys.argv) >= 7 else 'METRLA'
    P.seed_SS = int(sys.argv[7]) if len(sys.argv) >= 8 else -1
    P.IS_DESEASONED = bool(int(sys.argv[8])) if len(sys.argv) >= 9 else True
    P.weight_decay = float(sys.argv[9]) if len(sys.argv) >= 10 else 0.0
    P.adp_adj = bool(int(sys.argv[10])) if len(sys.argv) >= 11 else False
    P.is_SGA = bool(int(sys.argv[11])) if len(sys.argv) >= 12 else True

device = torch.device('cuda:0') 
###########################################################
def main():
    script_start_time = datetime.now()

    get_argv()

    # DATASET
    P.KEYWORD = 'pred_' + P.DATANAME + '_' + P.MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M") + '_' + str(os.getpid())
    P.PATH = '../save/' + P.KEYWORD
    global data
    global data_ds
    global scaler
    n_dct_coeff = None
    if P.DATANAME == 'METRLA':
        print('P.DATANAME == METRLA')
        P.FLOWPATH = '../METRLA/metr-la.h5'
        P.n_dct_coeff = 3918
        P.ADJPATH = '../METRLA/adj_mx.pkl'
        P.N_NODE = 207
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSBAY':
        print('P.DATANAME == PEMSBAY')
        P.FLOWPATH = '../PEMSBAY/pems-bay.h5'
        P.n_dct_coeff = 4107
        P.ADJPATH = '../PEMSBAY/adj_mx_bay.pkl'
        P.N_NODE = 325
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSD7M':
        print('P.DATANAME == PEMSD7M')
        P.FLOWPATH = '../PEMSD7M/V_228.csv'
        P.n_dct_coeff = 860
        P.ADJPATH = '../PEMSD7M/W_228.csv'
        P.N_NODE = 228
        data = pd.read_csv(P.FLOWPATH,index_col=[0]).values
    elif P.DATANAME == 'PEMS11160':
        print('P.DATANAME == PEMS11160')
        P.BATCHSIZE = 16
        P.EPOCH = 20
        P.FLOWPATH = '../PEMS11160/pems12kSPEED2m.npy'
        P.n_dct_coeff = 2179
        P.ADJPATH = '../PEMS11160/adj_mat.pkl'
        P.N_NODE = 11160
        with open(P.FLOWPATH, 'rb') as f:
            data = np.load(f)
    else:
        print('NO DATA LOADED')

    # de-season
    if P.IS_DESEASONED:
        P.CHANNEL = 2
        data_ = dct(data, axis=0)
        data_[n_dct_coeff:, :] = 0
        data_ds = data - idct(data_, axis=0) # the seasonal data

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # de-season scaler
    if P.IS_DESEASONED:
        scaler = StandardScaler()
        data_ds = scaler.fit_transform(data_ds)

    print('data.shape', data.shape)

    pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
    train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
    adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a = setups()

    if P.IS_PRETRN:
        print(P.KEYWORD, 'pretraining started', time.ctime())
        pretrainModel('encoder', 'pretrain', pretrn_iter, preval_iter)
    else:
        print(P.KEYWORD, 'No pre-training')

    print(P.KEYWORD, 'training started', time.ctime())
    trainModel(P.MODELNAME, 'train',
        train_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod)
    
    print(P.KEYWORD, 'testing started', time.ctime())
    testModel(P.MODELNAME, 'test_u', tst_u_iter, adj_tst_u, spatialSplit_unseen)
    testModel(P.MODELNAME, 'test_a', tst_a_iter, adj_tst_a, spatialSplit_allNod)
    print('SCRIPT DURATION', datetime.now()-script_start_time)

if __name__ == '__main__':
    main()

