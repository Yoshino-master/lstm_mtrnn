'''
@author: jinglingzhiyu
'''
import argparse
import numpy as np
import six
import torch
from torch import nn
import torch.nn.functional as F
import random, os, copy
from matplotlib import pylab
from scipy.interpolate import interp1d
import data
import time
from PIL import Image
import pickle

seed_def = 0
batchsize_def = 1
epoch_def = 100
hidden_def = 100
wdecay_def = 0.0
f_hidden_def = 100
s_hidden_def = 30
tau_io_def = 2.0
tau_fh_def = 5.0
tau_sh_def = 70.0
lr_def = 100.0
num_cls = 10
cls_pred_datapath = r'D:\北航\毕设\准备\意图识别调研\文献\MTRNN_CTRNN\MTRNN-master\data\cls_pred_mid_data\formal_cls_pred.pkl'

parser = argparse.ArgumentParser(description='RNN MNIST generation')
parser.add_argument('-S','--seed',default=seed_def, metavar='INT', type=int,
                    help='random seed (default: %d)' % seed_def)
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('-FH', '--fh', default=f_hidden_def, metavar='INT', type=int,
                    help='fast context hidden layer size (default: %d)' % f_hidden_def)
parser.add_argument('-SH', '--sh', default=s_hidden_def, metavar='INT', type=int,
                    help='slow context hidden layer size (default: %d)' % s_hidden_def)
parser.add_argument('-B', '--batchsize', default=batchsize_def, metavar='INT', type=int,
                    help='minibatch size (default: %d)' % batchsize_def)
parser.add_argument('-I', '--epoch', default=epoch_def, metavar='INT', type=int,
                    help='number of training epoch (default: %d)' % epoch_def)
parser.add_argument('-W', '--weightdecay', default=wdecay_def, metavar='FLOAT', type=float,
                    help='weight decay (default: %d)' % wdecay_def)
parser.add_argument('-IO', '--tau_io', default=tau_io_def, metavar='FLOAT', type=float,
                    help='tau_io (default: %f)' % tau_io_def)
parser.add_argument('-TFH', '--tau_fh', default=tau_fh_def, metavar='FLOAT', type=float,
                    help='tau_fh (default: %f)' % tau_fh_def)
parser.add_argument('-TSH', '--tau_sh', default=tau_sh_def, metavar='FLOAT', type=float,
                    help='tau_sh (default: %f)' % tau_sh_def)
parser.add_argument('-L', '--lr', default=lr_def, metavar='FLOAT', type=float,
                    help='lr (default: %f)' % lr_def)
parser.add_argument('--cls_pred_datapath', default=cls_pred_datapath, type=str)
parser.add_argument('-seed', default=999)
parser.add_argument('-cls_nums', default=num_cls, type=int)
args = parser.parse_args()

def select_x(datas, x):
    for i, data in enumerate(datas):
        perm = [int(num) for num in np.linspace(0, len(data)-1, x)]
        datas[i] = [v for j,v in enumerate(data) if j in perm]
    return datas

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, record_all=False):
        self.record_all = record_all
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        if self.record_all is True:
            self.record = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.record_all is True:
            self.record.append(self.val)
    
    def get_avg(self):
        return self.avg

# 计算top K准确率
def accuracy(y_pred, y_actual, topk=(1, 1)):
    """Computes the precision@k for the specified values of k"""
    final_acc = 0
    maxk = max(topk)
    # for prob_threshold in np.arange(0, 1, 0.01):
    PRED_COUNT = y_actual.size(0)
    PRED_CORRECT_COUNT = 0
    prob, pred = y_pred.topk(maxk, 1, True, True)
    # prob = np.where(prob > prob_threshold, prob, 0)
    for j in range(pred.size(0)):
        if int(y_actual[j]) == int(pred[j]):
            PRED_CORRECT_COUNT += 1
    if PRED_COUNT == 0:
        final_acc = 0
    else:
        final_acc = PRED_CORRECT_COUNT / PRED_COUNT
    return final_acc * 100, PRED_COUNT

def seq_accuracy(y_pred, y_actual, topk=(1,1), percentage=True):
    """计算序列准确度"""
    #y_pred : 序列预测结果, 为list,list每个元素的shape=(batch, classes,)
    #y_actual : 序列标注结果,shape=(batch,)
    mean_seq_acc = AverageMeter(record_all=True)
    seq_acc = AverageMeter()
    for i in range(len(y_pred)):
        acc, _ = accuracy(y_pred[i], y_actual)
        seq_acc.update(acc)
        mean_seq_acc.update(seq_acc.get_avg())
    if percentage is False : return mean_seq_acc.record
    y = [0.0] + mean_seq_acc.record
    x = np.linspace(0, 100, num=len(y))
    f = interp1d(x, y, kind='linear')
    percent = np.linspace(0, 100, num=101)
    res = f(percent)
    return res

# Prepare dataset
# print('load MNIST dataset')
# mnist = data.load_mnist_data()
# mnist['data'] = mnist['data'].astype(np.float32)
# mnist['data'] /= 255
# mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
# x_train, x_test = np.split(mnist['data'],   [N])
# x_train = x_train.reshape((-1, 28, 28))
# x_test = x_test.reshape((-1, 28, 28))

# y_train, y_test = np.split(mnist['target'], [N])
# N_test = y_test.size
set_random_seed(args.seed)

with open(args.cls_pred_datapath, 'rb') as f:
    x_train = pickle.load(f)
# x_train = select_x(x_train, 28)
y_train = [0, 1, 1, 1, 2, 2]
# x_train = [x_train[0], x_train[0], x_train[0], x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], x_train[5]]
x_train = [x_train[0], copy.deepcopy(x_train[0]), copy.deepcopy(x_train[0]), x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], copy.deepcopy(x_train[5])]
y_train = [0, 0, 0, 1, 1, 1, 2, 2, 2]

class CTLSTM_cell(nn.Module):
    def __init__(self, tau, n_in, n_hidden):
        super(CTLSTM_cell, self).__init__()
        self.tau, self.n_hidden = tau, n_hidden
        self.x2f = torch.nn.Linear(n_in, n_hidden)
        self.h2f = torch.nn.Linear(n_hidden, n_hidden)
        self.x2i = torch.nn.Linear(n_in, n_hidden)
        self.h2i = torch.nn.Linear(n_hidden, n_hidden)
        self.x2g = torch.nn.Linear(n_in, n_hidden)
        self.h2g = torch.nn.Linear(n_hidden, n_hidden)
        self.x2o = torch.nn.Linear(n_in, n_hidden)
        self.h2o = torch.nn.Linear(n_hidden, n_hidden)
        
    def forward(self, data, c, h):
        f = torch.sigmoid(self.x2f(data) + self.h2f(h))
        i = torch.sigmoid(self.x2i(data) + self.h2i(h))
        g = torch.tanh(self.x2g(data) + self.h2g(h))
        o = torch.sigmoid(self.x2o(data) + self.h2o(h))
        c = (1-1/self.tau) * c + self.tau * (f * c + i * g)
        h = o * torch.tanh(c)
        return h, c

class mtrnn(nn.Module):
#     def __init__(self, n_in=8, fast_hidden_size=100, slow_hidden_size=30, cls_nums=3, tau_fh=3.0, tau_sh=10.0):
    def __init__(self, n_in=8, fast_hidden_size=20, slow_hidden_size=10, cls_nums=3, tau_fh=3.0, tau_sh=10.0):
        super(mtrnn, self).__init__()
        self.n_fh, self.n_sh = fast_hidden_size, slow_hidden_size
        self.tau_fh, self.tau_sh = tau_fh, tau_sh
        self.fc = nn.Linear(slow_hidden_size, cls_nums)
        self.cls_loss = nn.CrossEntropyLoss()
        self.lstm_fh = CTLSTM_cell(1, n_in, fast_hidden_size)
        self.lstm_sh = CTLSTM_cell(1, fast_hidden_size, slow_hidden_size)
#         self.x2hf = nn.Linear(n_in, fast_hidden_size)
        self.hs2hf = nn.Linear(slow_hidden_size, fast_hidden_size)
        self.hf2hs = nn.Linear(fast_hidden_size, slow_hidden_size)
        
    def forward(self, data):
        outs = []
        hf = torch.zeros(data.shape[1], self.n_fh).cuda()
        hs = torch.zeros(data.shape[1], self.n_sh).cuda()
        cf = torch.zeros(data.shape[1], self.n_fh).cuda()
        cs = torch.zeros(data.shape[1], self.n_sh).cuda()
        for i in range(data.shape[0]):
            hf, cf = self.lstm_fh(data[i], hf, cf)
            hs, cs = self.lstm_sh(hf, hs, cs)
            hf = (1-1/self.tau_fh) * hf + torch.tanh(hf + self.hs2hf(hs)) / self.tau_fh
            hs = (1-1/self.tau_sh) * hs + torch.tanh(hs + self.hf2hs(hf)) / self.tau_sh
            outs.append(self.fc(hs))
        return outs
    
    def loss(self, cls_preds, cls_label):
        loss_cls = 0.0
        for cls_pred in cls_preds:
            loss_cls += self.cls_loss(cls_pred, cls_label)
        return loss_cls
    
    def acc(self, cls_pred, cls_label):
        return accuracy(cls_pred, cls_label)
    
model = mtrnn().cuda()
model.load_state_dict(torch.load('intend_pred_model.pth.tar')['state_dict'])

def make_initial_state(batchsize, n_hidden, train=True):
    return np.array(np.zeros((batchsize, n_hidden)),dtype=np.float32)

#optional
def make_initial_state_random(batchsize, n_hidden, train=True):
    return np.array(np.random.uniform(-1,1,(batchsize, n_hidden)),dtype=np.float32)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Initialize slow context initial internal states (for each data)
Csc0_train = make_initial_state(len(x_train), args.sh)
# Csc0_test = make_initial_state(len(x_test), args.sh)

# Learning loop
print('training start ...')
errors = []

for epoch in range(1, args.epoch+1):
    
#     #training epoch
#     model.train()
#     print('epoch:', epoch)
#     now=time.time()
#     cur=now
#     err = torch.zeros(()).cuda()
#     perm = np.random.permutation(len(y_train))
#     
#     for i in perm:
#         x_batch = x_train[i].cuda()
#         x_batch = x_batch.unsqueeze(1)
#         y_batch = torch.Tensor([y_train[i]]).long().cuda()
#         cls_pred = model(x_batch)
#         loss_i = model.loss(cls_pred, y_batch)
#         loss_i /= len(cls_pred)
#         optimizer.zero_grad()
#         loss_i.backward()
#         optimizer.step()
#         acc, count_ = accuracy(cls_pred[-1], y_batch)
#         
#         print(acc, cls_pred[-1], y_batch, loss_i)
#         err += acc
#         
#     print('Done a eopch.')
#     error = (err / len(y_train)).cpu().data
#     print(error)
#     errors.append(error)
#     now=time.time()
#     print('elapsed time:',now-cur)
    
    #计算平均序列准确率
    model.eval()
    with torch.no_grad():
        for i in range(len(x_train)):
            x_batch = x_train[i].cuda()
            x_batch = x_batch.unsqueeze(1)
            y_batch = torch.Tensor([y_train[i]]).long().cuda()
            cls_pred = model(x_batch)
            acc = np.array([seq_accuracy(cls_pred, y_batch)])
            accs = acc if i == 0 else np.concatenate([acc, accs])
    acc = accs.mean(axis=0)
    
#     #evaluate epoch
#     model.eval()
#     accs = AverageMeter()
#     with torch.no_grad():
#         print('evaluation...')
    
    
#     for i in six.moves.range(0, N, args.batchsize):
#         x_batch = torch.Tensor(x_train[perm[i:i + args.batchsize]]).cuda()
#         x_batch = x_batch.permute(1, 0, 2)
#         y_batch = torch.Tensor(y_train[perm[i:i + args.batchsize]]).long().cuda()
#         print(x_batch.shape)
#         print(y_batch.shape)
#         
#         x_batch = x_train[0].cuda()
#         x_batch = x_batch.unsqueeze(1)
#         
#         cls_pred = model(x_batch)
#         loss_i = model.loss(cls_pred, y_batch)
#         err += loss_i.item()*args.batchsize
#         optimizer.zero_grad()
#         loss_i.backward()
#         optimizer.step()
#     
#     print('Done a eopch.')
#     error = err/N/27
#     print('train MSE = %f' %(error))
#     now=time.time()
#     print('elapsed time:',now-cur)
#     
#     with open('train.txt','a+') as f:
#         f.write( '%s %s %s' %(epoch, error.cpu().data.numpy(), now-cur) + '\n' )
#     
#     #evaluate epoch
#     model.eval()
#     accs = AverageMeter()
#     with torch.no_grad():
#         print('evaluation...')
#         now = time.time()
#         cur = now
#         err = torch.zeros(()).cuda()
#         perm = range(N_test)
#         for i in six.moves.range(0, N_test, args.batchsize):
# #             x_batch = torch.Tensor(x_test[perm[i:i + args.batchsize]]).cuda()
#             x_batch = torch.Tensor(x_train[perm[i:i + args.batchsize]]).cuda()
#             x_batch = x_batch.permute(1, 0, 2)
# #             y_batch = torch.Tensor(y_test[perm[i:i + args.batchsize]]).long().cuda()
#             y_batch = torch.Tensor(y_train[perm[i:i + args.batchsize]]).long().cuda()
#             
#             cls_pred = model(x_batch)
#             loss_i = model.loss(cls_pred, y_batch)
#             err += loss_i.item()*args.batchsize
#             acc, _ = model.acc(cls_pred[-1], y_batch)
#             accs.update(acc, args.batchsize)
#             
#         print('Done a evaluation')
#         error = err/N_test/27
#         print('evaluation MSE = %f' %(error))
#         print('accuracy = %f' % (accs.get_avg()))
#         now=time.time()
#         print('elapsed time:',now-cur)
#         with open('evaluation.txt','a+') as f:
#             f.write( '%s %s %s' %(epoch, error.cpu().data.numpy(), now-cur) + '\n')
    
# state = {'state_dict' : model.state_dict()}
# torch.save(state, 'intend_pred_model.pth.tar')
# 
# with open('tmp_err_label.pkl', 'wb') as f:
#     pickle.dump(errors, f)
    
print('finish')








