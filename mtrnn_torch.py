'''
Created on 2020年3月10日

@author: jinglingzhiyu
'''
import argparse

import numpy as np
import six
# import chainer
# from chainer import computational_graph as c
# from chainer import cuda
# import chainer.links as L
# import chainer.functions as F
# from chainer import optimizers
import torch
from torch import nn
import torch.nn.functional as F
import random, os
import data
import time
from PIL import Image
import pickle
from astor.source_repr import count

seed_def = 0
batchsize_def = 256
epoch_def = 50
hidden_def = 100
wdecay_def = 0.0
f_hidden_def = 100
s_hidden_def = 30
tau_io_def = 2.0
tau_fh_def = 5.0
tau_sh_def = 7.0
lr_def = 100.0
num_cls = 10

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
parser.add_argument('-cls_nums', default=num_cls, type=int)
args = parser.parse_args()

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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
set_random_seed(999)

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
x_train = x_train.reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

def accuracy(y_pred, y_actual, topk=(1,)):
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

class mtrnn(nn.Module):
#     def __init__(self, io_size, fast_hidden_size, slow_hidden_size, cls_nums, tau_io=2.0, tau_fh=5.0, tau_sh=70.0, tau_cls=10.0):
    def __init__(self, io_size, fast_hidden_size, slow_hidden_size, cls_nums, tau_io=2.0, tau_fh=5.0, tau_sh=70.0, tau_cls=30.0):
        super(mtrnn, self).__init__()
        self.tau_io, self.tau_fh, self.tau_sh, self.tau_cls = tau_io, tau_fh, tau_sh, tau_cls
        self.x_to_fh = nn.Linear(io_size, fast_hidden_size)
        self.fh_to_fh=nn.Linear(fast_hidden_size, fast_hidden_size)
        self.fh_to_sh=nn.Linear(fast_hidden_size, slow_hidden_size)
        self.sh_to_fh=nn.Linear(slow_hidden_size, fast_hidden_size)
        self.sh_to_sh=nn.Linear(slow_hidden_size, slow_hidden_size)
        self.fh_to_y=nn.Linear(fast_hidden_size, io_size)
        self.fh_to_cls=nn.Linear(fast_hidden_size, cls_nums)
        self.sh_to_cls=nn.Linear(slow_hidden_size, cls_nums)
        self.tmp = nn.Linear(slow_hidden_size, cls_nums)
        self.cls_loss = nn.CrossEntropyLoss()
        self.fc = nn.Linear(slow_hidden_size, cls_nums)
    
    def forward2(self, data, u_io, u_fh, u_sh, cls_io):
        fh = torch.sigmoid(u_fh)
        sh = torch.sigmoid(u_sh)
        y_pred = torch.sigmoid(u_io)
#         y_cls = F.softmax(cls_io, dim=1)
        
        u_io2 = (1-1/self.tau_io)*u_io+(self.fh_to_y(fh))/self.tau_io
#         cls_io2 = (1-1/self.tau_cls)*cls_io+(self.fh_to_cls(fh)+self.sh_to_cls(sh))/self.tau_cls
        cls_io2 = self.tmp(sh)
        
        u_fh2 = (1-1/self.tau_fh)*u_fh+(self.x_to_fh(data)+self.fh_to_fh(fh)+self.sh_to_fh(sh))/self.tau_fh
        u_sh2 = (1-1/self.tau_sh)*u_sh+(self.fh_to_sh(fh)+self.sh_to_sh(sh))/self.tau_sh
        return u_io2, u_fh2, u_sh2, y_pred, cls_io2, y_cls
    
    def forward(self, data):
        u_fh = torch.Tensor(make_initial_state(x_batch.shape[1], args.fh)).cuda()
        u_sh = torch.Tensor(make_initial_state(x_batch.shape[1], args.sh)).cuda()
        for i in range(data.shape[0]):
            u_fh = (1-1/self.tau_fh)*u_fh + (nn.functional.tanh(self.x_to_fh(data[i]) + self.fh_to_fh(u_fh) + self.sh_to_fh(u_sh))) / self.tau_fh
            u_sh = (1-1/self.tau_sh)*u_sh + (nn.functional.tanh(self.fh_to_sh(u_fh) + self.sh_to_sh(u_sh))) / self.tau_sh
        y = self.fc(u_sh)
        return y, u_fh, u_sh
    
    def loss(self, y_pred, y_label, cls_pred, cls_label, count_cls=False):
        loss_pred = F.mse_loss(y_pred, y_label)
        if count_cls is True:
            loss_cls = self.cls_loss(cls_pred, cls_label)
            return loss_cls
        else: return loss_pred
    
    def acc(self, cls_pred, cls_label):
        return accuracy(cls_pred, cls_label)
    
model = mtrnn(28, args.fh, args.sh, args.cls_nums).cuda()

def make_initial_state(batchsize, n_hidden, train=True):
    return np.array(np.zeros((batchsize, n_hidden)),dtype=np.float32)

#optional
def make_initial_state_random(batchsize, n_hidden, train=True):
    return np.array(np.random.uniform(-1,1,(batchsize, n_hidden)),dtype=np.float32)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Initialize slow context initial internal states (for each data)
Csc0_train = make_initial_state(len(x_train), args.sh)
Csc0_test = make_initial_state(len(x_test), args.sh)

criterion = nn.CrossEntropyLoss()

# Learning loop
print('training start ...')
for epoch in range(1, args.epoch+1):
    
    #training epoch
    model.train()
    print('epoch:', epoch)
    now=time.time()
    cur=now
    err = torch.zeros(()).cuda()
    perm = np.random.permutation(N)
    
    for i in six.moves.range(0, N, args.batchsize):
        x_batch = torch.Tensor(x_train[perm[i:i + args.batchsize]]).cuda()
        x_batch = x_batch.permute(1, 0, 2)
        y_batch = torch.Tensor(y_train[perm[i:i + args.batchsize]]).long().cuda()
        Csc0 = torch.Tensor(Csc0_train[perm[i:i + args.batchsize]]).cuda()
        u_io = torch.Tensor(make_initial_state(x_batch.shape[0], 28)).cuda()
        u_fh = torch.Tensor(make_initial_state(x_batch.shape[0], args.fh)).cuda()
        cls_io = torch.Tensor(make_initial_state(x_batch.shape[0], args.cls_nums)).cuda()
        
        y, _, _ = model(x_batch)
#         j=0
#         y, u_fh, Csc0 = model(x_batch[:,28*j:28*(j+1)], u_fh, Csc0)
        loss_i = criterion(y, y_batch)
#         for j in range(1,27):
#             y, u_fh, Csc0 = model(x_batch[:,28*j:28*(j+1)], u_fh, Csc0)
#             loss_i += criterion(y, y_batch)
        optimizer.zero_grad()
        loss_i.backward()
        optimizer.step()
    
    print('Done a eopch.')
    error = err/N/27
    print('train MSE = %f' %(error))
    now=time.time()
    print('elapsed time:',now-cur)
    
    with open('train.txt','a+') as f:
        f.write( '%s %s %s' %(epoch, error.cpu().data.numpy(), now-cur) + '\n' )
    
    #evaluate epoch
    model.eval()
    accs = AverageMeter()
    with torch.no_grad():
        print('evaluation...')
        now = time.time()
        cur = now
        err = torch.zeros(()).cuda()
        perm = range(N_test)
        for i in six.moves.range(0, N_test, args.batchsize):
            x_batch = torch.Tensor(x_test[perm[i:i + args.batchsize]]).cuda()
            x_batch = x_batch.permute(1, 0, 2)
            y_batch = torch.Tensor(y_test[perm[i:i + args.batchsize]]).long().cuda()
            Csc0 = torch.Tensor(Csc0_test[perm[i:i + args.batchsize]]).cuda()
            u_io = torch.Tensor(make_initial_state(x_batch.shape[0], 28, train=False)).cuda()
            u_fh = torch.Tensor(make_initial_state(x_batch.shape[0], args.fh, train=False)).cuda()
            cls_io = torch.Tensor(make_initial_state(x_batch.shape[0], args.cls_nums)).cuda()
            
            y, _, _ = model(x_batch)
#             j=0
#             _, u_fh, Csc0 = model(x_batch[:,28*j:28*(j+1)], u_fh, Csc0)
#             for j in range(1,27):
#                 y, u_fh, Csc0 = model(x_batch[:,28*j:28*(j+1)], u_fh, Csc0)
            acc, _ = model.acc(y, y_batch)
            accs.update(acc, args.batchsize)
        
        print('Done a evaluation')
        error = err/N_test/27
        print('evaluation MSE = %f' %(error))
        print('accuracy = %f' % (accs.get_avg()))
        now=time.time()
        print('elapsed time:',now-cur)
        with open('evaluation.txt','a+') as f:
            f.write( '%s %s %s' %(epoch, error.cpu().data.numpy(), now-cur) + '\n')
            
    print('finish')
    
    



