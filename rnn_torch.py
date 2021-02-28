'''
Created on 2020年3月11日

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
epoch_def = 10
hidden_def = 100
wdecay_def = 0.0
f_hidden_def = 100
s_hidden_def = 30
tau_io_def = 2.0
tau_fh_def = 5.0
tau_sh_def = 70.0
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
    def __init__(self, n_in=28, n_hidden=100, cls_nums=10, n_layers=1):
        super(mtrnn, self).__init__()
        self.n_hidden = n_hidden
        self.x2h = nn.Linear(n_in, n_hidden)
        self.h2h = nn.Linear(n_hidden, n_hidden)
        self.fc = nn.Linear(n_hidden, cls_nums)
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, data):
#         out, _ = self.rnn(data)
#         out = self.fc(out[-1])
        h = torch.zeros(data.shape[1], self.n_hidden).cuda()
        for i in range(data.shape[0]):
            h = torch.nn.functional.tanh(self.x2h(data[i]) + self.h2h(h))
        out = self.fc(h)
        return out
    
#     def lstm(self, data, C, h):
#         h = torch.cat((C, h), 0)
    
    def loss(self, cls_pred, cls_label):
        loss_cls = self.cls_loss(cls_pred, cls_label)
        return loss_cls
    
    def acc(self, cls_pred, cls_label):
        return accuracy(cls_pred, cls_label)
    
model = mtrnn().cuda()

def make_initial_state(batchsize, n_hidden, train=True):
    return np.array(np.zeros((batchsize, n_hidden)),dtype=np.float32)

#optional
def make_initial_state_random(batchsize, n_hidden, train=True):
    return np.array(np.random.uniform(-1,1,(batchsize, n_hidden)),dtype=np.float32)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Initialize slow context initial internal states (for each data)
Csc0_train = make_initial_state(len(x_train), args.sh)
Csc0_test = make_initial_state(len(x_test), args.sh)

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
        
        cls_pred = model(x_batch)
        loss_i = model.loss(cls_pred, y_batch)
        err += loss_i.item()*args.batchsize
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
            
            cls_pred = model(x_batch)
#             loss_i = F.mse_loss(y,x_batch[:,28*(j+1):28*(j+2)])
            loss_i = model.loss(cls_pred, y_batch)
            err += loss_i.item()*args.batchsize
            acc, _ = model.acc(cls_pred, y_batch)
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
