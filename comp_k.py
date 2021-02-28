'''
Created on 2020年3月28日

@author: jinglingzhiyu
'''
import data
import numpy as np
import pickle

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
    dist = 1. - similiarity
    return dist

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

# select_num = 1000
# x_train, y_train = x_train[ : select_num], y_train[ : select_num]

ks = np.zeros((x_train.shape[0], x_train.shape[1], 10))
class_pos = []
for i in range(10):
    class_pos.append(np.where(y_train == i))
length = 25
for i in range(28):
    start = max(0, i - length)
    features = x_train[:, start:i+1].reshape((x_train.shape[0], -1))
    features_norm = np.linalg.norm(features, axis=1)
    centres, dists = [], []
    for j in range(10):
        centres.append(np.mean(features[class_pos[j]], axis=0))
        similiarity = np.dot(features, centres[-1].T)/(features_norm * np.linalg.norm(centres[-1]))
        ks[:, i, j] = 1. / (np.exp(similiarity - 1.) + 1)

with open('ks_train.pkl', 'wb') as f:
    pickle.dump(ks, f)

print('finish')