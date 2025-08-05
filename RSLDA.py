# math——科学计算
import math
# random ——选随机数
import random
# time——计时
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# cvxopt——规划
import cvxopt
import cvxopt.solvers
# mat4py——导出.mat型数据文件（仅支持mat5及以下版本）
import mat4py
# import matplotlib.pyplot as plt
import numpy
import h5py
# numpy——矩阵运算
import numpy as np
import scipy.io
# import tensorflow as tf
# from    tensorflow import keras
# from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import sklearn
from sklearn.decomposition import PCA
import sklearn.svm as svm
from scipy.optimize import linprog
# from liquidSVM import *
from sklearn.metrics import classification_report
import mat4py

import scipy.io as scio

import warnings
warnings.filterwarnings('ignore')

#########################################定义所需函数###############################################
## define kernel functions

# linear kernel functions ## 线性核函数
def linear_kernel(x1, x2):
 return np.dot(x1,numpy.transpose(x2)) # x1*x2的转置（numpy.transpose）(x1和x2位置不可交换)
# polynomial_kernel functions ## 多项式核函数
def polynomial_kernel(x, y, p=3):
 return (1 + np.dot(x, y)) ** p # 对x*y得到的矩阵的所有元素加一后平方
# gaussian_kernel functions ## 高斯核函数
def gaussian_kernel(x, y, sigma=2):
    n1=len(x) # 求x的行数=n1
    n2,n3=numpy.shape(y) # 求y的行数和列数=n2,n3
    if n1==n3: # 当n1为n3时，令n1为1，防止输入矩阵行数为一导致的无法取到正确函数的情况
        n1=1
        sq1=numpy.sum(x*x) # 矩阵对应位置相乘后列向求和（输入的是1*n1的矩阵，可相乘后直接求和）
    else:
     sq1=numpy.transpose([numpy.sum(x*x,axis=1)]) #矩阵对应位置相乘后列向求和再转置（求和:numpy.sum(x（n,m）,axis=),无asxi
     # 是所有元素求和（输出为1*1）,asxi=0是对每一列求和（输出为1*m），asxi=1是对每一行求和（输出为（1*n）） ）
    sq2=[numpy.sum(y*y,axis=1)] # 矩阵对应位置相乘后求和
    # 利用矩阵运算高斯核：sq1(n1,1)*ones(1,n2)+ones(n1,1)*sq2(1,n2)-2*x(n1,n3)*y'(n3,n1)
    # numpy.ones(n,m)生成一个n*m的矩阵，矩阵中所有元素均为1
    z=numpy.dot(sq1,numpy.ones((1,n2)))+numpy.dot(numpy.ones((n1,1)),sq2)-2*np.dot(x,np.transpose(y))
    # 返回 exp(-z/(2*(sigma^2)))
    return np.exp(-z/(2 * (sigma ** 2)))
######################降维函数
def Dimensionality_reduction(train_data,test_data,k):
 model_pca = PCA(n_components=k)# Initialize the PCA module 初始化PCA
 DR_Train_data = model_pca.fit(train_data).transform(train_data)# Build the training set of PCA 构建PCA的训练集合
 DR_Test_data = model_pca.fit(train_data).transform(test_data)# Build the test set of PCA 构建PCA测试集合
 full = {"DR_Train_data": DR_Train_data, "DR_Test_data": DR_Test_data}# Perform PCA training 进行PCA训练
 return full
####fisher训练
def Fisher(train_features, train_targets,test_features,test_targets):
 train_pos = [i for i in range(len(train_targets)) if train_targets[i]==1] # Select the address with label 1 in the training set 选择出训练集中标签为1的地址
 train_neg = [i for i in range(len(train_targets)) if train_targets[i]==-1] # Select the address with label -1 in the training set 选择出训练集中标签为1的地址

 s0 = np.cov(train_features[train_neg,:].T,ddof =0)# Solve the covariance of the training data with label -1 对标签为-1的训练数据解协方差
 m0 = np.mean(train_features[train_neg,:],axis=0)#Average the training data with a label of -1 对标签为-1的训练数据求平均值
 s1 = np.cov(train_features[train_pos,:].T,ddof =0)# Solve the covariance of the training data with label 1 对标签为1的训练数据解协方差
 m1 = np.mean(train_features[train_pos,:],axis=0)#Average the training data with a label of 1 对标签为1的训练数据求平均值
 sw = s0 + s1
 w = np.dot(np.linalg.inv(sw+0.001*np.eye(np.size(sw,0))),(m1-m0).T)#Solve for w by formula 通过公式求解w

 features = np.dot(train_features , w)# 预测训练集的标签
 features_pos = features[train_pos] #预测训练集中标签为1的样本的标签
 features_neg = features[train_neg] #预测训练集中标签为-1的样本的标签
 wm1 = np.mean(features_pos)
 wm2 = np.mean(features_neg)
 baise = -(wm1 + wm2)/2 #计算b
 features_test = np.dot( test_features , w )+ baise #通过y=wx+b对测试集进行预测
 predict_targets = np.sign(features_test) #通过符号函数sign对预测进行划分
 full = {"features": features, "features_test": features_test, "w": w, "baise": baise, "predict_targets":predict_targets} #输出字典
 return full
####求解x,y的二范数的平方
def two_Norm(x, y):
  n1 = len(x)  # 求x的行数=n1
  n2, n3 = numpy.shape(y)  # 求y的行数和列数=n2,n3
  sq1=np.zeros((1,n1))
  sq1[0,:] = numpy.sum(x * x, axis=1) # 矩阵对应位置相乘后列向求和再转置（求和:numpy.sum(x（n,m）,axis=),无asxi

  # 是所有元素求和（输出为1*1）,asxi=0是对每一列求和（输出为1*m），asxi=1是对每一行求和（输出为（1*n）） ）
  sq2=np.zeros((1,n2))
  sq2[0,:] = numpy.sum(y * y, axis=1)  # 矩阵对应位置相乘后求和

  # 利用矩阵运算高斯核：sq1(n1,1)*ones(1,n2)+ones(n1,1)*sq2(1,n2)-2*x(n1,n3)*y'(n3,n1)
  # numpy.ones(n,m)生成一个n*m的矩阵，矩阵中所有元素均为1
  z = numpy.dot(sq1.T, numpy.ones((1, n2))) + numpy.dot(numpy.ones((n1, 1)), sq2) - 2 * np.dot(x, np.transpose(y))
  # 返回 exp(-z/(2*(sigma^2)))
  return z
####Psvm :Efficient Algorithm for Localized Support Vector Machine

def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x,y

data_path="E:/新建文件夹/数据集/imdata/skin_nonskin.mat"

# data_path="F:/数据集/imdata/UCI_breast-cancer.mat"
# 命名数据data（字典）

# data=h5py.File(data_path)
# train_data=np.transpose(data['train_data']) # Take out the array of train_data in the data dictionary (取出data字典中train_data对应的数组)
# train_labels=np.transpose(data['train_labels']) # Take out the array corresponding to train_labels in the data dictionary (取出data字典中train_labels对应的数组)
# test_data=np.transpose(data['test_data'])    # Take out the array corresponding to text_data in the data dictionary (取出data字典中text_data对应的数组)
# test_labels=np.transpose(data['test_labels'])   # Take out the array corresponding to text_labels in the data dictionary (取出data字典中text_labels对应的数组)

data = scio.loadmat(data_path)
train_data=data['train_data']
train_labels=data['train_labels']
test_data=data['test_data']
test_labels=data['test_labels']

print(train_data)
print('12345', train_data.shape)
print(train_labels)
print('12345', train_labels.shape)


n_step = 1    # Number of experiments (试验次数)
err_all = np.zeros((1, n_step))
Time_all = np.zeros((1, n_step))

err1 = np.zeros((n_step, 1))
for step in range(0,n_step): # 定义实验次数循环
    print("训练次数：", step + 1)
    X = np.vstack((train_data, test_data))  # 创立数组X
    print('X', X.shape)
    Y = np.vstack((train_labels, test_labels))  # 创立数组Y
    nall = [i for i in range(len(X))]  # 定义一个长度为样本总数的自然数列的数组
    ntrn = len(train_data)  # 训练集数Dtrain
    random.shuffle(nall)  # 将nall变为随机置换
    idxtrn = nall[0:ntrn]  # 取nall的前Dtrain位
    idxtst = nall[ntrn:]  # 取nall剩下的
    train_data = X[idxtrn, :]
    train_labels = Y[idxtrn, :]  # 取出对应的样本和标签
    test_data = X[idxtst, :]
    test_labels = Y[idxtst, :]  # 取出对应的样本和标签

    train_data = train_data.T
    test_data = test_data.T
    lambda1 = 0.00001
    lambda2 = 0.0001
    dim = 115
    mu = 0.1
    rho = 1.01
    max_iter = 100

    # RSLDA函数
    m = np.size(train_data, 0)
    n = np.size(train_data, 1)
    max_mu = 10^5
    regu = 10^-5


    time_end = time.time()
    err1[step] = np.size(np.where(predict_targets + test_labels == 0)[1])
    err_all[0, step] = np.sum(err1[step]) / np.size(test_data, axis=0)
    Time_all[0, step] = time_end - time_start

    print('err_all', err_all)
    print('Time_all', Time_all)

print('err_all_mean:', np.mean(err_all))
print('err_all_std:', np.std(err_all))
print('Time_all_sum:', np.sum(Time_all))
print('err_all_min', np.min(err_all))