# -*- coding: utf-8 -*-
#encoding="utf-8"
#!/usr/bin/env python  
""" 
Created on Thu Oct 08 17:36:23 2015 
 
@author: zhangzhiyong 
"""  
  
import numpy as np  
import codecs  
  
data = open('text.txt', 'r').read() #读取txt一整个文件的内容为字符串str类型  
chars = list(set(data))#去除重复的字符  
print chars  
#打印源文件中包含的字符个数、去重后字符个数  
data_size, vocab_size = len(data), len(chars)  
print 'data has %d characters, %d unique.' % (data_size, vocab_size)  
#创建字符的索引表  
char_to_ix = { ch:i for i,ch in enumerate(chars) }  
ix_to_char = { i:ch for i,ch in enumerate(chars) }  
print char_to_ix  
hidden_size = 100 # 隐藏层神经元个数  
seq_length = 20 #  
learning_rate = 1e-1#学习率  
  
#网络模型  
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # 输入层到隐藏层  
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层与隐藏层  
Why = np.random.randn(vocab_size, hidden_size)*0.01 # 隐藏层到输出层，输出层预测的是每个字符的概率  
bh = np.zeros((hidden_size, 1)) #隐藏层偏置项  
by = np.zeros((vocab_size, 1)) #输出层偏置项  
#inputs  t时刻序列，也就是相当于输入  
#targets t+1时刻序列，也就是相当于输出  
#hprev t-1时刻的隐藏层神经元激活值  
def lossFun(inputs, targets, hprev):  
  
  xs, hs, ys, ps = {}, {}, {}, {}  
  hs[-1] = np.copy(hprev)  
  loss = 0  
  #前向传导  
  for t in xrange(len(inputs)):  
    xs[t] = np.zeros((vocab_size,1)) #把输入编码成0、1格式，在input中，为0代表此字符未激活  
    xs[t][inputs[t]] = 1  
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # RNN的隐藏层神经元激活值计算  
    ys[t] = np.dot(Why, hs[t]) + by # RNN的输出  
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # 概率归一化  
    loss += -np.log(ps[t][targets[t],0]) # softmax 损失函数  
  #反向传播  
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)  
  dhnext = np.zeros_like(hs[0])  
  for t in reversed(xrange(len(inputs))):  
    dy = np.copy(ps[t])  
    dy[targets[t]] -= 1 # backprop into y  
    dWhy += np.dot(dy, hs[t].T)  
    dby += dy  
    dh = np.dot(Why.T, dy) + dhnext # backprop into h  
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity  
    dbh += dhraw  
    dWxh += np.dot(dhraw, xs[t].T)  
    dWhh += np.dot(dhraw, hs[t-1].T)  
    dhnext = np.dot(Whh.T, dhraw)  
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:  
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients  
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]  
#预测函数，用于验证，给定seed_ix为t=0时刻的字符索引，生成预测后面的n个字符  
def sample(h, seed_ix, n):  
  
  x = np.zeros((vocab_size, 1))  
  x[seed_ix] = 1  
  ixes = []  
  for t in xrange(n):  
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)#h是递归更新的  
    y = np.dot(Why, h) + by  
    p = np.exp(y) / np.sum(np.exp(y))  
    ix = np.random.choice(range(vocab_size), p=p.ravel())#根据概率大小挑选  
    x = np.zeros((vocab_size, 1))#更新输入向量  
    x[ix] = 1  
    ixes.append(ix)#保存序列索引  
  return ixes  
  
n, p = 0, 0  
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad  
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0  
  
while n<20000:  
  #n表示迭代网络迭代训练次数。当输入是t=0时刻时，它前一时刻的隐藏层神经元的激活值我们设置为0  
  if p+seq_length+1 >= len(data) or n == 0:   
    hprev = np.zeros((hidden_size,1)) #   
    p = 0 # go from start of data  
  #输入与输出  
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]  
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]  
  
  #当迭代了1000次，  
  if n % 1000 == 0:  
    sample_ix = sample(hprev, inputs[0], 200)  
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)  
    print '----\n %s \n----' % (txt, )  
  
  # RNN前向传导与反向传播，获取梯度值  
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)  
  smooth_loss = smooth_loss * 0.999 + loss * 0.001  
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress  
    
  # 采用Adagrad自适应梯度下降法,可参看博文：http://blog.csdn.net/danieljianfeng/article/details/42931721  
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],   
                                [dWxh, dWhh, dWhy, dbh, dby],   
                                [mWxh, mWhh, mWhy, mbh, mby]):  
    mem += dparam * dparam  
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) #自适应梯度下降公式  
  p += seq_length #批量训练  
  n += 1 #记录迭代次数 