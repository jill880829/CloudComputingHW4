# Add Spark Python Files to Python Path
import sys
import os
import numpy as np

input_data = np.genfromtxt("data_banknote_authentication.txt",delimiter=',')
features = input_data[:,:-1]
features = np.insert(features, 0, 1, axis=1)
label = input_data[:,-1]
w = np.zeros(features.shape[1])
sigma = 0

for i in range(20000):
    z = np.dot(features,w)
    ep = np.exp(-z)
    sigmoid = 1/(1+ep)
    loss = sigmoid-label
    gra = np.dot(features.transpose(),loss)
    sigma = sigma+gra**2
    lr = 5/(sigma**0.5)
    w = w-lr*gra

trainErr = 0
for i, x in enumerate(label):
    if sigmoid[i] >= 0.5 and x == 0:
        trainErr += 1
    elif sigmoid[i] < 0.5 and x == 1:
        trainErr += 1

trainErr /= len(label)
print("Training Error = " + str(trainErr))