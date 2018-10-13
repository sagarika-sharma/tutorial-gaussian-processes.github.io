import numpy as np
import random
import matplotlib.pyplot as plt
import time
import scipy.stats as sc
import math
from numpy.linalg import inv
from sklearn.preprocessing import normalize
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

f1 = open('linregdata','rU')
matrix = [([line.split(',')[0]]+[float(item) for item in line.split(',')[1:len(line.split(','))-1]]+[int(line.split(',')[-1])]) for line in f1]
matrix = np.array(matrix)
matrix = np.transpose(matrix)
sex = matrix[0]
target = matrix[-1]
matrix = (matrix[1:-1]).astype(float)
matrix = np.concatenate((matrix,[np.exp(matrix[0])],[np.exp(matrix[1])],[matrix[3]**0.5],[matrix[4]**0.5],[matrix[5]**0.5],[matrix[6]**0.5]),axis=0)
stdmatrix = [np.array((matrix[i] - np.mean(matrix[i]))/np.std(matrix[i])) for i in range(len(matrix))]
matrix = np.transpose(np.concatenate(([sex],stdmatrix,[target]),axis = 0))
dataset = [",".join(i) for i in matrix]
f1.close()
random.shuffle(dataset)
train_data_size = int(0.2 * len(dataset))
train_data = dataset[:train_data_size]
test_data = dataset[train_data_size:]


w = np.array([i.split(',')[1:-1] for i in train_data]).astype(float)
wt = np.array([i.split(',')[1:-1] for i in test_data]).astype(float)


target = np.array([[int(i.split(',')[-1]) for i in train_data]]).T
test_target = np.array([[int(i.split(',')[-1]) for i in test_data]]).T

x_train = w
y_train = target
x_test = wt
y_test = test_target

train_samples = x_train.shape[0]
test_samples = x_test.shape[0]

mu,nsigma = 0,1

ls_sigma = np.linspace(1, 200, 20)
ls_l = np.linspace(1, 200, 20)





def kernel(x1,x2,si,lj):
    sigma = si**2
    l = lj**2
    prod = np.dot((x1-x2),(x1-x2).T)
    value = math.exp(-prod/(2.0*l))
    return sigma*value



def kernel_matrix(x1,x2,si,lj):
	x1shape = np.shape(x1)[0]
	x2shape = np.shape(x2)[0]
	k = np.empty([x1shape, x2shape])
	for i in range(0,x1shape):
		for j in range(0,x2shape):
			k[i][j] = kernel(x1[i,:],x2[j,:],si,lj)
	return k


def gaussian_process(x_train,y_train,x_test,train_samples,test_samples,sigma,si,lj):
    m = np.array([[0]]*train_samples)
    st = np.array([[0]]*test_samples)
    d = y_train - m
    k = kernel_matrix(x_train,x_train,si,lj) + (sigma**2)*np.identity(train_samples) 
    inverse_k = inv(k)
    k_t = kernel_matrix(x_train,x_test,si,lj).T
    dot1 = np.dot(inverse_k,d)
    return st + np.dot(k_t,dot1)

regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)
mselr = mean_squared_error(y_test,y_pred)
print("Mean squared error for the linear model is: %.10f"
      % mean_squared_error(y_test,y_pred))

resnp = []
for si in ls_sigma:
	for lj in ls_l:
		g = gaussian_process(x_train,y_train,x_test,train_samples,test_samples,nsigma,si, lj)




		## This is for printing the accuracy of GP regression!
		
		error = g-y_test
		#print(error)
		sq_error = np.square(error)
		#print(sq_error)
		mean_sq_error = sum(sq_error)/sq_error.shape[0]
		res = []
		res.append(si)
		res.append(lj)
		res.append(mean_sq_error)
		res.append(mselr)
		resnp.append(res)
		print("Mean squared error: %.10f"
      		% mean_sq_error)


resnp = np.asarray(resnp, dtype = np.float32)
print(resnp)
np.save("result.npy",resnp)