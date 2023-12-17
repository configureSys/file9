import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np1
import numpy.linalg as np
from scipy.stats.stats import pearsonr
import pdb
def kernel(point, xmat, k):
	m,n=np1.shape(xmat)  #size of matrix m
	weights=np1.mat(np1.eye(m)) #np.eye returns mat with 1 in the diagonal
	for j in range(m):
		diff=point-xmat[j]
		weights[j,j]=np1.exp(diff*diff.T/(-2.0*k**2))
	return weights

def localWeight(point,xmat,ymat,k):
	wei=kernel(point,xmat,k)
	W=(xmat.T*(wei*xmat)).I*(xmat.T*(wei*ymat.T))
	return W

def localWeightRegression(xmat,ymat,k):
	row,col=np1.shape(xmat) #return 244 rows and 2 columnsmns
	ypred=np1.zeros(row)
	for i in range(row):
		ypred[i]=xmat[i]*localWeight(xmat[i],xmat,ymat,k)                            
	return ypred

data=pd.read_csv('ml110.csv')
bill=np1.array(data.total_bill)
tip=np1.array(data.tip)

mbill=np1.mat(bill)
mtip=np1.mat(tip)

mbillMatCol=np1.shape(mbill)[1] # 1 for vertical i.e columns
onesArray=np1.mat(np1.ones(mbillMatCol))
xmat=np1.hstack((onesArray.T,mbill.T))
print(xmat)

ypred=localWeightRegression(xmat,mtip,2)
SortIndex=xmat[ :,1].argsort(0)
xsort=xmat[SortIndex][:,0]

fig= plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(bill,tip,color='blue')
ax.plot(xsort[:,1],ypred[SortIndex],color='red',linewidth=1)
plt.xlabel('Total bill')
plt.ylabel('tip')
plt.show();
pdb.set_trace()

#*****************short code ******************
#*****************short code ******************
#*****************short code ******************


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)
y = np.log(np.abs((x ** 2) - 1) + 0.5)
x = x + np.random.normal(scale=0.05, size=1000) 
plt.scatter(x, y, alpha=0.3)
def local_regression(x0, x, y, tau): 
    x0 = np.r_[1, x0]
    x = np.c_[np.ones(len(x)), x]
    xw =x.T * radial_kernel(x0, x, tau) 
    beta = np.linalg.pinv(xw @ x) @ xw @ y 
    return x0 @ beta


def radial_kernel(x0, x, tau):
    return np.exp(np.sum((x - x0) ** 2, axis=1) / (-2 * tau ** 2))


def plot_lr(tau):
    domain = np.linspace(-5, 5, num=500)
    pred = [local_regression(x0, x, y, tau) for x0 in domain] 
    plt.scatter(x, y, alpha=0.3)
    plt.plot(domain, pred, color="red") 
    return plt


plot_lr(1).show()
