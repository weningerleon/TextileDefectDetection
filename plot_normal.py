import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy
from scipy.stats.mstats import normaltest



points = pickle.load(open('D:\\data\\textil\\stoff2_doubleLight\\inner_points2.p', "rb"))

X = np.zeros([len(points), 3])

for idx, pt in enumerate(points):
    X[idx, 0] = np.sqrt(pt.area)
    X[idx, 1] = pt.lower_dist
    X[idx, 2] = pt.upper_dist
    #X[idx, 1] = pt.right_dist
    #X[idx, 2] = pt.left_dist

Y = X[X[:, 1] < 150, :]
Y = Y[Y[:, 1] > 50, :]

Y = Y[Y[:, 2] < 150, :]
Y = Y[Y[:, 2] > 50, :]

np.random.shuffle(Y)
#Left values
plt.figure(0)

mean = np.mean(Y[:,1])
var = np.var(Y[:,1])

test = (Y[:,1]-mean) / np.sqrt(var)
test = Y[:,1]
var2 = test.var()

sum=0
sum1=0
for idx in range(0,14000,160):
    z,pval1 = normaltest(Y[idx:idx+100,0])
    sum+=pval1
    sum1+=1
a = sum/sum1

y = scipy.stats.norm.rvs(size = test.__len__())
z2,pval2 = normaltest(y)

np.var(test)
x=3

x_axis = np.arange(60, 100, 0.001)

plt.xlabel('Distance in px')
plt.ylabel('Probability')
plt.title('Distances to lower neighbor')
plt.text(65, .08, r'$\mu=88.2,\ \sigma=3.98$')


plt.plot(x_axis, norm.pdf(x_axis,mean,np.sqrt(var)), 'r')
#plt.plot(Y[:,1],Y[:,2],'*')
plt.hist(Y[:,1], bins=50, range=(70,110), normed=True)


#Area vaues
plt.figure(1)

mean = np.mean(Y[:,0])
var = np.var(Y[:,0])

z,pval = normaltest(Y[:,0])

plt.xlabel('Square Root of Area')
plt.ylabel('Probability')
plt.title('Weft-Floats: Size in pixel')
plt.text(40, .08, r'$\mu=33.3,\ \sigma=3.46$')


x_axis = np.arange(20, 49.9, 0.001)

plt.plot(x_axis, norm.pdf(x_axis,mean,np.sqrt(var)), 'r')
plt.hist(X[:,0], bins=50, range=(20,49.9), normed=True)
plt.show()




x=3
