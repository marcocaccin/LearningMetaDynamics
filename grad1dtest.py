from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import scipy as sp

krr = KernelRidge(kernel='rbf', gamma=2., alpha=1.0e-2)
X = 10*sp.random.random_sample(200) - 5
y = sp.sin(X) + sp.random.normal(scale=0.1, size=X.shape)
X = sp.atleast_2d(X).T
krr.fit(X,y)
Xtest = sp.atleast_2d(sp.linspace(X.min(),X.max(), 200)).T
yprime_smooth = krr.predict_gradient(Xtest)
yprime_ = krr.predict_gradient(Xtest, smooth=False)
plt.scatter(Xtest, yprime_smooth, c='r', marker = 'o', alpha = .5)
plt.scatter(Xtest, yprime_, c='b', marker = '^', alpha = .5)

plt.show()
