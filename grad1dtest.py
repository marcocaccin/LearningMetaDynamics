from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import scipy as sp
from gpfit import test_points

def fun(x):
    return sp.sin(x)**2 + sp.exp(- sp.cos(x))


def dfun(x):
    return 2*sp.sin(x)*sp.cos(x) + sp.exp(-sp.cos(x)) * sp.sin(x)


lengthscale = 1.
gamma = 1 / (2 * lengthscale**2)

krr = KernelRidge(kernel='rbf_periodic', gamma=gamma, alpha=1.0e-1, gammaL=0.1*gamma, gammaU=10*gamma, max_lhood=False)
X = 12*sp.random.random_sample(210) - 1
X.sort()
y = fun(X) + sp.random.normal(scale=0.1, size=len(X))
X = sp.atleast_2d(X).T
krr.fit_w_noise(X, y)
Xtest = sp.atleast_2d(sp.linspace(X.min(), X.max(), 200)).T

y_pred, MSE = krr.predict(Xtest, MSE=True)
y_smooth = krr.predict(X).ravel()

yprime_ = krr.predict_gradient(Xtest).ravel()

print("noise = %.3e, lengthscale = %.3e" % (krr.noise.mean(), 1/(2 * krr.gamma)**0.5))

plt.clf()
plt.close()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(24, 16))

ax0.plot(Xtest, fun(Xtest), 'g--')
ax0.scatter(X, y, c='r', marker = '+', alpha = .5)
ax0.scatter(Xtest, y_pred, c='b', marker = 'o', alpha = .5)
# ax0.fill(sp.concatenate([X, X[::-1]]),
#          sp.concatenate([y_smooth - 1.9600 * krr.noise,
#                         (y_smooth + 1.9600 * krr.noise)[::-1]]),
#          alpha=.2, fc='b', ec='None', label='95% confidence interval')
ax0.fill(sp.concatenate([Xtest, Xtest[::-1]]),
         sp.concatenate([y_pred - 1.9600 * MSE,
                        (y_pred + 1.9600 * MSE)[::-1]]),
         alpha=.2, fc='b', ec='None', label='95% confidence interval')
ax1.plot(Xtest, dfun(Xtest), 'g--')
ax1.plot(Xtest, yprime_, 'b^')

plt.show()
