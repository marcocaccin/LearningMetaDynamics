from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import scipy as sp
from gpfit import test_points

lengthscale = 1.
gamma = 1 / (2 * lengthscale**2)

krr = KernelRidge(kernel='rbf', gamma=gamma, alpha=1.0e-1, gammaL=0.1*gamma, gammaU=10*gamma, max_lhood=True)
X = 4*sp.random.random_sample(210) - 1
X.sort()
y = sp.sin(X**2) + sp.random.normal(scale=0.1, size=len(X))
X = sp.atleast_2d(X).T
krr.fit_w_noise(X, y)
Xtest = sp.atleast_2d(sp.linspace(X.min(), X.max(), 200)).T

y_pred, MSE = krr.predict(Xtest, MSE=True)
y_smooth = krr.predict(X).ravel()

# yprime_smooth = krr.predict_gradient(Xtest, smooth=True).ravel()

yprime_ = krr.predict_gradient(Xtest, smooth=False).ravel()

print("noise = %.3e, lengthscale = %.3e" % (krr.noise.mean(), 1/(2 * krr.gamma)**0.5))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(24, 16))

ax0.plot(Xtest, sp.sin(Xtest**2), 'g--')
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
ax1.plot(Xtest, 2*Xtest * sp.cos(Xtest**2), 'g--')
# ax1.plot(Xtest, yprime_smooth, 'ro')
ax1.plot(Xtest, yprime_, 'b^')

plt.show()
