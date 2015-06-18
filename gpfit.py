#! /usr/bin/env python
from __future__ import division, print_function
import scipy as sp
from numpy import atleast_2d
from sklearn.gaussian_process import GaussianProcess
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
import random
from matplotlib import pyplot as plt
import time
import scipy.spatial as spsp
import theano
import theano.tensor as the
import theanets


def remove_duplicate_vectors(X):
    X_purged = [X[0]]
    for v in X[1:]:
        if True: #v not in sp.asarray(X_purged):
            X_purged.append(v)
    return sp.asarray(X_purged)


def test_points(X, yprime, reach, n_test_pts):
    yp = sp.absolute(yprime)
    reaches = reach * sp.exp(- yp / yp.mean())
    # reaches[...] = 1.
    X_test = []
    for point, r in zip(X, reaches):
        # points within rectangle. meshgrid will work in > 2D for numpy > 1.7
        xs = [sp.linspace(point[i] - r[i], point[i] + r[i], n_test_pts)
              for i in range(n_features)]
        xs = sp.meshgrid(*xs)
        X_test.append(sp.vstack([x.ravel() for x in xs]).reshape(n_features,-1).T)
    X_test = sp.asarray(X_test).reshape(-1,n_features)
    X_test = remove_duplicate_vectors(X_test)    
    # X_test = sp.sort(X_test, axis = 0)
    return X_test


def is_point_within_rectangle(point, Xmin, Xmax):
    return sp.alltrue(point >= Xmin) and sp.alltrue(point <= Xmax)
    
    
def remove_pts_outside_boundaries(X, Xmin, Xmax):
    yes = [is_point_within_rectangle(point, Xmin, Xmax) for point in X]
    return X[sp.where(yes)[0]]

def twoD_gauss(x, y, x0, sigma, angle):
    R = sp.array([[the.cos(angle), the.sin(angle)],[-the.sin(angle), the.cos(angle)]])
    Rx = R[0,0] * x + R[0,1] * y
    Ry = R[1,0] * x + R[1,1] * y
    return the.exp(- 0.5 * ((Rx - x0[0]) / sigma[0])**2 - 0.5 * ((Ry - x0[1]) / sigma[1])**2)


def target_fun(X):
    X = sp.asarray(X)
    if len(sp.shape(X)) == 0:
        thX = the.scalar('thX')
    elif len(X.shape) == 1:
        thX = the.dvector('thX')
        y = - 0.8 * twoD_gauss(thX[0], thX[1], sp.array([-1,-1]), sp.array([3,2]), sp.pi/4
          ) + 1.2 * twoD_gauss(thX[0], thX[1], sp.array([3,0]), sp.array([1,1]), 0
          ) + 2.0 * twoD_gauss(thX[0], thX[1], sp.array([-2,2]), sp.array([1,4]), 0)
        fun = theano.function([thX], y, allow_input_downcast=True)
        return fun(X)
    elif len(X.shape) == 2:
        thX = the.dmatrix('thX')
        y = - 0.8 * twoD_gauss(thX[:,0], thX[:,1], sp.array([-1,-1]), sp.array([3,2]), sp.pi/4
          ) + 1.2 * twoD_gauss(thX[:,0], thX[:,1], sp.array([3,0]), sp.array([1,1]), 0
          ) + 2.0 * twoD_gauss(thX[:,0], thX[:,1], sp.array([-2,2]), sp.array([1,4]), 0)
        fun = theano.function([thX], y, allow_input_downcast=True)
        return fun(X)
    else:
        print("Bad Input")
    

def dtarget_fun(X):
    results = []
    X = sp.asarray(X)
    thX = the.dvector('thX')
    y = - 0.8 * twoD_gauss(thX[0], thX[1], sp.array([-1,-1]), sp.array([3,2]), sp.pi/4
      ) + 1.2 * twoD_gauss(thX[0], thX[1], sp.array([3,0]), sp.array([1,1]), 0
      ) + 2.0 * twoD_gauss(thX[0], thX[1], sp.array([-2,2]), sp.array([1,4]), 0)
    grady = the.grad(y, thX)
    dfun = theano.function([thX], grady, allow_input_downcast=True)
    
    if len(sp.shape(X)) == 0:
        thX = the.scalar('thX')
        return dfun(X)
    elif len(X.shape) == 1:
        return dfun(X)
    elif len(X.shape) == 2:
        results = []
        for x in X:
            results.append(dfun(x))
        return sp.array(results)
    else:
        print("Bad Input")
        
        
def h_diag_target_fun(X):
    results = []
    X = sp.asarray(X)
    thX = the.dvector('thX')
    y = - 0.8 * twoD_gauss(thX[0], thX[1], sp.array([-1,-1]), sp.array([3,2]), sp.pi/4
      ) + 1.2 * twoD_gauss(thX[0], thX[1], sp.array([3,0]), sp.array([1,1]), 0
      ) + 2.0 * twoD_gauss(thX[0], thX[1], sp.array([-2,2]), sp.array([1,4]), 0)
    grady = the.grad(y, thX)
    H, updates = theano.scan(lambda i, grady, thX : the.grad(grady[i], thX), 
                             sequences=the.arange(grady.shape[0]), 
                             non_sequences=[grady, thX])    
    hfun =  theano.function([thX], H, updates=updates, allow_input_downcast=True)
    if len(sp.shape(X)) == 0:
        thX = the.scalar('thX')
        return sp.diagonal(hfun(X))
    elif len(X.shape) == 1:
        return sp.diagonal(hfun(X))
    elif len(X.shape) == 2:
        results = []
        for x in X:
            results.append(sp.diagonal(hfun(x)))
        return sp.array(results)
    else:
        print("Bad Input")
     
    

def select_next_x(sort_order, x_list, old_x):
    x_list = x_list[sort_order]
    for x in x_list:
        if x not in old_x:
            return x
    

def dynamics_step(pos, vel, deltat, posminmax):
    """
    Basic: x_1 = x_0 + v_0 * dt + 0.5 * a * (dt)**2
    Bounce back from extremes
    """
    dpos = vel * deltat - 0.5 * deltat**2 * dtarget_fun(pos)
    dvel = - deltat * dtarget_fun(pos)
    if (pos + dpos <= posminmax[1]) and (pos + dpos >= posminmax[0]):
        return pos + dpos, vel + dvel
    else:
        return pos - dpos, - (vel + dvel)


def highest_MSE_x(gp, X_test, X_old, Xminmax):
    """
    Given a trained Gaussian Process gp, predict the values y_pred on a given
    grid x_test. Return the x value corresponding to the highest predicted 
    variance which is not present in x_old and is within the given extrema 
    xminmax.
    """
    
    y_pred, MSE = gp.predict(atleast_2d(X_test), eval_MSE=True)
    sort_order = sp.argsort(MSE)[::-1]
    X_t = X_test[sort_order]
    X_t = list(X_t)
    while len(X_t) > 1:
        Xnew = X_t[0]
        if is_point_within_rectangle(Xnew, Xminmax[0], Xminmax[1]):
            _, max_pred_err = gp.predict(atleast_2d(Xnew), eval_MSE=True)
            return Xnew, y_pred, MSE, max_pred_err
        else:
            X_t.pop(0)

def pick_Xnew(gp, X_test, X_old, Xminmax, X_old_scores):
    X_old_scores = sp.absolute(X_old_scores).max(axis=1)
    yprimes = sp.absolute(dtarget_fun(X_old)).max(axis=1)
    y_pred, MSE = gp.predict(atleast_2d(X_test), eval_MSE=True)
    distances = spsp.distance.cdist(X_test, X)
    nearest_2_X_old = map(sp.argmin, distances)

    hess_scores = sp.asarray([X_old_scores[i] for i in nearest_2_X_old]) / X_old_scores.max()
    grad_scores = sp.exp(- .5 * sp.asarray( [yprimes[i] for i in nearest_2_X_old]) / yprimes.mean())
    grad_scores /= yprimes.max()
    scores = MSE * (1 * grad_scores + 0.8 *hess_scores)

    sort_order = sp.argsort(scores)[::-1]
    X_t, MSE_sorted, scores_sorted = X_test[sort_order], MSE[sort_order], scores[sort_order]
    X_t = list(X_t)
    while len(X_t) > 1:
        Xnew = X_t[0]
        if is_point_within_rectangle(Xnew, Xminmax[0], Xminmax[1]):
            print("scores details: grad = %.3e, hess = %.3e" % (grad_scores.max(), hess_scores.max()))    
            return Xnew, y_pred, scores, scores_sorted[0]
        else:
            X_t.pop(0)
            scores_sorted.pop(0)
                
    
def draw_2Dplot(ax, X, y, X_test, y_test, scores, Xnew):
    ax0 = ax[0,0]
    #bubbles = sp.exp(-MSE)
    #bubbles = sp.array((bubbles - bubbles.min()) / bubbles.std() * 5 + 1, dtype='int')
    ax0.clear()
    ax0.scatter(X_test[:,0], X_test[:,1], s = 100, c = scores, cmap = 'YlGnBu', 
                alpha = 0.7, edgecolors='none')
    ax0.scatter(X[:,0], X[:,1], s = 80, marker = '+', c = '#E58B24',
                cmap = 'Spectral', alpha = 1)
    ax0.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)
    # ax0.scatter(X_test[:,0], X_test[:,1], s = 20, c = y_test, cmap = 'YlGnBu', alpha = 0.5, edgecolors='none')
    ax0.set_title('Data points, predict grid, MSE')
    ax0.set_xlabel('$CV 1$')
    ax0.set_ylabel('$CV 2$')
    ax0.set_xlim(Xmin[0], Xmax[0])
    ax0.set_ylim(Xmin[1], Xmax[1])


def Xplotgrid(Xminmax, n_features, n_pts):
    Xmin, Xmax = Xminmax
    X_grid = [sp.linspace(Xmin[i], Xmax[i], n_pts) for i in range(n_features)]
    X_grid = sp.meshgrid(*X_grid)
    return sp.vstack([x.ravel() for x in X_grid]).reshape(n_features,-1).T


def draw_target_fun(ax, X):
    ax1 = ax[1,1]
    y_target = target_fun(X)
    ax1.scatter(X[:,0], X[:,1], s = 200, marker='s', c = y_target, 
                cmap = 'Spectral', alpha = 1, edgecolors='none')
    ax1.set_title('Target Function')
    ax1.set_xlabel('$CV 1$')
    ax1.set_ylabel('$CV 2$')
    ax1.set_xlim(X_grid[:,0].min(), X_grid[:,0].max())
    ax1.set_ylim(X_grid[:,1].min(), X_grid[:,1].max())
    return y_target.min(), y_target.max()
    
    
def draw_2Dreconstruction(ax, gp, Xnew, X_grid, yminmax):
    ymin, ymax = yminmax
    ax2 = ax[1,0]
    ax3 = ax[0,1]
    ax6 = ax[2,1]

    y_grid, MSE_grid = gp.predict(X_grid, eval_MSE=True)
    y_grid[0], y_grid[-1] = ymin, ymax
    ax2.clear()
    ax2.scatter(X_grid[:,0], X_grid[:,1], marker = 'h', s = 200, c = MSE_grid, 
                cmap = 'YlGnBu', alpha = 1, edgecolors='none')
    ax2.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)

    ax3.clear()
    ax3.scatter(X_grid[:,0], X_grid[:,1], s = 200, c = y_grid, 
                cmap = 'Spectral', alpha = 1, edgecolors='none')
    ax3.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)

    ax2.set_title('MSE')
    ax3.set_title('GP Prediction')
    
    ax6.scatter(X_grid[:,0], X_grid[:,1], s = 200, c = sp.absolute(y_grid - target_fun(X_grid)), 
                cmap = 'Spectral', alpha = 1, edgecolors='none')
    for axx in [ax2, ax3, ax6]:
        axx.set_xlabel('$CV 1$')
        axx.set_ylabel('$CV 2$')
        axx.set_xlim(X_grid[:,0].min(), X_grid[:,0].max())
        axx.set_ylim(X_grid[:,1].min(), X_grid[:,1].max())


def draw_nn_reconstruction(ax, net, X, Xnew, yminmax):
    ymin, ymax = yminmax
    ax2 = ax[1,0]
    ypred = net.network.predict(X)
    ypred[0], ypred[-1] = ymin, ymax 
    ax2.clear()
    ax2.scatter(X[:,0], X[:,1], s = 200, c = ypred, cmap = 'Spectral', alpha = 1, edgecolors='none')
    ax2.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)
    ax2.set_title('Neural Net Prediction')
    ax2.set_xlabel('$CV 1$')
    ax2.set_ylabel('$CV 2$')
    ax2.set_xlim(X[:,0].min(), X[:,0].max())
    ax2.set_ylim(X[:,1].min(), X[:,1].max())


def dataset_fringes(X, cluster_algo, min_compression=64):
    if cluster_algo =='none' or len(X) <= min_compression:
        return X
    elif cluster_algo == 'AffinityPropagation':
        algo = AffinityPropagation()
        D = -spsp.distance.squareform(sp.spatial.distance.pdist(X))
        algo.fit(D)
        return X[algo.cluster_centers_indices_]
    elif cluster_algo == 'DBSCAN':
        algo = DBSCAN(metric='precomputed', min_samples=2)
        D = -spsp.distance.squareform(sp.spatial.distance.pdist(X))
        labels = algo.fit(D).labels_
        return NearestCentroid().fit(X, labels).centroids_
    elif cluster_algo == 'svm_outlier':
        algo = svm.OneClassSVM(nu=0.95 * 0.25 + 0.05,
                               kernel="rbf") #, gamma=0.1)
        #UNFINISHED!!!
    else:
        print("BOH")
    

#######################################################
#######################################################
#######################################################



def main():
    Xmin, Xmax = sp.array([-6., -6.]), sp.array([6., 6.])
    theta = 1.0e-0
    nugget = 1.0e-8
    method = 'local_high_variance'# 'highest_variance_grid', 'local_high_variance'
    convergence_threshold = 1.0e-4
    n_features = 2
    cluster_algo = 'none'

    # Setup Gaussian Process

    gp = GaussianProcess(corr='squared_exponential', 
                         theta0=theta, thetaL=5e-2, thetaU=5e+1,
                         nugget = nugget, normalize=True)
    # gp = GaussianProcess(corr='squared_exponential', 
    #                      theta0=theta, thetaL=1e-1, thetaU=1e+2, 
    #                      nugget = nugget, normalize=False)

    # first evaluation points drawn at random from range
    X = sp.array([ 
                  [random.uniform(Xmin[0], Xmax[0]), 
                   random.uniform(Xmin[1], Xmax[1])] for i in range(2)])

    net = theanets.Experiment(theanets.Regressor,
                     layers=(2, 40, 30, 1),
                     activation='sigmoid',
                     algorithm='sgd')
            
    y = target_fun(X)
    yprime = h_diag_target_fun(X)

    # teach the first 2 trial points
    gp.fit(atleast_2d(X), y)
    # net.train([X, y[:,None]], allow_input_downcast=True)

    # setup plot
    plt.close('all')
    plt.ion()
    fig, ax = plt.subplots(3, 2, figsize=(18, 12))
    X_grid = Xplotgrid([Xmin, Xmax], n_features, 50)
    step = 0

    if method == 'local_high_variance':
        """
        --- Go wherever the predicted variance is highest around data points ---
        few points drawn at random -> evaluate energy + force
        train GP
        pick new x from neighbourhood of previous ones as the one with largest MSE
        evaluate energy landscape via GP for next value of x
        """
        n_test_pts = 10
        reach = 3.
        X_test = test_points(X, yprime, reach, n_test_pts)
        X_test = remove_pts_outside_boundaries(X_test, Xmin, Xmax)
        X_test = dataset_fringes(X_test, cluster_algo)
                
        # Xnew, y_pred, MSE, max_pred_err = highest_MSE_x(gp, X_test, X, [Xmin, Xmax])
        Xnew, y_pred, MSE, max_pred_err = pick_Xnew(gp, X_test, X, [Xmin, Xmax], yprime)
        draw_2Dplot(ax, X, y, X_test, y_pred, MSE, Xnew)
        yminmax = draw_target_fun(ax, X_grid)
        draw_2Dreconstruction(ax, gp, Xnew, X_grid, yminmax)
        # draw_nn_reconstruction(ax, net, X_grid, Xnew, yminmax)
        fig.canvas.draw()
    

        while max_pred_err > convergence_threshold:
            print("Step %03d | MAX ERROR: %.03e, MEAN ERROR: %.03e" 
                  % (step, max_pred_err, MSE.mean()))
            # plt.savefig("run-%03d.png" % step)
            # do new calculation on the point with largest predicted variance
            X = sp.vstack((X, Xnew))
            y = sp.hstack((y, target_fun(Xnew)))
            yprime = sp.vstack((yprime, h_diag_target_fun(Xnew)))
        
            gp.fit(atleast_2d(X), y)
        
            # net.train([X, y[:,None]], allow_input_downcast=True)
            X_test = test_points(X, yprime, reach, n_test_pts)
            X_test = remove_pts_outside_boundaries(X_test, Xmin, Xmax)
            X_test = dataset_fringes(X_test, cluster_algo)
                
            # Xnew, y_pred, MSE, max_pred_err = highest_MSE_x(gp, X_test, X, [Xmin, Xmax])
            Xnew, y_pred, MSE, max_pred_err = pick_Xnew(gp, X_test, X, [Xmin, Xmax], yprime)
            draw_2Dplot(ax, X, y, X_test, y_pred, MSE, Xnew)
            draw_2Dreconstruction(ax, gp, Xnew, X_grid, yminmax)
            # draw_nn_reconstruction(ax, net, X_grid, Xnew, yminmax)
            ax[2,0].scatter(step, sp.log(max_pred_err), c='r')
            ax[2,0].scatter(step, sp.log(MSE.mean()), c='g')
            ax[2,0].set_xlim((0, step))
            fig.canvas.draw()
            # print("mean force: %.03f" % sp.absolute(yprime).mean())
        
            # print("Step %03d | MAX ERROR: %.03f" % (step, max_pred_err))
            # plt.savefig('ajvar-%03d.png' % step)
            # time.sleep(3)
            step += 1

    
    elif method == 'highest_variance_grid':
        """
        --- Go wherever the predicted variance is highest ---
        2 points drawn at random -> evaluate energy + force
        train GP
        evaluate energy landscape via GP for a given range of values 
        err
        """
        n_test_pts = 200
        x_test = sp.linspace(xmin, xmax, n_test_pts)

        y_pred, MSE = gp.predict(atleast_2d(x_test).T, eval_MSE=True)
        max_pred_err = (sp.sqrt(MSE)).max()
        sort_order = sp.argsort(MSE)[::-1]
        lines = draw_plot(fig, x, y, x_test, y_pred, MSE)

        xnew = select_next_x(sort_order, x_test, x)
        while max_pred_err > convergence_threshold:
            # do new calculation on the point with largest predicted variance
            x = sp.hstack((x, xnew))
            y = target_fun(x).ravel() # wasteful, I'm recalculating all y. But who cares for now

            gp.fit(atleast_2d(x).T,y)

            xnew, y_pred, MSE, max_pred_err = highest_MSE_x(gp, x_test, x, [xmin, xmax])
        
            # y_pred, MSE = gp.predict(atleast_2d(x_test).T, eval_MSE=True)
            # max_pred_err = (sp.sqrt(MSE)).max()
            # sort_order = sp.argsort(MSE)[::-1]
            # xnew = select_next_x(sort_order, x_test, x)
            lines = draw_plot(fig, x, y, x_test, y_pred, MSE)
            print("Step %03d | MAX ERROR: %.03f" % (step, max_pred_err))
            # pl.savefig('ajvar-%03d.png' % step)
            # time.sleep(.4)
            step += 1 
        

if __name__ == '__main__':
    main()

