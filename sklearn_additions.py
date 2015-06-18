from __future__ import print_function, division

import scipy as sp
from sklearn.utils import check_X_y
from sklearn.gaussian_process import GaussianProcess
from sklearn.metrics.pairwise import manhattan_distances

##### Functions to be added to sklearn GaussianProcess Class #####
def predict_gradient_(self, X):
    """
    This function evaluates the Gaussian Process model at x.
    
    Parameters
    ----------
    X : array_like
        An array with shape (n_eval, n_features) giving the point(s) at
        which the prediction(s) should be made.
    
    Returns
    -------
    yprime : array_like, shape (n_eval, n_features).
    """
    
    # assert self.corr == 'squared_exponential'
    
    X = (X - self.X_mean) / self.X_std
    X = sp.atleast_2d(X)
    
    n_eval, _ = X.shape
    n_samples, n_features = self.X.shape
    n_samples_y, n_targets = self.y.shape
    
    # Get pairwise componentwise L1-distances to the input training set
    dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
    # Get regression correlation
    r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)
    # Get derivative of regression correlation
    dr_dx = -2 * self.theta_ * (X - self.X) * r.T
    # Predictor of function derivative
    yprime = self.y_std * sp.dot(dr_dx.T, self.gamma)
    return yprime


def accumulate_data(self, X, y):
    """
    Add datapoints without fitting them.
    Parameters
    ----------
    X : double array_like
        An array with shape (n_samples, n_features) with the input at which
        new observations were made.
    y : double array_like
        An array with shape (n_samples, ) or shape (n_samples, n_targets)
        with the new observations of the output to be predicted.
    Returns
    -------
    gp : self
        A Gaussian Process model object.
    """
    # Force data to 2D numpy.array
    X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
    if y.ndim == 1:
        y = y[:, None]
    
    if hasattr(self, 'X'):
        # retrieve old X and y to make them compatible with new observations.
        if self.normalize:
            X_old = self.X_std * (self.X + self.X_mean)
            y_old = self.y_std * (self.y + self.y_mean)
        else:
            X_old, y_old = self.X, self.y
        # stack unscaled old and new data together
        self.X, self.y = sp.vstack((X_old, X)), sp.vstack((y_old, y))
    else:
        # No data present
        self.X, self.y = X, y
    # Clear out the data scaling
    self.X_mean, self.y_mean = sp.zeros(1), sp.zeros(1)
    self.X_std, self.y_std = sp.ones(1), sp.ones(1)
    
    return self


def update_fit_rough(self, X, y):
    """
    The Gaussian Process model fitting method.
    Parameters
    ----------
    X : double array_like
        An array with shape (n_samples, n_features) with the input at which
        new observations were made.
    y : double array_like
        An array with shape (n_samples, ) or shape (n_samples, n_targets)
        with the new observations of the output to be predicted.
    Returns
    -------
    gp : self
        A re-fitted Gaussian Process model object awaiting data to perform
        predictions.
    """
    # Add (X, y) training points to the current dataset. 
    self.accumulate_data(X, y)
    # start ML optimisation from the previously optimised hyperparameters (if present)
    if hasattr(self, 'theta_'):
        self.theta0 = self.theta_
    # perform fitting
    self.fit(self.X, self.y)
    return self
#############################################################