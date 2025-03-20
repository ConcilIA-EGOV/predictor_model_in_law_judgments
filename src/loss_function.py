import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from util.parameters import FILE_PATH

def plotData(X,y,theta):
  plt.scatter(X,y,c='red')

  x0 = min(X)
  x1 = max(X)

  y0 = theta[0] + theta[1]*x0
  y1 = theta[0] + theta[1]*x1
  x_values = [x0, x1]
  y_values = [y0, y1]
  plt.plot(x_values, y_values, '-k')

def plotL2surface(X,y, theta0_vals, theta1_vals):

  # initialize L2_vals to a matrix of 0's
  L2_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

  # Fill out J_vals
  for i, theta0 in enumerate(theta0_vals):
       for j, theta1 in enumerate(theta1_vals):
                 L2_vals[i, j] = computeCost(X, y, np.array([theta0, theta1]))

  # Because of the way meshgrids work in the surf command, we need to
  # transpose J_vals before calling surf, or else the axes will be flipped
  L2_vals = L2_vals.T

  plt.contour(theta0_vals, theta1_vals, L2_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
  plt.xlabel('theta0')
  plt.ylabel('theta1')
  #plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
  plt.title('L2 Surface Contour')



def plotRegression(target,predicted):
   plt.title('R2: ' + str(r2_score(target, predicted)))
   plt.scatter(target,predicted,c='red')
   plt.xlabel('Target')
   plt.ylabel('Predicted')

   x0 = target.min()
   x1 = target.max()

   y0 = (LinearRegression().fit(target, predicted).predict(x0.reshape(1,-1))).flatten()
   y1 = (LinearRegression().fit(target, predicted).predict(x1.reshape(1,-1))).flatten()

   x_values = [x0, x1]
   y_values = [y0, y1]

   plt.plot(x_values, y_values,'-.k')


def computeCost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n), where m is the number of examples,
        and n is the number of features.
        We need to append a vector of one's to the features so we have
        n+1 columns.

    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, 1).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    L2 : float
        The value of the regression cost function.

    """
    # initialize some useful values
    m = np.size(y)  # number of training examples

    # make a copy of X and theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    Xin = X.copy() # copy training data to preserve original data
    thetain = theta.copy() # copy theta to preserve original data
    thetain = thetain.reshape(-1,1) # transform theta from a 1D array to 2D array

    # You need to return the following variables correctly
    L2 = 0

    # Add a column of ones to X. The numpy function stack joins arrays along a given axis.
    # The first axis (axis=0) refers to rows (training examples)
    # and second axis (axis=1) refers to columns (features).
    # a = np.ones((m,1))
    # Xin = np.hstack((a, Xin))

    # Calculate the hypothesis X * theta  [m,n+1] x [n+1,1] matrix multiplication
    hx = Xin @ thetain

    L2 = np.sum(np.sum(np.square(hx-y), axis=0))/(2.0*m)
    return L2
    


def normalEqn(X, y) -> np.ndarray:
    """
    Computes the closed-form solution to linear regression using the normal equations.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        The value at each data point. A vector of shape (m, 1).

    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, 1).

    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.

    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    We still need to add a column of 1s to the X matrix to have an
    intercept term ($\theta_0$). The code in the next cell will add the
    column of 1s to X for you.
    """
    m = y.size

    # make a copy of X and theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    Xin = X.copy()

    # Add a column of ones to X. The numpy function stack joins arrays along a given axis.
    # The first axis (axis=0) refers to rows (training examples)
    # and second axis (axis=1) refers to columns (features).
    # a = np.ones((m,1))
    # Xin = np.hstack((a, Xin))
    # ===================== YOUR CODE HERE ============================

    theta = np.linalg.pinv(Xin.T @ Xin) @ Xin.T @ y

    # =================================================================
    return theta.to_numpy()



def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).

    y : arra_like
        Value at given features. A vector of shape (m, 1).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # transform theta to a 2D-array
    #theta = theta.reshape(-1,1)
    # Initialize some useful values
    m = y.size  # number of training examples

    # make a copy of X and theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    Xin = X.copy()
    theta_tmp = theta.copy()



    # Add a column of ones to X. The numpy function stack joins arrays along a given axis.
    # The first axis (axis=0) refers to rows (training examples)
    # and second axis (axis=1) refers to columns (features).
    # a = np.ones((m,1))
    # Xin = np.hstack((a, Xin))
    # theta_tmp = np.reshape(theta_tmp,(Xin.shape[1],1))

    J_history = []      # Use a python list to save cost in every iteration
    theta0_history = [] # Use a python list to save theta0 in every iteration
    theta1_history = [] # Use a python list to save theta1 in every iteration

    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================

        aux = np.dot(Xin, theta_tmp) - y
        aux = np.dot(Xin.T, aux)
        theta_tmp = theta_tmp - (alpha/m) * aux

        # ==========================    ===========================================

        # save the theta0 and theta1 and cost J in every iteration

        aux1 = theta_tmp.flatten()
        theta0_history.append(aux1[0])
        theta1_history.append(aux1[1])
        J_history.append(computeCost(X, y, theta_tmp))

    return theta_tmp, theta0_history, theta1_history, J_history


def main():
    pass

if __name__ == '__main__':
    main()
