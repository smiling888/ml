import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn import linear_model
reg = linear_model.LinearRegression()
#reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
#print(reg.coef_)

# [x,y]
def bingqiling():
    dataX = [[25],[27],[31],[33],[35]]
    dataY=[110,115,155,160,180]
    reg.fit(dataX,dataY)
    diabetes_y_pred = reg.predict(dataX)
    print(reg.coef_)
    #sandiantu
    plt.scatter(dataX, dataY, color='black', linewidth=3)
    #line
    plt.plot(dataX, diabetes_y_pred, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    plt.show()

bingqiling()


## https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
def diabetes():


    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

#diabetes()