from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import numpy as np
import statsmodels
#import matplotlib.pyplot as plt

from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from sklearn.model_selection import train_test_split
from pandas import DataFrame

import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error



df = pd.read_csv("G:\Programming\Projects\Index_price_movement\All_Stock_Data\XOMX30.csv")
df = df.drop("Volume", 1)
df = df.drop("Open", 1)
df = df.drop("Low", 1)
df = df.drop("High", 1)
df = df.drop("Adj Close", 1)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.set_index("Date", inplace = True)
df = df.dropna()

series = df["2000-01-01" :]

#train = df["2000-01-01" : "2018-12-31"]
#test = df["2019-01-01":]

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    # train_size = int(len(X) * 0.66)
    # train, test = X[0:train_size], X[train_size:]
    # train = X["2000-01-01": "2018-12-31"].values
    # test = X["2019-01-01":].values

    train = X["2018-01-01": "2019-04-10"].values
    test = X["2019-04-10":].values


    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# load dataset
#series = Series.from_csv('daily-total-female-births.csv', header=0)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series, p_values, d_values, q_values)