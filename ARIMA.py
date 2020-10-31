import pandas as pd
import numpy as np
import statsmodels
#import matplotlib.pyplot as plt

from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
from pandas import DataFrame


df = pd.read_csv("G:\Programming\Projects\Index_price_movement\All_Stock_Data\XOMX30.csv")
df = df.drop("Volume", 1)
df = df.drop("Open", 1)
df = df.drop("Low", 1)
df = df.drop("High", 1)
df = df.drop("Adj Close", 1)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.set_index("Date", inplace = True)
df = df.dropna()

df = df["2000-01-01" :]

''' This area is to check how the data look alike and decide what order to use. 

#print (df.head())

#df.plot()
#autocorrelation_plot(df) #Show autocorrelation
#pyplot.show()


model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit(disp=0)

#print (model_fit.summary())

residuals = DataFrame(model_fit.resid)
fig, ax = pyplot.subplots(1,2)
#fig, ax = plt.subplot(1,2, sharex=True)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind="kde", title="Density", ax=ax[1])
model_fit.plot_predict(dynamic=False)
#pyplot.show()
#print (residuals.describe())
'''
#train, test = train_test_split(df, train_size=0.99, test_size=0.01, shuffle=False)
#train = df["2000-01-01" : "2018-12-31"]
#test = df["2019-01-01":]
train = df["2018-01-01" : "2019-04-10"]
test = df["2019-09":]
print (train.head())
print (train.tail())
print (test.head())
print (test.tail())
model = ARIMA(train, order=(1,2,1))
#model = ARIMA(train, order=(1,1,1))
fitted = model.fit(disp=-1)

print (fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(50, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
pyplot.figure(figsize=(12,5), dpi=100)
pyplot.plot(train, label='training')
pyplot.plot(test, label='actual')
pyplot.plot(fc_series, label='ARIMA forecast')
pyplot.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
pyplot.title('Forecast vs Actuals')
pyplot.legend(loc='upper left', fontsize=8)
pyplot.show()

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    #corr = np.corrcoef(forecast, actual)[0,1]   # corr
    #mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    #maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    #minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(forecast-actual)[1]                      # ACF1
    #return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 'corr':corr, 'minmax':minmax})
    return (
    {'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe, 'rmse': rmse})

# print (len(fc))
# print (len(test.values))
print (forecast_accuracy(fc, test.values))
#forecast = fc
#actual = test.values
#print (np.mean(np.abs(fc - test.values)/np.abs(test.values)))
#print (np.mean(forecast - actual))
#print (test.values)