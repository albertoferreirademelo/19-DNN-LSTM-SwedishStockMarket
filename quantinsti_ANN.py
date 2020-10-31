#With this data I could see a trend going down and up, which is what the training set did. But still not good enough

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

df = pd.read_csv("G:\Programming\Projects\Index_price_movement\All_Stock_Data\XOMX30.csv")
df = df.drop("Volume", 1)
df = df.drop("Open", 1)
df = df.drop("Low", 1)
df = df.drop("High", 1)
df = df.drop("Adj Close", 1)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.set_index("Date", inplace = True)
df = df.dropna()

prediction_df = df["2018-12-31" : "2019-04-26"]

df = df["2015-01-01" : "2018-12-31"]

scaler = MinMaxScaler(feature_range = (0, 1))

apple_training_scaled = scaler.fit_transform(df)

features_set = []
labels = []
for i in range(60, 1005): #4915 total of data
    features_set.append(apple_training_scaled[i-60:i, 0])
    labels.append(apple_training_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)

features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(features_set, labels, epochs = 3, batch_size = 32)

apple_testing_complete = prediction_df

apple_total = pd.concat((df["Close"], apple_testing_complete["Close"]), axis=0)

test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values

test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

test_features = []
#for i in range(60, 300):
print (len(test_inputs))
for i in range(60, 80):
    test_features.append(test_inputs[i-60:i, 0])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

predictions = model.predict(test_features)

predictions = scaler.inverse_transform(predictions)

plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.plot(prediction_df["Close"].values, color='blue', label='Actual Stock Price')
plt.show()
# plt.figure(figsize=(10,6))
# plt.plot(prediction_df, color='blue', label='Actual Apple Stock Price')
# plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
# plt.title('Apple Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Apple Stock Price')
# plt.legend()
# plt.show()