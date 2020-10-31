import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("G:\Programming\Projects\Index_price_movement\All_Stock_Data\XOMX30.csv")
print (df.info())
df = df.drop("Volume", 1)
#df = df.drop("Open", 1)
#df = df.drop("Low", 1)
#df = df.drop("High", 1)
df = df.drop("Adj Close", 1)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.set_index("Date", inplace = True)
df = df.dropna()

df = df["2000-01-01" :]


def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape

    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))

    return X_train, X_test


def preprocess_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.values

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[: int(row), :]

    train, result = standard_scaler(train, result)

    X_train = train[:, : -1]
    y_train = train[:, -1][:, -1]
    X_test = result[int(row):, : -1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    #print (f"X_test: {X_test}")

    return [X_train, y_train, X_test, y_test]

def build_model(layers):
    print (layers)

    #print (f"Layers: {layers}")
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

window = 20
X_train, y_train, X_test, y_test = preprocess_data(df[:: -1], window)
# print("X_train", X_train.shape)
# print("y_train", y_train.shape)
# print("X_test", X_test.shape)
# print("y_test", y_test.shape)

#print (X_train.shape[2])
model = build_model([X_train.shape[2], window, 100, 1])

model.fit(X_train, y_train, batch_size=768, epochs=5, validation_split=0.1, verbose=1)

trainScore = model.evaluate(X_train, y_train, verbose=1)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=1)
print (testScore)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

diff = []
ratio = []
pred = model.predict(X_test)
for u in range(len(y_test)):
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

plt.plot(pred, color='red', label='Prediction')
plt.plot(y_test, color='blue', label='Ground Truth')
plt.legend(loc='upper left')
plt.show()