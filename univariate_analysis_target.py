import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt

SEQ_LEN = 7
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "XOMX30"
EPOCHS = 5
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify (current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop("future", 1)
    df.dropna(inplace=True)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change(fill_method='ffill') #normalizing the data

            df[col] = preprocessing.scale(df[col].values) #scaling the data

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


df = pd.read_csv(f"G:\Programming\Projects\Index_price_movement\All_Stock_Data\{RATIO_TO_PREDICT}.csv")
df.rename(columns = {"Adj Close": "AdjClose"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.set_index("Date", inplace = True)
df = df[["AdjClose"]]

df['future'] = df["AdjClose"].shift(-FUTURE_PERIOD_PREDICT)

df['target'] = list(map(classify, df["AdjClose"], df["future"]))


main_df = df

times = sorted(main_df.index.values)
last_10pct = times[-int(0.10*len(times))]

validation_main_df = main_df[main_df.index >= last_10pct]
main_df = main_df[main_df.index < last_10pct]

train_x, train_y = preprocess_df(main_df)

validation_x, validation_y = preprocess_df(validation_main_df)

print (f"train_data: {len(train_x)} validation: {len(validation_x)}")
print (f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print (f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

#define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, FUTURE_PERIOD_PREDICT)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#fit model
model.fit(train_x, train_y, epochs=EPOCHS, verbose=1)

# demonstrate prediction
tboard = TensorBoard(log_dir=f"new_logs/{NAME}")

filepath = "new_models/RNN_Final-{epoch:02d}-{val_acc}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones
#callbacks_list = [tboard, checkpoint]

#history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), callbacks=callbacks_list, verbose=1)

#history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), verbose=1)

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

