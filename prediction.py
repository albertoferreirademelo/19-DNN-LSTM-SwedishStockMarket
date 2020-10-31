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

#from keras.optimizers import sgd, Adam
#import tensorboard
#from tensorflow.contrib.rnn import *
#import matplotlib.pyplot as plt



#directory = "stock_data_yahoo_finance"
directory = "All_Stock_Data"
files = os.listdir(directory)


SEQ_LEN = 2
FUTURE_PERIOD_PREDICT = 1
RATIO_TO_PREDICT = "XOMX30"
EPOCHS = 100
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def preprocess_df(df):
    #print (df.head())
    df = df.drop("future", 1)
    #print(df.head())
    df.dropna(inplace=True)
    #print(df.head())

    for col in df.columns:
        if col != "target":
            #df[col].plot(title=col, figsize=(10,4))
            #plt.show()
            #plt.plot(df[col])
            #print (df.info())
            df[col] = df[col].pct_change(fill_method='ffill') #normalizing the data
            #print(df[col])
            #print (df.info())
            #df.dropna(inplace=True)
            #df[col].plot(title=(f"{col} Normalized"), figsize=(10, 4))
            #plt.show()
            #plt.savefig(f"{col}_NORMALIZED.png")
            #print (df[col].head())

            #df[col].plot(title=(f"{col} Normalized"), figsize=(10, 4))
            #plt.plot(df[col])

            #plt.savefig(f"{col}.png")
            #plt.clf()
            #plt.show()
            #df[col] = df[col].values
            df[col] = preprocessing.scale(df[col].values) #scaling the data
            #df[col].plot(title=(f"{col} Scaled"), figsize=(10, 4))
            #plt.savefig(f"{col}_Scaled.png")
            #plt.clf()
            #input("waiting...")

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



def list_together(listan, directory):
    main_df = pd.DataFrame()
    df = pd.DataFrame()
    for company in listan:
        df = pd.read_csv(f"{directory}/{company}")
        company = company.replace(".csv", "")
        df.rename(columns = {"Adj Close": f"{company}_AdjClose"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df.set_index("Date", inplace = True)
        df = df[[f"{company}_AdjClose"]]

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)

    main_df.to_csv("ALL_OMX30_adjclose.csv")
    #df.dropna(inplace = True)
    #print (main_df.info())
    return main_df

def classify (current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

df = list_together(files, directory)

# for col in df.columns:
#     df[col].plot(title=(f"{col} Normalized"), figsize=(10, 4))
#     plt.savefig(f"{col}.png")
#     plt.clf()

df['future'] = df[f"{RATIO_TO_PREDICT}_AdjClose"].shift(-FUTURE_PERIOD_PREDICT)

df['target'] = list(map(classify, df[f"{RATIO_TO_PREDICT}_AdjClose"], df["future"]))

main_df = df

times = sorted(main_df.index.values)
last_5pct = times[-int(0.10*len(times))]


validation_main_df = main_df[main_df.index >= last_5pct]
main_df = main_df[main_df.index < last_5pct]


train_x, train_y = preprocess_df(main_df)

validation_x, validation_y = preprocess_df(validation_main_df)

print (f"train_data: {len(train_x)} validation: {len(validation_x)}")
print (f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print (f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
#model.add(Dense(32, activation="sigmoid"))
model.add(Dropout(0.1))

model.add(Dense(2, activation="softmax"))

opt = optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#model.compile(loss="mean_squared_error", optimizer="sgd")

tboard = TensorBoard(log_dir=f"logs/{NAME}")

filepath = "models/RNN_Final-{epoch:02d}-{val_acc}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones
callbacks_list = [tboard, checkpoint]

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), callbacks=callbacks_list, verbose=2)

print(history.history.keys())

#Anaconda tensorflow-gpu installer: run "conda create --name tf_gpu tensorflow-gpu"