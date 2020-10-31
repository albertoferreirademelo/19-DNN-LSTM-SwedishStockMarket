import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from collections import deque
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers

Future_to_predict = 20
PROCENT = 0.10
SEQ_LEN = 100
BATCH_SIZE = 64
EPOCHS = 15
NAME = f"{SEQ_LEN}-SEQ-{Future_to_predict}-PRED-{int(time.time())}"

####################################################################################################

data_csv = pd.read_csv("G:\Programming\Projects\Index_price_movement\All_Stock_Data\XOMX30.csv")

data_csv["Date"] = pd.to_datetime(data_csv["Date"], format="%Y-%m-%d")
data_csv.set_index("Date", inplace = True)

data_csv = data_csv.drop("Volume", 1)
data_csv = data_csv.drop("Adj Close", 1)
data_csv = data_csv.drop("Open", 1)
data_csv = data_csv.drop("High", 1)
data_csv = data_csv.drop("Low", 1)

data_csv = data_csv["2001-01-01":]

####################################################################################################

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop("Future", 1)
    df.dropna(inplace=True)

    print (df.columns)

    for col in df.columns:
        if col != "Target":
            df[col] = df[col].pct_change(fill_method='ffill')  # normalizing the data
            plt.plot(df[col], label= "pct_change")
            #plt.show()

            df[col] = preprocessing.scale(df[col].values)  # scaling the data
            plt.plot(df[col], label= "scaled")
            #plt.show()


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

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        #print (seq)
        #print ("--------")
        X.append(seq)
        y.append(target)

    return np.array(X), y


data_csv["Future"] = data_csv["Close"].shift(Future_to_predict)
data_csv["Target"] = list(map(classify, data_csv["Close"], data_csv["Future"]))

times = sorted(data_csv.index.values)
last_pct = times[-int(PROCENT*len(times))]

validation_main_df = data_csv[data_csv.index >= last_pct]
main_df = data_csv[data_csv.index < last_pct]

#print (validation_main_df.head())
#print (main_df.head())

train_x, train_y = preprocess_df(main_df)

validation_x, validation_y = preprocess_df(validation_main_df)


print (f"train_data: {len(train_x)} validation: {len(validation_x)}")
print (f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print (f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

#####################################################################################################
#ANN

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(2, activation="softmax"))
#model.add(Dense(1, activation="sigmoid")) #triyng something nbew

opt = optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"]) #tying something new

tboard = TensorBoard(log_dir=f"ManoLogs/{NAME}")

filepath = "ManoModels/RNN_Final-{epoch:02d}-{val_acc}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones
#checkpoint = ModelCheckpoint(filepath, monitor='mse', verbose=1, save_best_only=True, mode='min') # saves only the best ones #trying something new
callbacks_list = [tboard, checkpoint]

# for i in validation_x:
#     print (i)
#     print ("---------")
history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), callbacks=callbacks_list, verbose=2)
#history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2) #tryiing something new
pred = model.predict(validation_x)
#print (pred)




############################################################################################################################################
#Trying to show the graph


#y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))

#plt.plot( y_test, label="actual")
#plt.plot(pred1, label="predictions



#print (validation_main_df.head(20))
#print (validation_main_df.tail(20))