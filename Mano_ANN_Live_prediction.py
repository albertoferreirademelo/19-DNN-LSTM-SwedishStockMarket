import h5py
import numpy as np
import pandas as pd
import random

from keras.models import load_model
from collections import deque
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers

# with h5py.File("G:\Programming\Projects\Index_price_movement\ManoModels\RNN_Final-11-0.9876543209876543.hdf5", "r") as f:
#     x_data = f["x_data"]
#     prediction = model.predict(x_data)
#     print (prediction)


SEQ_LEN = 20

####################################################################################################

data_csv = pd.read_csv("G:\Programming\Projects\Index_price_movement\OMX30_2019.csv")

data_csv["Date"] = pd.to_datetime(data_csv["Date"], format="%Y-%m-%d")
data_csv.set_index("Date", inplace = True)

data_csv = data_csv.drop("Volume", 1)
data_csv = data_csv.drop("Adj Close", 1)
data_csv = data_csv.drop("Open", 1)
data_csv = data_csv.drop("High", 1)
data_csv = data_csv.drop("Low", 1)

####################################################################################################

def preprocess_df(df):
    #X = []
    for col in df.columns:
        if col != "Target":
            df[col] = df[col].pct_change(fill_method='ffill')  # normalizing the data
            df[col] = preprocessing.scale(df[col].values)  # scaling the data

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = []
    #prev_days = deque(maxlen=SEQ_LEN)
    #print (sequential_data)

    for i in df.values:
        prev_days.append(i)
        if len(prev_days) >= SEQ_LEN:
            sequential_data.append(prev_days[-SEQ_LEN:])
        #sequential_data.append([n for n in i[:-1]])
        #print (prev_days)
        #if len(prev_days) == SEQ_LEN:
            #sequential_data.append([np.array(prev_days)])

    #print (sequential_data)

    #for seq in sequential_data:
        #X.append(seq)

    #return np.array(X)
    # for i in sequential_data:
    #     print (i)
    #print (sequential_data)
    return sequential_data


model = load_model("G:\Programming\Projects\Index_price_movement\ManoModels\RNN_Final-11-0.9876543209876543.hdf5")
opt = optimizers.Adam(lr=0.001, decay=1e-6)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

preprocessed_data = preprocess_df(data_csv)

print (model.predict_classes(np.array(preprocessed_data)))

# for i in preprocessed_data:
#     print (np.array(i))
#     print ("--------")
#     #print (model.predict(np.array(i)))
#     #print (len(i))
#
# #validation_x, validation_y = preprocess_df(data_csv)
#
# #print (validation_x)
# #print (validation_y)


