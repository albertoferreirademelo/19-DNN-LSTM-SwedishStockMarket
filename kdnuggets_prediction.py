import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers, callbacks
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os, datetime
import json
import pickle


TIME_STEPS = 60
BATCH_SIZE = 20
EPOCHS = 5
lr = 0.0001

date_now = datetime.datetime.now()
date_now = date_now.strftime("%d%B%H%M%S")




df = pd.read_csv("G:\Programming\Projects\Index_price_movement\All_Stock_Data\XOMX30.csv")
#df = df["2000-01-01" :]
df = df.drop("Volume", 1)
df = df.drop("Adj Close", 1)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.set_index("Date", inplace = True)
df = df.dropna()
# print (df.tail())
#
# plt.figure()
# plt.plot(df["Close"])
# plt.title("Swedish stock price (OMX30) history")
# plt.ylabel("Price (SEK)")
# plt.xlabel("Years")
# plt.show()



train_cols = ["Open", "High", "Low", "Close"]
df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])


def build_timeseries(mat, y_col_index):
    #TIME_STEPS = 60
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    # for i in tqdm.tqdm_notebook(range(dim_0)):
    #     x[i] = mat[i:TIME_STEPS + i]
    #     y[i] = mat[TIME_STEPS + i, y_col_index]
    #print("length of time-series i/o", x.shape, y.shape)
    return x, y


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    #print (no_of_rows_drop)
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def find_lowest_model (files):
    lowest_possible = 0
    saving_file_name = "a"
    for i in files:
        if (int(i[-5:-3])) > lowest_possible:
            print (i[-5:-3])
            lowest_possible = (int(i[-5:-3]))
            #print (i)
            saving_file_name = i
    return saving_file_name



#Building training set and validation set
x_t, y_t = build_timeseries(x_train, 3)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = build_timeseries(x_test, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)


#Creating the ANN
lstm_model = Sequential()
es = EarlyStopping(monitor="val_loss", verbose=1, patience=40, min_delta=0.0001)
filepath = "new_models/"
mcp = ModelCheckpoint(os.path.join(filepath, "RNN_Final-{epoch:02d}-{val_loss}.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1, save_weights_only=True)

#csv_logger = callbacks.CSVLogger(os.path.join("./towarddatascience_log", "training_log_" + time.ctime().replace(" ", "_") + '.log'), append=True)
csv_logger = callbacks.CSVLogger(os.path.join("./towarddatascience_log", date_now + '.log'), append=True)

# This comments down below is from another tutorial
# filepath = "models/RNN_Final-{epoch:02d}-{val_acc}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones


lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform', return_sequences=True))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))

lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform', return_sequences=True))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))

lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))


optimizer = optimizers.RMSprop(lr=lr)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)


history = lstm_model.fit(x_t, y_t, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE, shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE)), callbacks=[es, mcp, csv_logger])



filename = (date_now+".sav")

# model_json = model.to_json()
# with open("model_in_json.json", "w") as json_file:
#     json.dump(model_json, json_file)
#
# model.save_weights("model_weights.h5")

#pickle.dump(lstm_model, open("./new_models/", "wb"))


lstm_model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE)
y_pred = lstm_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])


# convert the predicted value to range of real data
y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
# min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
# min_max_scaler.inverse_transform(y_test_t)
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

# Visualize the training data
# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# list_of_files = os.listdir("new_models")
# lowest_number = find_lowest_model(list_of_files)
#
# saved_model = load_model(f"new_models/{lowest_number}")
#
# y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
# y_pred = y_pred.flatten()
# y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
# error = mean_squared_error(y_test_t, y_pred)
# print("Error is", error, y_pred.shape, y_test_t.shape)
# print(y_pred[0:15])
# print(y_test_t[0:15])
# y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_pred)
# y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_test_t)
# print(y_pred_org[0:15])
# print(y_test_t_org[0:15])


plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()







# callbacks_list = [tboard, checkpoint]
#
# history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), callbacks=callbacks_list, verbose=2)
