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

directory = "All_Stock_Data"
files = os.listdir(directory)

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
        #print (df.info())
        print (df.count())
        print ("---------------------------")
        #print(f"Tail: {main_df.tail(1)}")

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)

    #df.to_csv("OMX30_adjclose.csv")
    #df.dropna(inplace = True)
    #print (main_df.info())
    return main_df

df = list_together(files, directory)