import os
import pandas as pd

files = os.listdir("stock_data_yahoo_finance")
# print (files)

# def list_together(listan):
#     df = pd.DataFrame()
#     for company in listan:
#         print (company)
#         data = pd.read_csv(f"stock_data_yahoo_finance/{company}")
#         data.rename(columns = {"Adj Close": f"{company}_AdjClose", "Volume": f"{company}_Volume"}, inplace=True)
#         df = pd.concat([df, data[f"{company}_AdjClose"], data[f"{company}_Volume"]], axis = 1)
#
#     df.to_csv("all_OMX30_clean.csv")
#     df.dropna(inplace = True)
#     return df

def list_together(listan):
    df = pd.DataFrame()
    for company in listan:
        print (company)
        data = pd.read_csv(f"stock_data_yahoo_finance/{company}")
        data.rename(columns = {"Adj Close": f"{company}_AdjClose"}, inplace=True)
        df = pd.concat([df, data[f"{company}_AdjClose"]], axis = 1)

    #df.to_csv("OMX30_adjclose.csv")
    df.dropna(inplace = True)
    return df


list_together(files)





