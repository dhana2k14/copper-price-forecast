# copper price forecast using LSTM 
# setting up necessary libraries 
# * this source code tested on tfp3.6 env with anaconda 

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# intial configurations

weeks_lag = 2  
max_date = 

# python functions

# read datasets and consolidate
# copper spot prices
print("Current Working Directory is %s" % os.getcwd())
main_df = pd.read_csv("./data/copper_df.csv", usecols = [0,3], parse_dates = ['Date'])
main_df['Date'] = main_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
main_df = main_df.groupby('Date', as_index = False).mean()
main_df = main_df.loc[main_df['Date'] >= '2013-12-31',:]
main_df.head()

# Copper Cathode prices 

cathode_df = pd.read_csv("./data/cu_cathode_df.csv", usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], parse_dates = ['Date'])
cathode_df['Date'] = cathode_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
cathode_df = cathode_df.groupby('Date', as_index = False).mean()
cathode_df = cathode_df.loc[cathode_df['Date'] >= '2013-12-31']
cathode_df.tail()

# Copper scarp prices

scrap_df = pd.read_csv("./data/cu_scrap_df.csv", usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8], parse_dates = ['Date'])
scrap_df['Date'] = scrap_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
scrap_df = scrap_df.groupby('Date', as_index = False).mean()
scrap_df = scrap_df.loc[scrap_df['Date'] >= '2013-12-31']
scrap_df.tail()

# Crude oil prices
# Convert to weekly data

oil_df = pd.read_csv("./data/crude_oil_df.csv", parse_dates = ['Date'])
temp_df = pd.date_range(start = '2012-01-01', end = '2018-11-31', freq = 'W')

# 




















## train-test split
#
#main_data = main_df.values
#train = main_df.iloc[0:600,:]
#test = main_df.iloc[600:,:]
#
## normalise data
#scalar = MinMaxScaler(feature_range = (-1, 1))
#scaled_data = scalar.fit_transform(main_data)
#
#x_train, y_train = [],[]
#for i in range(400, len(train)):
#    x_train.append(scaled_data[i-400:i,0])
#    y_train.append(scaled_data[i,0])
#
#x_train, y_train = np.array(x_train), np.array(y_train)
#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
## lstm network
#
#model = Sequential()
#model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences = True))
#model.add(LSTM(50))
#model.add(Dense(1))
#model.compile(loss = 'mean_squared_error', optimizer = 'adam')
#model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2)
#
## prediction
#inputs = main_df.iloc[len(main_df) - len(test) - 400:].values
#inputs = inputs.reshape(-1, 1)
#inputs = scalar.transform(inputs)
#
#X_test = []
#for i in range(400, inputs.shape[0]):
#    X_test.append(inputs[i-400:i,0])    
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#pred_price = model.predict(X_test)
#pred_price = scalar.inverse_transform(pred_price)
#
## Plot
#
#train = main_df.iloc[:600]
#test = main_df[600:]
#test['predictions'] = pred_price
#plt.plot(train['Spot'])
#plt.plot(test[['Spot', 'predictions']])

print(main_df['Date'].tail())
print(main_df['Date'].tail().apply(lambda x: x + pd.offsets.Week(2)))