# copper price forecast using LSTM 
# setting up necessary libraries 
# * this source code tested on tfp3.6 env with anaconda 

import pandas as pd
import os
import numpy as np
from numpy import concatenate
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# read datasets and consolidate
# copper spot prices
print("Current Working Directory is %s" % os.getcwd())
main_df = pd.read_csv("./data/copper_df.csv", usecols = [0,3], parse_dates = ['Date'])
main_df['Date'] = main_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
main_df = main_df.groupby('Date', as_index = False).mean()
main_df = main_df.loc[main_df['Date'] >= '2013-12-31',:]
main_df.tail()

# Copper Cathode prices 

cathode_df = pd.read_csv("./data/cu_cathode_df.csv", usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], parse_dates = ['Date'])
cathode_df['Date'] = cathode_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
cathode_df = cathode_df.groupby('Date', as_index = False).mean()
cathode_df = cathode_df.loc[cathode_df['Date'] >= '2013-12-31']
cathode_df = cathode_df.fillna(method = 'bfill')
cathode_df.tail()

# Copper scrap prices

scrap_df = pd.read_csv("./data/cu_scrap_df.csv", usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8], parse_dates = ['Date'])
scrap_df['Date'] = scrap_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
scrap_df = scrap_df.groupby('Date', as_index = False).mean()
scrap_df = scrap_df.loc[scrap_df['Date'] >= '2013-12-31']
scrap_df.tail()

# Crude oil prices
# Convert to weekly data

oil_df = pd.read_csv("./data/crude_oil_df.csv", parse_dates = ['Date'], dtype = {'Crude_Oil_Index':np.float64})
temp_df = pd.DataFrame(pd.date_range(start = pd.to_datetime('2012-01-01'), end = max(oil_df['Date']), freq = 'W'), columns = ['Date'])
temp_df = pd.merge(temp_df, oil_df, how = 'left', on = 'Date')
oil_df = temp_df.fillna(method = 'ffill')
oil_df.tail()

# Copper demand & supply

demand_df = pd.read_csv("./data/cu_demand_df.csv", parse_dates = ['Date'], dtype = {'Production':np.float64, 'Consumption':np.float64})
temp_df = pd.DataFrame(pd.date_range(start = pd.to_datetime('1998-01-01'), end = pd.to_datetime('2019-12-31'), freq = 'MS'), columns = ['Date'])
temp_df = pd.merge(temp_df, demand_df, how = 'left', on = 'Date')
temp_df['Date'] = temp_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
demand_df = temp_df.fillna(method = 'ffill')
demand_df.tail()

# Copper Concentrate prices

concen_df = pd.read_csv("./data/cu_concentrate_df.csv", parse_dates = ['Date'], dtype = {'CU_Concentrate_TC':np.float64,'CU_Concentrate_RC':np.float64})
concen_df['Date'] = concen_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
concen_df = concen_df.groupby('Date', as_index = False).mean()
concen_df.tail()

# merge 

main_df = pd.merge(main_df, cathode_df, how = 'left', on = 'Date')
main_df = pd.merge(main_df, scrap_df, how = 'left', on = 'Date')
main_df = pd.merge(main_df, demand_df, how = 'left', on = 'Date')
main_df = pd.merge(main_df, concen_df, how = 'left', on = 'Date')
# replace missing values with backfill

main_df['Production'] = main_df['Production'].fillna(method = 'bfill')
main_df['Consumption'] = main_df['Consumption'].fillna(method = 'bfill')
main_df['CU_Concentrate_TC'] = main_df['CU_Concentrate_TC'].fillna(method = 'bfill')
main_df['CU_Concentrate_RC'] = main_df['CU_Concentrate_RC'].fillna(method = 'bfill')
main_df = main_df.fillna(method = 'ffill')

## two models (training sequence, testing sequence)
# name Date as index 

main_df.index = main_df['Date']
main_df.drop('Date', axis = 1, inplace = True)
values = main_df.values

# train-test split
train = main_df.iloc[0:234,:]
test = main_df.iloc[234:,:]

# normalize features

scalar = MinMaxScaler(feature_range = (-1, 1))
scaled_data = scalar.fit_transform(values)
print("Transformed Data Shape : {}".format(scaled_data.shape))

# training and test sequence
train_X, train_y = [], []
for i in range(100, len(train)):
    train_X.append(scaled_data[i-100:i,:])
    train_y.append(scaled_data[i,0])
    
train_X, train_y = np.array(train_X), np.array(train_y)

# model
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = False))
model.add(Dense(1))
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(train_X, train_y, epochs = 10, batch_size = 100, verbose = 2)
print("Model Summary")
model.summary()

# test-cases 

inputs = main_df.iloc[len(main_df) - len(test) - 100:].values
inputs = scalar.transform(inputs)

X_test = []
for i in range(100, inputs.shape[0]):
    X_test.append(inputs[i-100:i,:])    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# prediction - testcases 
pred = model.predict(X_test)
pred = concatenate((scaled_data[-29:,1:], pred), axis = 1)
pred = scalar.inverse_transform(pred)
pred_df = pd.DataFrame(pred)

# plotting

main_df.reset_index(inplace = True) # for merging on index
train = main_df.iloc[0:234,:]
test = main_df.iloc[234:,:]
test.reset_index(inplace = True)
test['pred_price'] = pred_df.iloc[:,0]
plt.plot(train['copper_price'])
plt.plot(test[['copper_price', 'pred_price']])

# write to file

test.to_csv('../output/multi-seq-lstm-output_wt_data_till_Jan_23Jan19.csv', index = False)
        
# ******************************************************************************************
## prediction - Multisteps into Future
# ******************************************************************************************
# Line 14 to 77 remain same:

main_df.index = main_df['Date']
main_df.drop('Date', axis = 1, inplace = True)
values = main_df.values

# train-test split

train = main_df.iloc[0:262,:]
test = main_df.iloc[262:,:]

# normalize features

scalar = MinMaxScaler(feature_range = (-1, 1))
scaled_data = scalar.fit_transform(values)
print("Transformed Data Shape : {}".format(scaled_data.shape))

# training and test sequence
train_X, train_y = [], []
for i in range(100, len(train)):
    train_X.append(scaled_data[i-100:i,:])
    train_y.append(scaled_data[i,0])
    
train_X, train_y = np.array(train_X), np.array(train_y)

# model
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = False))
model.add(Dense(1))
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(train_X, train_y, epochs = 10, batch_size = 100, verbose = 2)
print("Model Summary")
model.summary()

# test-cases 
inputs = main_df.iloc[len(main_df) - len(test)  - 100:].values
inputs = scalar.transform(inputs)

#X_test = []
#for i in range(100, inputs.shape[0]):
#    X_test.append(inputs[i-100:i,:])    
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
#
#pred = model.predict(X_test)
#pred = concatenate((pred, scaled_data[-1:,1:]), axis = 1)


# Method - 1
# prediction - testcases 

df = pd.DataFrame()
for i in range(0, 50):
    X_test = []
    inputs = inputs[1:,:]
    X_test.append(inputs)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    pred = model.predict(X_test)
    pred = concatenate((pred, inputs[-1:,1:]), axis = 1)
    inputs = concatenate((inputs, pred), axis = 0)
    pred = scalar.inverse_transform(pred)
    df = df.append(pd.DataFrame(pred))

# Method - 2 













