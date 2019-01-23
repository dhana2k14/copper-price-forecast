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

# python functions
# input sequence 
#def series_to_supervised(data, n_in, n_out, dropnan = True):
#    names, cols = [], []
#    n_vars = data.shape[1]
#    df = pd.DataFrame(data)
#    for i in range(n_in, 0, -1):
#        cols.append(df.shift(i))
#        names += ["var%d(t-%d)" % (j+1, i) for j in range(n_vars)]
## forecast sequence
#    for i in range(0, n_out):
#        cols.append(df[0].shift(-i))
#        if i == 0:
#            names += ["var%d(t)" % (j+1) for j in range(1)] 
#        else:
#            names += ["var%d(t+%d)" % (j+1, i) for j in range(1)]
#    agg = concat(cols, axis = 1)
#    agg.columns = names
#    if dropnan:
#        agg.dropna(inplace = True)
#    return agg

# read datasets and consolidate
# copper spot prices
print("Current Working Directory is %s" % os.getcwd())
main_df = pd.read_csv("../data/copper_df.csv", usecols = [0,3], parse_dates = ['Date'])
main_df['Date'] = main_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
main_df = main_df.groupby('Date', as_index = False).mean()
main_df = main_df.loc[main_df['Date'] >= '2013-12-31',:]
main_df.tail()

# Copper Cathode prices 

cathode_df = pd.read_csv("../data/cu_cathode_df.csv", usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], parse_dates = ['Date'])
cathode_df['Date'] = cathode_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
cathode_df = cathode_df.groupby('Date', as_index = False).mean()
cathode_df = cathode_df.loc[cathode_df['Date'] >= '2013-12-31']
cathode_df = cathode_df.fillna(method = 'bfill')
cathode_df.tail()

# Copper scrap prices

scrap_df = pd.read_csv("../data/cu_scrap_df.csv", usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8], parse_dates = ['Date'])
scrap_df['Date'] = scrap_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
scrap_df = scrap_df.groupby('Date', as_index = False).mean()
scrap_df = scrap_df.loc[scrap_df['Date'] >= '2013-12-31']
scrap_df.tail()

# Crude oil prices
# Convert to weekly data

oil_df = pd.read_csv("../data/crude_oil_df.csv", parse_dates = ['Date'], dtype = {'Crude_Oil_Index':np.float64})
temp_df = pd.DataFrame(pd.date_range(start = pd.to_datetime('2012-01-01'), end = pd.to_datetime('2018-11-30'), freq = 'W'), columns = ['Date'])
temp_df = pd.merge(temp_df, oil_df, how = 'left', on = 'Date')
oil_df = temp_df.fillna(method = 'ffill')
oil_df.tail()

# Copper demand & supply

demand_df = pd.read_csv("../data/cu_demand_df.csv", parse_dates = ['Date'], dtype = {'Production':np.float64, 'Consumption':np.float64})
temp_df = pd.DataFrame(pd.date_range(start = pd.to_datetime('1998-01-01'), end = pd.to_datetime('2019-12-31'), freq = 'MS'), columns = ['Date'])
temp_df = pd.merge(temp_df, demand_df, how = 'left', on = 'Date')
temp_df['Date'] = temp_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
demand_df = temp_df.fillna(method = 'ffill')
demand_df.tail()

# Copper Concentrate prices

concen_df = pd.read_csv("../data/cu_concentrate_df.csv", parse_dates = ['Date'], dtype = {'CU_Concentrate_TC':np.float64,'CU_Concentrate_RC':np.float64})
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
pred = concatenate((scaled_data[-20:,1:], pred), axis = 1)
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

test.to_csv('../output/multi-seq-lstm-output_18Jan19_v2.0.csv', index = False)
        
## prediction - Multistep into Future
#
#newModel = Sequential()
#newModel.add(LSTM(50, batch_input_shape = (1, None, test_X_reshape.shape[2]), 
#                  return_sequences = False, stateful = True))
#newModel.add(Dense(1))
#newModel.set_weights(model.get_weights())
#
#step255 = newModel.predict(test_X_reshape).reshape(1, 1)
#temp = concatenate((test_X[-1:,1:], step255), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step256 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step256), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step257 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step257), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step258 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step258), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step259 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step259), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step260 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step260), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step261 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step261), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step262 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step262), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step263 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step263), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step264 = newModel.predict(temp).reshape(1, 1)
#
#temp = concatenate((test_X[-1:,1:], step264), axis = 1)
#temp = temp.reshape(1, 1, temp.shape[1])
#step265 = newModel.predict(temp).reshape(1, 1)
#
#
## merge the prediction with prediction sequence 
#
## week - 1
#step255 = concatenate((test_X[-1:,:], step255), axis = 1)
#step255 = scalar.inverse_transform(step255).tolist()
#
## week - 2
#step256 = concatenate((test_X[-1:,:], step256), axis = 1)
#step256 = scalar.inverse_transform(step256).tolist()
#
#
## week - 3
#step257 = concatenate((test_X[-1:,:], step257), axis = 1)
#step257 = scalar.inverse_transform(step257).tolist()
#
## week-4
#step258 = concatenate((test_X[-1:,:], step258), axis = 1)
#step258 = scalar.inverse_transform(step258).tolist()
#
## week - 5
#step259 = concatenate((test_X[-1:,:], step259), axis = 1)
#step259 = scalar.inverse_transform(step259).tolist()
#
## week - 6
#step260 = concatenate((test_X[-1:,:], step260), axis = 1)
#step260 = scalar.inverse_transform(step260).tolist()
#
## week - 7
#step261 = concatenate((test_X[-1:,:], step261), axis = 1)
#step261 = scalar.inverse_transform(step261).tolist()
#
## week - 8
#step262 = concatenate((test_X[-1:,:], step262), axis = 1)
#step262 = scalar.inverse_transform(step262).tolist()
#
## week - 9
#step263 = concatenate((test_X[-1:,:], step263), axis = 1)
#step263 = scalar.inverse_transform(step263).tolist()
#
## week - 10
#step264 = concatenate((test_X[-1:,:], step264), axis = 1)
#step264 = scalar.inverse_transform(step264).tolist()
#
## week - 11
#step265 = concatenate((test_X[-1:,:], step265), axis = 1)
#step265 = scalar.inverse_transform(step265).tolist()
#
#print("Week - 1 : %3.f" % step255[0][-1])
#print("Week - 2 : %3.f" % step256[0][-1])
#print("Week - 3 : %3.f" % step257[0][-1])
#print("Week - 4 : %3.f" % step258[0][-1])
#print("Week - 5 : %3.f" % step259[0][-1])
#print("Week - 6 : %3.f" % step260[0][-1])
#print("Week - 7 : %3.f" % step261[0][-1])
#print("Week - 8 : %3.f" % step262[0][-1])
#print("Week - 9 : %3.f" % step263[0][-1])
#print("Week - 10 : %3.f" % step264[0][-1])
#print("Week - 11 : %3.f" % step265[0][-1])
#
#
## model scoped to predict on all test cases
## model remains the same as in lines from 130 to 136
#
#newModel = Sequential()
#newModel.add(LSTM(50, batch_input_shape = (1, None, test_X_reshape.shape[2]),
#                  return_sequences = True))
#newModel.add(Dense(1))
#newModel.set_weights(model.get_weights())
#pred = newModel.predict(test_X_reshape)
#
## merge predictions onto test 
#pred = pred.reshape(pred.shape[1], pred.shape[2])
#test_pred_df = concatenate((test_X[:,:], pred), axis = 1)
#test_pred_df_inv = scalar.inverse_transform(test_pred_df)
#df = pd.DataFrame(test_pred_df_inv)
#df.iloc[:,[0,-1]].tail()
#
## write to disk
#df.to_csv("./output/prediction_test_cases_10.csv", index = False)
#
## Plot results 
#
#train_X_reshape = train_X.reshape(train_X.shape[1], train_X.shape[2])
#train_y_reshape = train_y.reshape(train_y.shape[1], train_y.shape[2])
#train_df = concatenate((train_X_reshape, train_y_reshape), axis = 1)
#train_df = pd.DataFrame(scalar.inverse_transform(train_df))
#
#tr_a = pd.concat((train_df.iloc[:,22], df.iloc[:,[0, 22]]), axis = 0)
#tr_a.reset_index(inplace = True)
#tr_a.drop('index', axis =1, inplace = True)
#plt.plot(tr_a.iloc[:,0])
#plt.plot(tr_a.iloc[:,[1]])



