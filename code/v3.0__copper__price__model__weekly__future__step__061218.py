# copper price forecast using LSTM 
# setting up necessary libraries 
# * this source code tested on tfp3.6 env with anaconda 

import pandas as pd
import os
import numpy as np
from numpy import concatenate, array
import matplotlib.pyplot as plt
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.utils import plot_model

# python functions
# input sequence 
def series_to_supervised(data, n_in, n_out, dropnan = True):
    names, cols = [], []
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += ["var%d(t-%d)" % (j+1, i) for j in range(n_vars)]
# forecast sequence
    for i in range(0, n_out):
        cols.append(df[0].shift(-i))
        if i == 0:
            names += ["var%d(t)" % (j+1) for j in range(1)] 
        else:
            names += ["var%d(t+%d)" % (j+1, i) for j in range(1)]
    agg = concat(cols, axis = 1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace = True)
    return agg

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
temp_df = pd.DataFrame(pd.date_range(start = pd.to_datetime('2012-01-01'), end = pd.to_datetime('2018-11-30'), freq = 'W'), columns = ['Date'])
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

# convert series to supervised 
# name Date as index 

main_df.index = main_df['Date']
main_df.drop('Date', axis = 1, inplace = True)
values = main_df.values

# normalize features

scalar = MinMaxScaler(feature_range = (-1, 1))
scaled_data = scalar.fit_transform(values)
reframed_data = series_to_supervised(scaled_data, 1, 1)
reframed_data.tail()

# train-test split
# convert sample into 3D format for LSTM 

values = reframed_data.values
train = values[0:values.shape[0]-1, :]
test = values[values.shape[0]-1:,:]
train_X, train_y = train[:, :-1], train[:,-1:]
test_X, test_y = test[:, :-1], test[:,-1:]
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
print(train_X.shape, train_y.shape, test_X.shape)

## lstm network

model = Sequential()
model.add(LSTM(50, batch_input_shape=(1, train_X.shape[1], train_X.shape[2]), stateful = False))
model.add(Dense(train_y.shape[1]))
model.compile(loss = 'mae', optimizer = 'adam')
fit_model = model.fit(train_X, train_y, epochs = 50, batch_size = 1, verbose = 0, shuffle = False)

# forecast on test dataset

pred = model.predict(test_X)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1])
test_X_df = concatenate((test_X[:,0:19], pred), axis = 1)
inv_test_X_df = scalar.inverse_transform(test_X_df)




# forecast at one point 

def forecast_lstm(model, x, batch_size = 1):
    x = x.reshape(1, 1, len(x))
    forecast = model.predict(x, batch_size)
    return [x for x in forecast[0,:]]

# convert forecast to array and invert transform

def make_forecasts(model, test, batch_size =1):
    forecasts = list()
    for i in range(len(test)):
        x, y = test[i,:-3], test[i,-3:]
        forecast = forecast_lstm(model, x, batch_size = 1)
        forecasts.append(forecast)
    return forecasts

forecasts = make_forecasts(model, test, batch_size = 1)
        
for i in range(len(forecasts)):
    print(forecasts[i])
    forecast = array(forecasts[i])
    print(len(forecast))
    forecast = forecast.reshape((1, len(forecast)))
    print(forecast.shape)
    inv_scale = scalar.inverse_transform(forecast)
    print(inv_scale)
        
        
        

    







## print train and test accuracy
#plt.plot(fit_model.history['acc'], label = 'train')
#plt.plot(fit_model.history['val_acc'], label = 'test')
#plt.title("Model Accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("Epoch")
#plt.legend()
#plt.show()

# Evlauate model 
# Prediction on test series
pred_test = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_test_X = concatenate((pred_test, test_X[:,1:]), axis = 1)
inv_test_X = scalar.inverse_transform(inv_test_X)

#Actual test series 
test_y = test_y.reshape(len(test_y), 1)
inv_test_y = concatenate((test_y, test_X[:,1:]), axis = 1)
inv_test_y = scalar.inverse_transform(inv_test_y)
inv_test_y = inv_test_y[:,0]

# naming columns

df_cols = list(main_df.columns)
df_cols.remove('copper_price')
df_cols.insert(0, "copper_pred")
test_df = pd.DataFrame(inv_test_X, columns = df_cols)
test_y = pd.DataFrame(inv_test_y, columns = ['copper_actual'])
output_df = test_df.merge(test_y, left_index = True, right_index = True)
#output_df.to_csv("./output/lstm_results_named_output_batch_100.csv", index = False)

# Calcuate RMSE
mae = mean_absolute_error(inv_test_X[:,0], inv_test_y)
print('Test MAE is %.3f' % mae)


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

# test

#df = main_df['copper_price'].tail(10)
#cols, names = [],[]
#for i in range(0,3):
#    cols.append(df.shift(-i))
#    if i ==0:
#        names += ["var%d(t)" %(j+1) for j in range(1)]
#    else:
#        names += ["var%d(t+%d)" %(j+1, i) for j in range(1)]
#    agg = concat(cols, axis = 1)
#    agg.columns = names
    
#names = []
#cols = []
#for i in range(0, 3):
#    cols.append(s[0].shift(-i))
#    if i ==0:
#        names += ["var%d(t)" %(j+1) for j in range(1)] 
#    else: 
#        names += ["var%d(t+%d)" %(j+1, i) for j in range(1)]
#    agg = concat(cols, axis = 1)


