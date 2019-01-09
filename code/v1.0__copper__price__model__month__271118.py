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

# data preparation 
# 1. Read Copper LME prices 

print("Current Working Directory is %s" % os.getcwd())
main_df = pd.read_excel("../data/Copper_LME_MCK.xlsx", sheet_name = 'Copper_LME_Price_History', skiprows = 1)
main_df.drop(['3-Month', '15-Month', '27-Month'], axis = 1, inplace = True)
main_df.sort_values('Date', ascending = False, inplace = True)
main_df['Date'] = main_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
main_df = main_df.groupby('Date').mean()

# train-test split

main_data = main_df.values
train = main_df.iloc[0:600,:]
test = main_df.iloc[600:,:]

# normalise data
scalar = MinMaxScaler(feature_range = (-1, 1))
scaled_data = scalar.fit_transform(main_data)

x_train, y_train = [],[]
for i in range(400, len(train)):
    x_train.append(scaled_data[i-400:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# lstm network

model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1, batch_size = 100, verbose = 2)

# prediction
inputs = main_df.iloc[len(main_df) - len(test) - 400:].values
inputs = inputs.reshape(-1, 1)
inputs = scalar.transform(inputs)

X_test = []
for i in range(400, inputs.shape[0]):
    X_test.append(inputs[i-400:i,0])    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scalar.inverse_transform(pred_price)

# prediction - Multistep into Future

newModel = Sequential()
newModel.add(LSTM(50, batch_input_shape = (X_test.shape[0], X_test.shape[1], X_test.shape[2]), 
                  return_sequences = True, stateful = True))
newModel.add(LSTM(50, return_sequences = False))
newModel.add(Dense(1))
newModel.set_weights(model.get_weights())

step_1 = newModel.predict(X_test, batch_size = 100)





# Plot

train = main_df.iloc[:600]
test = main_df[600:]
test['predictions'] = pred_price
plt.plot(train['Spot'])
plt.plot(test[['Spot', 'predictions']])


