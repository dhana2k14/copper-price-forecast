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

# python functions

# read datasets and consolidate
print("Current Working Directory is %s" % os.getcwd())
main_df = pd.read_csv("./data/copper_df.csv", usecols = [0,3], parse_dates = ['Date'])
main_df['Date'] = main_df['Date'].apply(lambda x: x - pd.offsets.Week(weekday = 6))
main_df = main_df.groupby('Date').mean()
main_df.tail()
























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
model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2)

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

# Plot

train = main_df.iloc[:600]
test = main_df[600:]
test['predictions'] = pred_price
plt.plot(train['Spot'])
plt.plot(test[['Spot', 'predictions']])


