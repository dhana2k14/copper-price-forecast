# copper price forecast using LSTM 
# setting up necessary libraries 
# * this source code tested on tfp3.6 env with anaconda 

import pandas as pd

# data preparation 
# 1. Read Copper LME prices 

main_df = pd.read_excel("../data/Copper_LME_MCK.xlsx", sheet_name = 'Copper_LME_Price_History', skiprows = 1)
main_df.drop(['3-Month', '15-Month', '27-Month'], axis = 1, inplace = True)
main_df.head()


