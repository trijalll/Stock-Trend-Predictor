import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime
import yfinance as yf
import datetime
import streamlit as st


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2022, 12, 31)

st.title('Stock Trend Prediction')

user_input =st.text_input ('Enter the Stock Ticker','AAPL')
# Create a Ticker object for AAPL
ticker = yf.Ticker(user_input)

# Get historical data
df = ticker.history(start=start, end=end)

#describing data
st.subheader('Data From 2019-2022')
st.write(df.describe())

#visualization
st.subheader('Closing Price VS Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close,'y')
st.pyplot(fig)

#plotting moving average(100ma)
st.subheader('Closing Price VS Time chart with 100 Moving Avg')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'y')
st.pyplot(fig)


#200 days ma
st.subheader('Closing Price VS Time chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'k')
plt.plot(df.Close,'y')
st.pyplot(fig)


#splitting data into training and testing 

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)


#scalling down the data b/w 0 and 1 for that we use min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)




#load my model
from keras.models import load_model
model = load_model('keras_model.h5')
#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


#plotting the predicted and orignal values
st.subheader('Predictions VS Orignal')
fig2 = plt.figure(figsize=(10,5))
plt.plot(y_test,'k',label = 'Orignal Price')
plt.plot(y_predicted,'y',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
#below graph shows the predicted Trend

#streamlit run app.py