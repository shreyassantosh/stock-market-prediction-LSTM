#util.py
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import  MinMaxScaler
def predict_future_prices(model, recent_data, days_to_predict):
    future_predictions = []
    current_data = recent_data

    for i in range(days_to_predict):
        x_input = current_data[-100:].reshape(1, 100, 1)
        predicted_price_scaled = model.predict(x_input)
        predicted_price = rescaling(predicted_price_scaled)
        future_predictions.append(predicted_price[0][0])
        
        # Append the predicted price to the current data
        current_data = np.append(current_data, predicted_price_scaled)
    
    return future_predictions
def scaling(data_train):
    global mmscaler
    mmscaler=MinMaxScaler(feature_range=(0,1))
    
    data_train_scaled=mmscaler.fit_transform(data_train)
    
    return data_train_scaled
def rescaling(y_pred):
   
    scale=1/mmscaler.scale_

    y_pred=y_pred*scale
    return y_pred

def compile_model():
    model=Sequential()
    model.add(LSTM(units=50,activation='relu',return_sequences=True , input_shape=((x.shape[1],1))))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60,activation='relu',return_sequences=True ))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80,activation='relu',return_sequences=True ))
    model.add(Dropout(0.4))

    model.add(LSTM(units=120,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')

    
    model.save("Stock_price.keras")
    
def Recaliberate(x,y,model):
    model.fit(x,y,epochs=50,batch_size=32,verbose=1)
    model.save("Stock_price.keras")



def feature_extraction(data_scaled):
    x=[]
    y=[]
    for i in range(100,data_scaled.shape[0]):
        x.append(data_scaled[i-100:i])
        y.append(data_scaled[i,0])
    x,y=np.array(x),np.array(y)
    return x,y







def display_ma(data):
    ma_50days=data.Close.rolling(50).mean()
    
    ma_100days=data.Close.rolling(100).mean()
    ma_200days=data.Close.rolling(200).mean()
    fig1,p1=plt.subplots()
    fig2,p2=plt.subplots()
    fig3,p3=plt.subplots()
    p1.plot(data.Close,'g',label="Closing price")
    p1.plot(ma_50days,'r',label="50 day MA")
    p1.legend()
    
    p2.plot(data.Close,'g',label="Closing price")
    p2.plot(ma_100days,'r',label="100 day MA")
    p2.legend()

    p3.plot(data.Close,'g',label="Closing price")
    p3.plot(ma_200days,'r',label="200 day MA")
    p3.legend()
    
    
    
    
    
    
    
    
    
    return fig1,fig2,fig3




# In[15]:

##
####plt.plot(ma_100days,'r')
####plt.plot(data.Close,'g')
####plt.plot(ma_200days,'b')
####plt.show()
##
##
### In[16]:
##
##
###preprocesseing
##data.dropna(inplace=True)
##
##
### In[17]:


#test train split
##data_train=pd.DataFrame(data.Close[0:int(len(data)*0.8)])
##data_test=pd.DataFrame(data.Close[int(len(data)*0.8):])
##

# In[18]:

##
##data_train.shape
##
##
### In[20]:
##
##
##data_test.shape
##
##
### In[23]:
##
##
###Scaling
##from sklearn.preprocessing import  MinMaxScaler
##mmscaler=MinMaxScaler(feature_range=(0,1))
##
##data_train_scaled=mmscaler.fit_transform(data_train)
##data_test_scaled=mmscaler.transform(data_test)
##
##
### In[49]:
##







# In[52]:


#
##x_train,y_train=feature_extraction(data_train_scaled)
##x_test,y_test=feature_extraction(data_test_scaled)
##train(x_train,y_train)
##
##
####
##y_pred=model.predict(x_test)
##
###re-scaling
##scale=1/mmscaler.scale_
##y_test=y_test*scale
##y_pred=y_pred*scale
##
##
##plt.plot(y_pred,'r',label="predicted")
##plt.plot(y_test,'g',label="actual")
##plt.legend()
##plt.show()
##
##
##
##
##
##
##
##
