#Main_app.py
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from keras.models import load_model
from datetime import date
from util import *
offset=10
today=date.today()




start = str(today.replace(year=today.year - offset))

try:

    model=load_model("Stock_price.keras")
except:
    pass

st.header("Stock Price Prediction")

ticker=st.text_input(
    placeholder="Enter Ticker Symbol",
    label="stock ticker",
    max_chars=5
    )
ticker=ticker.upper()
print(ticker)
if st.button("Submit"):
    
        data=yf.download(ticker,start,today)
        if(not(data.empty)):
            st.subheader(ticker)
           
            data.dropna(inplace=True)
            st.subheader("Previous Ten Years prices")
            st.write(data)
            figs=display_ma(data)
            data_train=pd.DataFrame(data.Close[:])
            data_train_scaled=scaling(data_train)
            x,y=feature_extraction(data_train_scaled)
           
            
            

            y_pred=model.predict(x[-100:])
            x=rescaling(x[-100:])
            
            
            
            
            for i in figs:
                st.write(i)
            days_to_predict = 365
            recent_data = data_train_scaled[-100:]  # Use the most recent 100 days of data
            future_prices = predict_future_prices(model, recent_data, days_to_predict)
            fig,p=plt.subplots()
            p.plot(data.index, data['Close'], label='Historical Prices')
            future_dates = pd.date_range(start=data.index[-1], periods=days_to_predict )
            
            p.plot(future_dates, future_prices, label='Predicted Prices', linestyle='--', color='r')
            p.legend()
            
            
            st.write(fig)
            if st.button("Recaliberate Model"):
                Recalberate(x,y,model)
            
            
        
        else:
           st.write("Invalid Stock Ticker")
        
