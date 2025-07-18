### Project Description: Stock Price Prediction using LSTM and Streamlit

#### Project Overview:
This project aims to predict future stock prices using historical stock data and advanced machine learning techniques. We leverage Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), which are particularly well-suited for time series forecasting tasks. The project is implemented in Python and uses Streamlit for an interactive web application interface, allowing users to input stock ticker symbols and visualize the predicted prices.

#### Key Components:

1. **Data Collection:**
   - The project utilizes `yfinance` to fetch historical stock data. Users can input the stock ticker symbol to retrieve data for the past ten years.

2. **Data Preprocessing:**
   - The historical stock data is preprocessed by handling missing values and scaling the data using MinMaxScaler to normalize the features, making them suitable for the LSTM model.

3. **Model Architecture:**
   - The model is built using TensorFlow's Keras API. It consists of several LSTM layers with dropout layers to prevent overfitting. The model is compiled with the Adam optimizer and mean squared error as the loss function.
   - The architecture involves:
     - Four LSTM layers with increasing units (50, 60, 80, 120) and ReLU activation.
     - Dropout layers (0.2 to 0.5) after each LSTM layer to mitigate overfitting.
     - A final Dense layer for the output.

4. **Feature Extraction:**
   - The preprocessed data is transformed into sequences of 100 days, which serve as input features for the model. The corresponding stock price of the next day is used as the target variable.

5. **Model Training:**
   - The model is trained on the prepared dataset for 50 epochs with a batch size of 32.

6. **Future Price Prediction:**
   - The model predicts stock prices for the next 365 days using the most recent 100 days of historical data.
   - Predictions are rescaled to the original scale using the inverse transform of MinMaxScaler.

7. **Visualization:**
   - The project includes functions to display historical closing prices along with 50-day, 100-day, and 200-day moving averages.
   - The Streamlit interface allows users to visualize the predicted stock prices along with historical data, providing a clear understanding of the model's performance and predictions.

8. **Streamlit Web Application:**
   - The user interface is built using Streamlit, providing an interactive platform where users can:
     - Input stock ticker symbols.
     - View historical stock prices and moving averages.
     - Trigger model recalibration (training) if desired.
     - Visualize future stock price predictions.

#### Usage Instructions:
1. **Run the Application:**
   - Ensure all required packages are installed (`numpy`, `pandas`, `yfinance`, `matplotlib`, `tensorflow`, `sklearn`, `streamlit`).
   - Execute the Streamlit application using the command:
     ```cmd
     streamlit run Main_app.py
     ```

2. **Interact with the Application:**
   - Enter the desired stock ticker symbol in the input field.
   - View the historical stock data and moving averages.
   - Click the "Submit" button to fetch and display the stock data.
   - Click "Recalibrate Model" if you wish to retrain the model on the fetched data.
   - Visualize the predicted stock prices for the next 365 days.

#### Conclusion:
This project demonstrates a practical application of LSTM networks for time series forecasting, specifically in predicting stock prices. The use of Streamlit enhances the user experience by providing an interactive interface to explore and visualize stock market data and predictions. This approach can be further extended and fine-tuned for more accurate predictions and broader financial applications.