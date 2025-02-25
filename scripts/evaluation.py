import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

class BrentOilARIMA:
    def __init__(self, file_path, order=(5, 1, 0)):
        """Initialize ARIMA Model with dataset path and ARIMA order (p,d,q)"""
        self.file_path = file_path
        self.df = None
        self.model = None
        self.results = None
        self.order = order

    def load_data(self):
        """Load dataset from CSV"""
        self.df = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')
        self.df = self.df.sort_index()

        # Check for missing dates
        full_date_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='D')
        missing_dates = full_date_range.difference(self.df.index)

        if not missing_dates.empty:
            print("⚠️ Missing dates found:", missing_dates)

        # Reindex the DataFrame to include all dates
        self.df = self.df.reindex(full_date_range)

        # Fill missing values (if needed)
        self.df['Price'] = self.df['Price'].fillna(method='ffill')  # Forward fill as an example

        # Set frequency only if the index is complete
        if self.df.index.is_unique and len(self.df.index) == len(full_date_range):
            self.df.index.freq = 'D'  # Set frequency to daily
        else:
            print("⚠️ Unable to set frequency due to non-unique or incomplete index.")

        print("✅ Data loaded for ARIMA!")
        return self.df


    def train_arima(self):
        """Train the ARIMA model"""
        self.model = sm.tsa.ARIMA(self.df['Price'], order=self.order)
        self.results = self.model.fit()
        print("✅ ARIMA model trained successfully!")

    def evaluate_model(self):
        """Evaluate ARIMA model performance"""
        predictions = self.results.fittedvalues
        mae = mean_absolute_error(self.df['Price'][1:], predictions[1:])
        rmse = np.sqrt(root_mean_squared_error(self.df['Price'][1:], predictions[1:]))
        print(f"📊 ARIMA Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")

    def forecast(self, steps=30):
        """Forecast future prices using ARIMA"""
        forecast_values = self.results.forecast(steps=steps)
        print(f"📈 ARIMA Forecast for next {steps} days:\n", forecast_values)
        return forecast_values

class BrentOilLSTM:
    def __init__(self, file_path, lookback=30):
        """Initialize LSTM model with dataset path and lookback window"""
        self.file_path = file_path
        self.lookback = lookback
        self.df = None
        self.scaler = MinMaxScaler()  # Initialize MinMaxScaler
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_and_preprocess_data(self):
        """Load and preprocess dataset"""
        self.df = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')
        self.df = self.df.sort_index()

        # Create a complete date range
        full_date_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='D')

        # Check for missing dates
        missing_dates = full_date_range.difference(self.df.index)

        if not missing_dates.empty:
            print("⚠️ Missing dates found:", missing_dates)

        # Reindex to include all dates
        self.df = self.df.reindex(full_date_range)

        # Fill missing values (using ffill directly)
        self.df['Price'] = self.df['Price'].ffill()  

        # Check for duplicates
        if self.df.index.duplicated().any():
            print("⚠️ Duplicate dates found in the index.")
            self.df = self.df[~self.df.index.duplicated(keep='first')]  # Remove duplicates

        # Set frequency only if the index is unique and complete
        if self.df.index.is_unique and len(self.df.index) == len(full_date_range):
            self.df.index.freq = 'D'  
        else:
            print("⚠️ Unable to set frequency due to non-unique or incomplete index.")

        print("✅ Data loaded and preprocessed for LSTM!")
        return self.df

    def prepare_data(self):
        """Prepare training and testing datasets"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.df['Price'].values.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # Split into training and testing sets (80% train, 20% test)
        split = int(0.8 * len(X))
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

        print(f"✅ Data prepared: {len(self.X_train)} training samples and {len(self.X_test)} testing samples.")
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")

    def build_lstm_model(self):
        """Build and compile the LSTM model"""
        self.model = Sequential()
        self.model.add(Input(shape=(self.lookback, 1)))  # Use Input layer
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print("✅ LSTM model built!")

    def train_lstm(self, epochs=50, batch_size=32):
        """Train the LSTM model"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data has not been prepared. Please call prepare_data() first.")

        self.model.fit(self.X_train.reshape(-1, self.lookback, 1), self.y_train, 
                       epochs=epochs, batch_size=batch_size, 
                       validation_data=(self.X_test.reshape(-1, self.lookback, 1), self.y_test), 
                       verbose=1)
        print("✅ LSTM model trained!")
    def evaluate_model(self):
        """Evaluate the trained LSTM model."""
        # Make predictions on the test set
        predicted_prices = self.model.predict(self.X_test.reshape(-1, self.lookback, 1))

        # Inverse transform to get actual prices
        predicted_prices = self.scaler.inverse_transform(predicted_prices)
        actual_prices = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        # Calculate evaluation metrics
        rmse = root_mean_squared_error(actual_prices, predicted_prices)
        mae = mean_absolute_error(actual_prices, predicted_prices)

        print(f"✅ Evaluation Metrics:\n - RMSE: {rmse}\n - MAE: {mae}")

        # Plot the results
        plt.figure(figsize=(14, 7))
        plt.plot(actual_prices, color='blue', label='Actual Prices')
        plt.plot(predicted_prices, color='red', label='Predicted Prices')
        plt.title('Brent Oil Prices: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    def forecast(self, steps=30):
        """Forecast future prices using LSTM"""
        if self.X_test.shape[0] == 0:
            raise ValueError("❌ Error: X_test is empty. Check your data preprocessing.")

        input_seq = np.array(self.X_test[-1]).reshape(1, self.lookback, 1).astype(np.float32)

        predictions = []

        for _ in range(steps):
            print(f"Input sequence shape: {input_seq.shape}")  # Debugging line

            pred = self.model.predict(input_seq, verbose=0)

            pred_value = float(pred[0, 0])
            predictions.append(pred_value)

            # Shift input sequence and append the new prediction
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1, 0] = pred_value  # Replace last value with prediction

        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        print(f"📈 LSTM Forecast for next {steps} days:\n", predictions.flatten())
        return predictions.flatten()
# Example Usage
if __name__ == "__main__":
    # ARIMA Model
    arima_model = BrentOilARIMA("/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/processed_BrentOilPrices.csv")
    
    # Run ARIMA pipeline
    arima_model.load_data()
    arima_model.train_arima()
    arima_model.evaluate_model()
    arima_forecast = arima_model.forecast(steps=30)

    # LSTM Model
    lstm_model = BrentOilLSTM("/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/processed_BrentOilPrices_ML.csv")
    
    # Run LSTM pipeline
    lstm_model.load_and_preprocess_data()
    lstm_model.prepare_data()
    lstm_model.build_lstm_model()
    lstm_model.train_lstm(epochs=50, batch_size=32)
    lstm_model.evaluate_model()
    lstm_forecast = lstm_model.forecast(steps=30)

    # Compare forecasts
    print("📊 Comparison of Forecasts:")
    print("ARIMA Forecast:\n", arima_forecast)
    print("LSTM Forecast:\n", lstm_forecast)
    print("✅ Evaluation complete!")