import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BrentOilLSTM:
    def __init__(self, file_path, lookback=30):
        """Initialize LSTM model with dataset path and lookback window"""
        self.file_path = file_path
        self.lookback = lookback
        self.df = None
        self.scaler = MinMaxScaler()
        self.model = None

    def load_and_preprocess_data(self):
        """Load dataset, normalize, and prepare sequences"""
        self.df = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')
        self.df = self.df.sort_index()
        
        # Normalize data
        self.df['Price'] = self.scaler.fit_transform(self.df[['Price']])

        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(self.df)):
            X.append(self.df['Price'].values[i-self.lookback:i])
            y.append(self.df['Price'].values[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)

        print("✅ Data prepared for LSTM!")

    def build_lstm_model(self):
        """Build and compile the LSTM model"""
        self.model = Sequential([
            LSTM(15, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print("✅ LSTM model built!")

    def train_lstm(self, epochs=50, batch_size=32):
        """Train the LSTM model"""
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test), verbose=1)
        print("✅ LSTM model trained!")

    def evaluate_model(self):
        """Evaluate LSTM model performance"""
        predictions = self.model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        actuals = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        print(f"📊 LSTM Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")

    def forecast(self, steps=30):
        """Forecast future prices using LSTM"""
        input_seq = self.X_test[-1].reshape(1, self.lookback, 1)
        predictions = []
        
        for _ in range(steps):
            pred = self.model.predict(input_seq)
            predictions.append(pred[0][0])
            input_seq = np.roll(input_seq, -1)
            input_seq[0, -1, 0] = pred[0][0]

        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        print(f"📈 LSTM Forecast for next {steps} days:\n", predictions.flatten())
        return predictions.flatten()

# Example Usage
if __name__ == "__main__":
    lstm_model = BrentOilLSTM("/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/processed_BrentOilPrices.csv")

    # Run LSTM pipeline
    lstm_model.load_and_preprocess_data()
    lstm_model.build_lstm_model()
    lstm_model.train_lstm(epochs=50, batch_size=32)
    lstm_model.evaluate_model()
    lstm_forecast = lstm_model.forecast(steps=30)
