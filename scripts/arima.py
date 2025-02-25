import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BrentOilARIMA:
    def __init__(self, file_path, order=(5,1,0)):
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
        rmse = np.sqrt(mean_squared_error(self.df['Price'][1:], predictions[1:]))
        print(f"📊 ARIMA Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")

    def forecast(self, steps=30):
        """Forecast future prices using ARIMA"""
        forecast_values = self.results.forecast(steps=steps)
        print(f"📈 ARIMA Forecast for next {steps} days:\n", forecast_values)
        return forecast_values

# Example Usage
if __name__ == "__main__":
    arima_model = BrentOilARIMA("/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/processed_BrentOilPrices.csv")
    
    # Run ARIMA pipeline
    arima_model.load_data()
    arima_model.train_arima()
    arima_model.evaluate_model()
    arima_forecast = arima_model.forecast(steps=30)
