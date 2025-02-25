import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class BrentOilFeatureEngineering:
    def __init__(self, file_path, output_path="processed_BrentOilPrices.csv"):
        """Initialize with dataset file path and output path"""
        self.file_path = file_path
        self.output_path = output_path
        self.df = None
        self.features = None
        self.target = None

    def load_data(self):
        """Load dataset from CSV"""
        self.df = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')
        self.df = self.df.sort_index()  # Ensure data is sorted by date
        print("✅ Data loaded successfully!")
        return self.df

    def create_datetime_features(self):
        """Extract numeric datetime features for ML models"""
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['day'] = self.df.index.day
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['days_since_start'] = (self.df.index - self.df.index.min()).days
        print("✅ Datetime features created!")

    def create_lag_features(self, lags=[1, 7, 30]):
        """Generate lag features to capture past trends"""
        for lag in lags:
            self.df[f'lag_{lag}'] = self.df['Price'].shift(lag)
        print(f"✅ Lag features {lags} created!")

    def create_moving_averages(self, windows=[7, 30, 90]):
        """Generate moving average features"""
        for window in windows:
            self.df[f'ma_{window}'] = self.df['Price'].rolling(window=window).mean()
        print(f"✅ Moving averages {windows} created!")

    def create_volatility_features(self, windows=[7, 30]):
        """Generate rolling standard deviation (volatility) features"""
        for window in windows:
            self.df[f'volatility_{window}'] = self.df['Price'].rolling(window=window).std()
        print(f"✅ Volatility features {windows} created!")

    def handle_missing_values(self):
        """Fill missing values caused by shifting and rolling operations"""
        self.df.dropna(inplace=True)
        print("✅ Missing values handled!")

    def prepare_features_and_target(self):
        """Prepare the feature matrix (X) and target variable (y)"""
        self.features = self.df.drop(columns=['Price'])  # All columns except 'Price' are features
        self.target = self.df['Price']  # Target variable
        print("✅ Features and target prepared!")

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, 
                                                            test_size=test_size, 
                                                            random_state=random_state,
                                                            shuffle=False)  # Keep time series order
        print(f"✅ Data split into training ({100 - test_size*100}%) and testing ({test_size*100}%) sets.")
        return X_train, X_test, y_train, y_test

    def save_processed_data(self):
        """Save processed dataset to CSV"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)  # Ensure directory exists
        self.df.to_csv(self.output_path)
        print(f"✅ Processed data saved to {self.output_path}")

    def get_processed_data(self):
        """Return the processed dataframe"""
        return self.df

# Example Usage
if __name__ == "__main__":
    feature_engineer = BrentOilFeatureEngineering(
        "/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/cleaned_BrentOilPrices.csv",
        "/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/processed_BrentOilPrices_ML.csv"
    )
    
    # Run feature engineering pipeline
    feature_engineer.load_data()
    feature_engineer.create_datetime_features()
    feature_engineer.create_lag_features()
    feature_engineer.create_moving_averages()
    feature_engineer.create_volatility_features()
    feature_engineer.handle_missing_values()
    feature_engineer.prepare_features_and_target()
    
    # Save processed dataset
    feature_engineer.save_processed_data()
    
    # Split data
    X_train, X_test, y_train, y_test = feature_engineer.split_data()

    # Get the final processed data
    processed_df = feature_engineer.get_processed_data()
    print(processed_df.head())
