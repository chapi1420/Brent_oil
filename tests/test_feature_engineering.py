import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.append(''))
from scripts.feature_engineering import BrentOilFeatureEngineering 

class TestBrentOilFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a temporary CSV file for testing"""
        cls.test_file = 'test_BrentOilPrices.csv'
        cls.output_file = 'processed_test_BrentOilPrices.csv'
        cls.data = {
            'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'Price': pd.Series(range(100)) + (pd.Series(range(100)) * 0.1).apply(lambda x: x * (1 if x % 10 == 0 else 0))  # Add some variation
        }
        df = pd.DataFrame(cls.data)
        df.to_csv(cls.test_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Remove temporary files after tests"""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)
        if os.path.exists(cls.output_file):
            os.remove(cls.output_file)

    def setUp(self):
        """Create an instance of the feature engineering class for each test"""
        self.feature_engineer = BrentOilFeatureEngineering(self.test_file, self.output_file)

    def test_load_data(self):
        """Test loading data"""
        df = self.feature_engineer.load_data()
        self.assertEqual(len(df), 100)
        self.assertIn('Price', df.columns)

    def test_create_lag_features(self):
        """Test lag feature creation"""
        self.feature_engineer.load_data()
        self.feature_engineer.create_lag_features(lags=[1, 7])
        self.assertIn('lag_1', self.feature_engineer.df.columns)
        self.assertIn('lag_7', self.feature_engineer.df.columns)

    def test_create_moving_averages(self):
        """Test moving average feature creation"""
        self.feature_engineer.load_data()
        self.feature_engineer.create_moving_averages(windows=[7, 30])
        self.assertIn('ma_7', self.feature_engineer.df.columns)
        self.assertIn('ma_30', self.feature_engineer.df.columns)

    def test_create_volatility_features(self):
        """Test volatility feature creation"""
        self.feature_engineer.load_data()
        self.feature_engineer.create_volatility_features(windows=[7])
        self.assertIn('volatility_7', self.feature_engineer.df.columns)

    def test_handle_missing_values(self):
        """Test handling missing values"""
        self.feature_engineer.load_data()
        self.feature_engineer.create_lag_features(lags=[1, 7])
        self.feature_engineer.handle_missing_values()
        self.assertFalse(self.feature_engineer.df.isnull().values.any())

    def test_prepare_features_and_target(self):
        """Test preparing features and target"""
        self.feature_engineer.load_data()
        self.feature_engineer.create_lag_features()
        self.feature_engineer.prepare_features_and_target()
        self.assertIsNotNone(self.feature_engineer.features)
        self.assertIsNotNone(self.feature_engineer.target)

    def test_split_data(self):
        """Test data splitting"""
        self.feature_engineer.load_data()
        self.feature_engineer.create_lag_features()
        self.feature_engineer.prepare_features_and_target()
        X_train, X_test, y_train, y_test = self.feature_engineer.split_data()
        self.assertEqual(X_train.shape[0], 80)  # 80% of 100
        self.assertEqual(X_test.shape[0], 20)   # 20% of 100

    def test_save_processed_data(self):
        """Test saving processed data"""
        self.feature_engineer.load_data()
        self.feature_engineer.save_processed_data()
        self.assertTrue(os.path.exists(self.output_file))

    def test_get_processed_data(self):
        """Test getting processed data"""
        self.feature_engineer.load_data()
        self.feature_engineer.create_lag_features()
        processed_data = self.feature_engineer.get_processed_data()
        self.assertIsNotNone(processed_data)

if __name__ == '__main__':
    unittest.main()
