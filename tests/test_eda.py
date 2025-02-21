import os 
import sys
import unittest
import pandas as pd
from io import StringIO
sys.path.append(os.path.abspath(".."))

from scripts.eda import BrentOilEDA  

class TestBrentOilEDA(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_data = StringIO("""Date,Price
        2022-01-01,80
        2022-01-02,82
        2022-01-03,81
        2022-01-04,83
        2022-01-05,85
        """)
        cls.file_path = 'test_brent_oil.csv'
        cls.df = pd.read_csv(cls.test_data, parse_dates=['Date'], index_col='Date')
        cls.df.to_csv(cls.file_path)  # Save to a temporary CSV for testing

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.file_path)  

    def setUp(self):
        self.eda = BrentOilEDA(self.file_path)
        self.eda.load_data()

    def test_load_data(self):
        """Test if data is loaded correctly."""
        self.assertIsNotNone(self.eda.df)
        self.assertEqual(len(self.eda.df), 5)  

    def test_monthly_average_price(self):
        """Test monthly average price calculation."""
        self.eda.monthly_average_price()  
        monthly_avg = self.eda.df.groupby([self.eda.df.index.year, self.eda.df.index.month])['Price'].mean()
        self.assertEqual(monthly_avg.iloc[0], 80.0) 

    def test_price_volatility(self):
        """Test price volatility calculation."""
        self.eda.price_volatility()
        self.assertIn('Rolling_Std', self.eda.df.columns)  # Check if the rolling std column is added
        self.assertFalse(self.eda.df['Rolling_Std'].isnull().all())  # Ensure there's no NaN in the rolling std

    def test_price_distribution(self):
        """Test price distribution visualization (no assertion, just to ensure it runs)."""
        self.eda.price_distribution()  # This method generates a plot

    def test_detect_change_points(self):
        """Test change point detection functionality."""
        self.eda.detect_change_points()  # This method generates a plot

    def test_monthly_trends_heatmap(self):
        """Test heatmap generation (no assertion, just to ensure it runs)."""
        self.eda.monthly_trends_heatmap()  # This method generates a plot

    def test_compare_before_after_events(self):
        """Test comparison of prices around events (no assertion, just to ensure it runs)."""
        self.eda.compare_before_after_events()  # This method generates a plot

if __name__ == '__main__':
    unittest.main()
