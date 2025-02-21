import unittest
import pandas as pd
from io import StringIO
import os
import sys
sys.path.append(os.path.abspath('..'))  
from  scripts.preprocess import BrentOilDataProcessor  

class TestBrentOilDataProcessor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Sample data for testing
        cls.test_data = StringIO("""Date,Price
        2022-01-01,80
        2022-01-02,82
        2022-01-03,81
        2022-01-04,83
        2022-01-05,85
        2022-01-01,80  # Duplicate row
        2022-01-06,NaN  # Missing value
        """)
        cls.file_path = 'test_brent_oil.csv'
        cls.df = pd.read_csv(cls.test_data, parse_dates=['Date'])
        cls.df.to_csv(cls.file_path, index=False)  # Save to a temporary CSV for testing

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.file_path)  # Clean up the test CSV file

    def setUp(self):
        self.processor = BrentOilDataProcessor(self.file_path)
        self.processor.load_data()

    def test_load_data(self):
        """Test if data is loaded correctly."""
        self.assertIsNotNone(self.processor.df)
        self.assertEqual(len(self.processor.df), 7)  # Check number of rows including duplicates and NaN

    def test_preprocess_data(self):
        """Test data preprocessing: handling duplicates and missing values."""
        self.processor.preprocess_data()
        self.assertEqual(len(self.processor.df), 5)  # After preprocessing, should have 5 rows
        self.assertFalse(self.processor.df.isnull().values.any())  # Ensure there are no NaN values
        self.assertEqual(self.processor.df.duplicated().sum(), 0)  # Ensure no duplicates remain

    def test_data_summary(self):
        """Test data summary generation (no assertion, just to ensure it runs)."""
        self.processor.data_summary()  # This method generates output but does not return anything

    def test_get_clean_data(self):
        """Test if cleaned data is saved correctly."""
        cleaned_df = self.processor.get_clean_data()
        self.assertIsNotNone(cleaned_df)
        self.assertEqual(len(cleaned_df), 5)  # Should still have 5 rows after cleaning

if __name__ == '__main__':
    unittest.main()
