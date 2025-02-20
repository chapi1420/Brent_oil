import pandas as pd

class BrentOilDataProcessor:
    def __init__(self, file_path):
        """Initialize with file path and load data."""
        self.file_path = file_path
        self.df = None  

    def load_data(self):
        """Load dataset from CSV file."""
        try:
            self.df = pd.read_csv(self.file_path)
            print("✅ Data loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
        return self.df

    def preprocess_data(self):
        """Perform data preprocessing: convert date column, remove duplicates, handle missing values."""
        if self.df is None:
            print("❌ No data loaded. Run load_data() first.")
            return

        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df.set_index('Date', inplace=True) 

        missing_values = self.df.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠️ Found {missing_values} missing values. Dropping missing values...")
            self.df.dropna(inplace=True)

        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            print(f"⚠️ Found {duplicate_count} duplicate rows. Removing duplicates...")
            self.df.drop_duplicates(inplace=True)

        print("✅ Data preprocessing complete!")
        return self.df

    def data_summary(self):
        """Generate summary statistics and basic insights."""
        if self.df is None:
            print("❌ No data loaded. Run load_data() first.")
            return

        print("\n🔍 Dataset Info:")
        print(self.df.info())

        print("\n📊 Summary Statistics:")
        print(self.df.describe())

        print("\n🗓 Date Range:")
        print(f"Start Date: {self.df.index.min()}, End Date: {self.df.index.max()}")

    def get_clean_data(self):
        """Return the cleaned dataset."""
        self.df.to_csv("/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/cleaned_BrentOilPrices.csv")
        return self.df

if __name__ == "__main__":
    processor = BrentOilDataProcessor("/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/BrentOilPrices.csv")
    processor.load_data()
    processor.preprocess_data()
    processor.data_summary()
