import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import os

class BrentOilEDA:
    def __init__(self, file_path):
        """Initialize with the dataset file path."""
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load dataset from CSV and preprocess it."""
        if not os.path.exists(self.file_path):
            print(f"❌ File not found: {self.file_path}")
            return
        
        self.df = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')
        self.df.sort_index(inplace=True)  # Ensure data is sorted by date
        print("✅ Data loaded successfully!")
        print(self.df.head())

    def monthly_average_price(self):
        """Calculate and visualize average oil price per month."""
        self.df['Year'] = self.df.index.year
        self.df['Month'] = self.df.index.month
        monthly_avg = self.df.groupby(['Year', 'Month'])['Price'].mean().reset_index()
        monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(day=1))

        # Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_avg, x='Date', y='Price', color='blue', linewidth=2)
        plt.title("📈 Brent Oil Price Trend Over Time", fontsize=14)
        plt.xlabel("Year")
        plt.ylabel("Oil Price (USD)")
        plt.grid(True)
        plt.show()

    def price_volatility(self):
        """Analyze oil price volatility using rolling standard deviation."""
        self.df['Rolling_Std'] = self.df['Price'].rolling(window=30).std()

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.df.index, y=self.df['Rolling_Std'], color='red')
        plt.title("🔍 Oil Price Volatility Over Time (30-Day Rolling Std)")
        plt.xlabel("Year")
        plt.ylabel("Price Volatility")
        plt.grid(True)
        plt.show()

    def price_distribution(self):
        """Show the distribution of oil prices."""
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['Price'], bins=50, kde=True, color='blue')
        plt.title("⛽ Oil Price Distribution")
        plt.xlabel("Price (USD)")
        plt.ylabel("Frequency")
        plt.show()

    def detect_change_points(self):
        """Use change point detection to identify major shifts in oil prices."""
        algo = rpt.Pelt(model="rbf").fit(self.df['Price'].values)
        change_points = algo.predict(pen=10)  # Adjust penalty for sensitivity

        # Plot price with detected change points
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Price'], label="Oil Price", color='blue')
        for cp in change_points:
            plt.axvline(self.df.index[cp-1], color='red', linestyle='dashed', label="Change Point")

        plt.title("⚠️ Major Oil Price Change Points")
        plt.xlabel("Year")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def monthly_trends_heatmap(self):
        """Generate a heatmap to analyze monthly trends over the years."""
        self.df['Year'] = self.df.index.year
        self.df['Month'] = self.df.index.month
        monthly_heatmap = self.df.pivot_table(values='Price', index='Year', columns='Month', aggfunc='mean')

        plt.figure(figsize=(12, 6))
        sns.heatmap(monthly_heatmap, cmap="coolwarm", annot=True, fmt=".1f", linewidths=0.5)
        plt.title("🔥 Monthly Oil Price Trends (Heatmap)")
        plt.xlabel("Month")
        plt.ylabel("Year")
        plt.show()

    def compare_before_after_events(self):
        """Compare oil prices before and after major events."""
        event_dates = {
            "COVID-19 Crash": "2020-03-01",
            "2008 Financial Crisis": "2008-09-15"
        }

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Price'], label="Oil Price", color='blue')

        for event, date in event_dates.items():
            plt.axvline(pd.to_datetime(date), color='red', linestyle='dashed', label=event)

        plt.title("📌 Oil Price Trends Around Key Events")
        plt.xlabel("Year")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage
if __name__ == "__main__":
    eda = BrentOilEDA("/home/nahomnadew/Desktop/10x/week10/Brent_oil/Data/Data/cleaned_BrentOilPrices.csv")
    eda.load_data()
    eda.monthly_average_price()
    eda.price_volatility()
    eda.price_distribution()
    eda.detect_change_points()
    eda.monthly_trends_heatmap()
    eda.compare_before_after_events()
