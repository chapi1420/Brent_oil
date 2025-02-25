import wbdata
import requests
import pandas as pd
import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class DataExtractor:
    def __init__(self):
        self.eia_api_key = "YLgVCJfXgcgLJc1cjzeTYeNNUksxVcse7USKkDlp"
        self.news_api_key = "9b14ec674d5c49b4b587af0a0c74e484"
        self.data = pd.DataFrame()

    def fetch_world_bank_data(self, country_code, indicators, start_date):
        # Fetch data from World Bank
        wb_data = wbdata.get_dataframe(indicators, country=country_code)
        wb_data.reset_index(inplace=True)
        wb_data.rename(columns={'country': 'Country', 'indicator': 'Indicator', 'value': 'Value', 'date': 'Date'}, inplace=True)

        # Convert 'Date' column to datetime
        wb_data['Date'] = pd.to_datetime(wb_data['Date'], errors='coerce')

        # Filter data for the specified date range
        wb_data = wb_data[wb_data['Date'] >= start_date]
        
        wb_data['Source'] = 'World Bank'
        self.data = pd.concat([self.data, wb_data], ignore_index=True)

    
    def fetch_eia_data(self):
        # EIA API for oil prices
        url = f"http://api.eia.gov/v2/seriesid/PET.RBRTE.D?api_key={self.eia_api_key}"
        response = requests.get(url)
        
        # Check if the response is successful
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        eia_data = response.json()
        
        # Print the entire response for debugging
        print("Full API Response:", eia_data)  # Debug line
        
        # Check if 'data' key exists in the expected structure
        if 'series' not in eia_data or not eia_data['series']:
            print("Error: 'series' key not found or is empty in response.")
            return

        # Extract the first series
        series_data = eia_data['series'][0]
        
        # Check if 'data' key exists in the series
        if 'data' not in series_data:
            print("Error: 'data' key not found in series.")
            return

        # Extract and format the data
        oil_prices = series_data['data']  # Access the 'data' key within the first series
        eia_df = pd.DataFrame(oil_prices, columns=['Date', 'Value', 'Units'])  # Adjust column names accordingly
        eia_df['Date'] = pd.to_datetime(eia_df['Date'], format='%Y%m%d')
        eia_df['Source'] = 'EIA'
        self.data = pd.concat([self.data, eia_df], ignore_index=True)




    # def fetch_news_data(self, start_date, end_date):
    #     # News API for events
    #     url = f"https://newsapi.org/v2/everything?q=oil&from={start_date}&to={end_date}&apiKey={self.news_api_key}"
    #     response = requests.get(url)
        
    #     # Check if the response is successful
    #     if response.status_code != 200:
    #         print(f"Error: {response.status_code} - {response.text}")
    #         return
        
    #     news_data = response.json()
        
    #     # Print the response to debug
    #     print(news_data)  # Debug line

    #     # Check if 'articles' key exists
    #     if 'articles' not in news_data:
    #         print("Error: 'articles' key not found in response.")
    #         return
        
    #     # Extract and format articles
    #     articles = news_data['articles']
    #     news_df = pd.DataFrame(articles)
    #     news_df = news_df[['title', 'publishedAt']]
    #     news_df.columns = ['Event', 'Date']
    #     news_df['Source'] = 'News API'
    #     news_df['Date'] = pd.to_datetime(news_df['Date'])
    #     self.data = pd.concat([self.data, news_df], ignore_index=True)


    def save_to_csv(self, filename):
        # Save the combined data to a CSV file
        self.data.to_csv(filename, index=False)

# Usage
if __name__ == "__main__":
    # Initialize the DataExtractor
    extractor = DataExtractor()
    
    # Define parameters
    country_code = "USA"  # Change as needed
    indicators = {"NY.GDP.MKTP.CD": "GDP"}  # Example indicator
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Fetch data
    extractor.fetch_world_bank_data(country_code, indicators, start_date)
    extractor.fetch_eia_data()
    # extractor.fetch_news_data(start_date.strftime('%Y-%m-%d'), end_date)

    # Save to CSV
    extractor.save_to_csv("combined_data.csv")
