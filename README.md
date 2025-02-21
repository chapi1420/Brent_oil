# Brent Oil Price Analysis

## Overview

This project analyzes Brent oil price data to uncover trends, detect major change points, and explore relationships between price fluctuations and external factors. The analysis involves preprocessing raw data, conducting exploratory data analysis (EDA), and visualizing key insights.

---

## Project Structure

### 1. **Step 1: Data Preprocessing**

- **Objective**: Clean and prepare the dataset for analysis.
- **Key Steps**:
  - Load raw data from CSV.
  - Convert date columns to `datetime` format.
  - Handle missing values by dropping or imputing them.
  - Remove duplicate records to ensure data integrity.
  - Set `Date` as the index for time-series analysis.
  - Save the cleaned dataset for further processing.
- **Output**: A structured and clean dataset ready for EDA.

### 2. **Step 2: Exploratory Data Analysis (EDA)**

- **Objective**: Gain insights into historical oil price trends.
- **Key Steps**:
  - Compute summary statistics (mean, median, standard deviation, etc.).
  - Visualize the historical trend of oil prices over the years.
  - Analyze average oil price per month to identify seasonality.
  - Detect major change points in price fluctuations using `ruptures`.
  - Overlay historical events to explore correlations with price changes.
- **Output**: Graphical insights and statistical summaries that highlight patterns in oil prices.

---

## Key Features

1. **Data Preprocessing**:
   - Ensures data consistency and removes inaccuracies.
   - Standardizes date formats for time-series analysis.
2. **Statistical Analysis**:
   - Generates key descriptive statistics for understanding trends.
3. **Visualization**:
   - Uses `Matplotlib` and `Seaborn` to create insightful visualizations.
   - Detects significant change points in price trends with `ruptures`.
   - Highlights key historical events affecting oil prices.

---

## Requirements

- **Python**: 3.8 or later
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `ruptures`
  - `datetime`

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository/oil-price-analysis.git
   cd oil-price-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the data preprocessing script:

```bash
python preprocess_data.py
```

Run the exploratory analysis script:

```bash
python eda_analysis.py
```

---

## Next Steps

- Implement predictive modeling to forecast future oil prices.
- Deploy insights via an interactive Streamlit dashboard.

---

## Author

Developed by Nahom T. Nadew
