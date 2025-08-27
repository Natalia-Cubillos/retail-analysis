# analysis.py
import pandas as pd
import matplotlib.pyplot as plt

def analyze_sales():
    # Load the data
    df = pd.read_csv("sales_data.csv")

    # Basic analysis
    print("Dataset shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)

    print("\nSummary statistics:")
    print(df.describe())

    # Sales by store
    sales_by_store = df.groupby('store_id')['sales_amount'].sum()
    print("\nTop 5 stores by sales:")
    print(sales_by_store.sort_values(ascending=False).head())

    # Plot sales distribution
    plt.figure(figsize=(10, 6))
    df['sales_amount'].hist(bins=50)
    plt.title('Sales Amount Distribution')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    plt.savefig('sales_distribution.png')
    print("\nSaved sales distribution plot to sales_distribution.png")

if __name__ == "__main__":
    analyze_sales()