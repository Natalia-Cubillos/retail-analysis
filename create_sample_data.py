# create_sample_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n_rows = 100000  # total rows

# Create a base date range of 1000 days (~3 years)
base_dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")

# Repeat the base_dates to fill n_rows
dates = np.tile(base_dates, n_rows // len(base_dates))

data = {
    'date': dates,
    'product_id': np.random.randint(1, 100, n_rows),
    'store_id': np.random.randint(1, 50, n_rows),
    'sales_amount': np.random.uniform(10, 500, n_rows),
    'units_sold': np.random.randint(1, 20, n_rows)
}

df = pd.DataFrame(data)
df.to_csv("sales_data.csv", index=False)
print(f"Created sales_data.csv with {n_rows} rows")