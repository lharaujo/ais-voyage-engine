import polars as pl

# Load your new file
df = pl.read_parquet("./data/gold/voyages_2025_01.parquet")

# Check the count and unique vessels
print(f"Total rows: {len(df)}")
print(f"Unique vessels (IMO): {df['imo'].n_unique()}")
print(f"Length of the dataset: {len(df['imo'])}")

# Look at the first few rows
print(df.head())
