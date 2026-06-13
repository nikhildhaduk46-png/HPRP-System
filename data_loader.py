import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def basic_info(df):
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Column Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Basic Stats ---")
    print(df.describe())
    
