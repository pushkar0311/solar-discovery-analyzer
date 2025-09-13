# debug_elements.py
import pandas as pd
import numpy as np

def check_element_data():
    """Check what's happening with the element column"""
    print("🔍 CHECKING ELEMENT DATA ISSUE")
    print("=" * 50)
    
    # Load the data
    df = pd.read_csv('solar_spectrum_realistic.csv')
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Element column type: {df['element'].dtype}")
    print(f"Unique values in element column: {df['element'].unique()}")
    print(f"First 10 element values: {df['element'].head(10).values}")
    
    # Check if elements are numeric
    if df['element'].dtype in [np.int64, np.float64]:
        print("❌ PROBLEM: Element column is numeric instead of strings!")
        print("This is why you're seeing '0' instead of element names")
    else:
        print("✅ Element column is correctly formatted as strings")

if __name__ == "__main__":
    check_element_data()