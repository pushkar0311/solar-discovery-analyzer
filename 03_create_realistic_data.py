# 03_create_realistic_data.py - UNBIASED VERSION
import pandas as pd
import numpy as np

# Real solar abundances (relative number of atoms)
solar_abundances = {
    'Hydrogen': 120000,  # 12.00 - 91% of atoms
    'Helium':    25000,   # 10.93 - 8.9% of atoms
    'Oxygen':      800,   # 8.69
    'Carbon':      500,   # 8.43
    'Iron':        350,   # 7.47
    'Neon':        300,   # 8.00
    'Nitrogen':    250,   # 7.83
    'Magnesium':   200,   # 7.60
    'Silicon':     180,   # 7.51
    'Sulfur':      120,   # 7.12
    'Calcium':      80,   # 6.36
    'Sodium':       50,   # 6.24
}

print("Loading original spectral data...")
df = pd.read_csv('solar_spectrum_clean.csv')

# Create realistic dataset based on abundances
realistic_data = []

print("Creating realistic abundance distribution...")
for element, abundance in solar_abundances.items():
    if element in df['element'].unique():
        element_data = df[df['element'] == element].copy()
        n_samples = min(abundance, len(element_data))
        sampled_data = element_data.sample(n=n_samples, random_state=42, replace=True)
        realistic_data.append(sampled_data)
    else:
        print(f"Warning: No data found for {element}")

# Combine all data
realistic_df = pd.concat(realistic_data, ignore_index=True)
realistic_df = realistic_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCreated realistic dataset with {len(realistic_df):,} samples")
print("\nElement distribution:")
element_counts = realistic_df['element'].value_counts()
print(element_counts)

print("\nPercentages:")
percentages = (element_counts / len(realistic_df) * 100).round(2)
print(percentages)

# Save the clean, unbiased dataset
realistic_df.to_csv('solar_spectrum_realistic.csv', index=False)
print(f"\nSaved realistic dataset to 'solar_spectrum_realistic.csv'")

# Save only the abundance data (no descriptions)
abundance_data = pd.DataFrame({
    'element': list(solar_abundances.keys()),
    'log_abundance': [12.00, 10.93, 8.69, 8.43, 7.47, 8.00, 7.83, 7.60, 7.51, 7.12, 6.36, 6.24]
})
abundance_data.to_csv('solar_abundances.csv', index=False)
print("Saved solar abundance data to 'solar_abundances.csv'")