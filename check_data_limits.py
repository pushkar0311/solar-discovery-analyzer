# check_data_limits.py
import pandas as pd

# Load your clean data
df = pd.read_csv('solar_spectrum_clean.csv')

print("📊 ACTUAL DATA BOUNDARIES DISCOVERY")
print("="*50)
print(f"Total data points: {len(df):,}")
print(f"Minimum wavelength: {df['wavelength'].min():.2f} Å")
print(f"Maximum wavelength: {df['wavelength'].max():.2f} Å")
print(f"Wavelength range: {df['wavelength'].max() - df['wavelength'].min():.2f} Å")

print("\n📈 Element-specific ranges:")
for element in df['element'].unique():
    element_data = df[df['element'] == element]
    min_wl = element_data['wavelength'].min()
    max_wl = element_data['wavelength'].max()
    print(f"   {element:10}: {min_wl:7.1f} - {max_wl:7.1f} Å ({(max_wl-min_wl):5.1f} Å range)")

print("\n🔍 Data distribution by wavelength:")
print(df['wavelength'].describe())

# Check if 10000 Å exists in data
has_10000 = (df['wavelength'] >= 9999.9).any()
print(f"\nContains 10000 Å: {has_10000}")
if has_10000:
    print(f"Data near 10000 Å: {df[df['wavelength'] >= 9999.9].head(3).to_string()}")