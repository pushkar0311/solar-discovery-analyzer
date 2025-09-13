# 05_visualize_coverage.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv('solar_spectrum_clean.csv')

print("📊 ACTUAL DATA COVERAGE ANALYSIS")
print("="*50)
print(f"Total data points: {len(df):,}")
print(f"Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} Å")

# Create coverage visualization
plt.figure(figsize=(12, 6))

# Plot each element's wavelength range
colors = {'Hydrogen': 'red', 'Sodium': 'blue', 'Magnesium': 'green', 
          'Iron': 'orange', 'Calcium': 'purple', 'Helium': 'cyan'}

for element in df['element'].unique():
    element_data = df[df['element'] == element]
    min_wl = element_data['wavelength'].min()
    max_wl = element_data['wavelength'].max()
    
    plt.plot([min_wl, max_wl], [element, element], 
             'o-', linewidth=3, markersize=8, color=colors.get(element, 'black'), 
             label=f'{element} ({min_wl:.1f}-{max_wl:.1f}Å)')

plt.xlabel('Wavelength (Å)')
plt.ylabel('Element')
plt.title('Actual Spectral Data Coverage in Your Dataset')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('data_coverage.png', bbox_inches='tight', dpi=300)
plt.show()

# Show gap analysis
print("\n📈 Data Coverage by 100Å Blocks:")
all_wavelengths = df['wavelength'].values
wavelength_bins = np.arange(3000, 10000, 100)
coverage = []

for i in range(len(wavelength_bins)-1):
    bin_min = wavelength_bins[i]
    bin_max = wavelength_bins[i+1]
    points_in_bin = np.sum((all_wavelengths >= bin_min) & (all_wavelengths < bin_max))
    coverage.append((bin_min, bin_max, points_in_bin))
    status = "✅ DATA" if points_in_bin > 0 else "❌ GAP"
    print(f"{bin_min:5.0f}-{bin_max:5.0f} Å: {points_in_bin:4} points {status}")

# Specifically check the 4300-4400 range
points_4300_4400 = np.sum((all_wavelengths >= 4300) & (all_wavelengths < 4400))
print(f"\n🔍 4300-4400 Å range: {points_4300_4400} data points")
if points_4300_4400 == 0:
    print("   This is a GAP in your training data!")