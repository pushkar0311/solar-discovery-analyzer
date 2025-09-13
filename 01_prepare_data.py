# 01_prepare_data.py
# FIXED VERSION: Correctly handles the data format

import pandas as pd
import os
import matplotlib.pyplot as plt

# 1. Define the files we downloaded and what element they represent
file_element_map = {
    '01_hydrogen.txt': 'Hydrogen',
    '02_sodium.txt': 'Sodium', 
    '03_magnesium.txt': 'Magnesium',
    '04_iron.txt': 'Iron',
    '05_calcium.txt': 'Calcium',
    '06_helium.txt': 'Helium'
}

# 2. Create an empty list to hold all our data
all_data_frames = []

# 3. Loop through each file and load its data
for filename, element_name in file_element_map.items():
    if os.path.exists(filename):
        print(f"Loading {filename} for {element_name}...")
        
        # FIX: Read the file correctly - skip header, handle the 3 columns properly
        df = pd.read_csv(filename, skiprows=1, header=None, usecols=[0, 1], 
                        names=['wavelength', 'intensity'])
        
        # Convert to numeric (just to be safe)
        df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        
        # Add a column to identify the element
        df['element'] = element_name
        
        # Add this data to our list
        all_data_frames.append(df)
        
        print(f"  Found {len(df)} data points.")
    else:
        print(f"Warning: File {filename} not found. Skipping.")

# 4. Check if we loaded any data
if not all_data_frames:
    print("No data was loaded! Please check your files.")
    exit()

# 5. Combine all data into one DataFrame
full_dataset = pd.concat(all_data_frames, ignore_index=True)

# 6. Clean the data
# Convert intensity from ~8000-10000 scale to 0-1 scale (where 1.0 is continuum)
full_dataset['intensity'] = full_dataset['intensity'] / 10000.0

# Remove any possible rows with missing values
full_dataset = full_dataset.dropna()

print(f"\nSuccessfully created combined dataset!")
print(f"Total number of spectral measurements: {len(full_dataset)}")
print(f"Elements in dataset: {full_dataset['element'].unique()}")

# 7. Save the clean, combined dataset to a new CSV file
output_filename = 'solar_spectrum_clean.csv'
full_dataset.to_csv(output_filename, index=False)
print(f"\nClean data saved to: {output_filename}")

# 8. (Optional) Create a plot to see our data
plt.figure(figsize=(12, 6))
for element in full_dataset['element'].unique():
    element_data = full_dataset[full_dataset['element'] == element]
    plt.scatter(element_data['wavelength'], element_data['intensity'], label=element, s=10, alpha=0.6)

plt.title('Solar Spectrum Data - All Elements')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to show absorption lines as dips
plt.tight_layout()
plt.savefig('data_preview.png')
print("Preview plot saved as 'data_preview.png'")

# 9. Show a sample of the final data
print("\nSample of the final cleaned data:")
print(full_dataset.head(10))