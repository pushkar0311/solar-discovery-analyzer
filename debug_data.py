# debug_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_training_data():
    """Analyze the training data to understand its structure"""
    print("🔍 ANALYZING TRAINING DATA")
    print("=" * 50)
    
    try:
        # Load training data
        df = pd.read_csv('solar_spectrum_realistic.csv')
        print(f"Training data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Check data types
        print(f"\nData types:\n{df.dtypes}")
        
        # Show unique elements
        print(f"\nUnique elements: {df['element'].unique()}")
        
        # Analyze each element's wavelength range
        print("\n📊 ELEMENT WAVELENGTH RANGES:")
        for element in df['element'].unique():
            element_data = df[df['element'] == element]
            min_wl = element_data['wavelength'].min()
            max_wl = element_data['wavelength'].max()
            count = len(element_data)
            print(f"{element:10}: {min_wl:7.2f} - {max_wl:7.2f} Å ({count} samples)")
        
        # Create a plot of the training data
        plt.figure(figsize=(12, 6))
        colors = {'Hydrogen': 'red', 'Helium': 'orange', 'Oxygen': 'green', 
                 'Nitrogen': 'blue', 'Iron': 'purple', 'Calcium': 'brown'}
        
        for element in df['element'].unique():
            element_data = df[df['element'] == element]
            plt.scatter(element_data['wavelength'], element_data['intensity'], 
                       label=element, alpha=0.7, s=10)
        
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Intensity')
        plt.title('Training Data: Spectral Lines by Element')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_data_plot.png')
        print("\n📈 Plot saved as 'training_data_plot.png'")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def analyze_test_data(test_file):
    """Analyze the test data and compare with training data"""
    print("\n🔍 ANALYZING TEST DATA")
    print("=" * 50)
    
    try:
        # Load test data
        test_df = pd.read_csv(test_file)
        print(f"Test data shape: {test_df.shape}")
        print(f"Columns: {list(test_df.columns)}")
        
        # Show test data
        print(f"\nTest data:\n{test_df}")
        
        # Check if wavelengths are in training ranges
        train_df = pd.read_csv('solar_spectrum_realistic.csv')
        train_min = train_df['wavelength'].min()
        train_max = train_df['wavelength'].max()
        
        out_of_range = test_df[(test_df['wavelength'] < train_min) | 
                              (test_df['wavelength'] > train_max)]
        
        print(f"\n📏 Training data range: {train_min:.2f} - {train_max:.2f} Å")
        
        if len(out_of_range) > 0:
            print(f"❌ {len(out_of_range)} test points outside training range:")
            print(out_of_range)
        else:
            print("✅ All test points within training range")
            
        return test_df
        
    except Exception as e:
        print(f"❌ Error analyzing test data: {e}")
        return None

def check_model_predictions():
    """Check what the model predicts for sample data"""
    print("\n🔍 CHECKING MODEL PREDICTIONS")
    print("=" * 50)
    
    try:
        # Load model and data
        df = pd.read_csv('solar_spectrum_realistic.csv')
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df[['wavelength', 'intensity']], df['element'])
        
        # Test with known spectral lines
        test_cases = [
            (3968.5, 0.25, 'Calcium'),
            (5172.7, 0.92, 'Magnesium'),
            (5889.95, 0.85, 'Sodium'),
            (5875.6, 0.80, 'Helium'),
            (6173.3, 0.88, 'Iron'),
            (6562.8, 0.15, 'Hydrogen')
        ]
        
        print("Testing model with known spectral lines:")
        print("\nWavelength | Intensity | Expected | Predicted | Confidence")
        print("-" * 60)
        
        for wl, intensity, expected in test_cases:
            prediction = model.predict([[wl, intensity]])
            confidence = np.max(model.predict_proba([[wl, intensity]]))
            status = "✅" if prediction[0] == expected else "❌"
            print(f"{wl:9.2f} | {intensity:8.2f} | {expected:8} | {prediction[0]:8} | {confidence:8.2%} {status}")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def create_proper_test_file():
    """Create a properly formatted test file"""
    print("\n🔧 CREATING PROPER TEST FILE")
    print("=" * 50)
    
    # Create test data with correct wavelengths
    test_data = {
        'wavelength': [3968.5, 5172.7, 5889.95, 5875.6, 6173.3, 6562.8],
        'intensity': [0.25, 0.92, 0.85, 0.80, 0.88, 0.15]
    }
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('proper_test_data.csv', index=False)
    print("✅ Created 'proper_test_data.csv' with correct wavelengths")
    print(test_df)
    
    return test_df

if __name__ == "__main__":
    # Analyze training data
    train_df = analyze_training_data()
    
    # Check model predictions
    check_model_predictions()
    
    # Create a proper test file
    create_proper_test_file()
    
    # Analyze the proper test file
    analyze_test_data('proper_test_data.csv')
    
    print("\n🎉 Debugging complete! Check the outputs above to identify issues.")