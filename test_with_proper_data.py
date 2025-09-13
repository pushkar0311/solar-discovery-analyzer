# test_with_proper_data.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def test_with_proper_data():
    """Test the model with the properly formatted test data"""
    print("🧪 TESTING WITH PROPER_TEST_DATA.CSV")
    print("=" * 50)
    
    # Load the training data and model
    df = pd.read_csv('solar_spectrum_realistic.csv')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[['wavelength', 'intensity']], df['element'])
    
    # Load the proper test data
    test_df = pd.read_csv('proper_test_data.csv')
    print("Test data:")
    print(test_df)
    print()
    
    # Make predictions
    predictions = model.predict(test_df[['wavelength', 'intensity']])
    confidence_scores = np.max(model.predict_proba(test_df[['wavelength', 'intensity']]), axis=1)
    
    # Display results
    print("📊 PREDICTION RESULTS:")
    print("Wavelength | Intensity | Predicted Element | Confidence")
    print("-" * 55)
    
    for i, (wl, intensity, pred, conf) in enumerate(zip(
        test_df['wavelength'],
        test_df['intensity'], 
        predictions,
        confidence_scores
    )):
        print(f"{wl:9.2f} | {intensity:8.2f} | {pred:16} | {conf:8.2%}")
    
    # Check which predictions are correct
    expected_elements = ['Calcium', 'Magnesium', 'Sodium', 'Helium', 'Iron', 'Hydrogen']
    correct = 0
    total = len(predictions)
    
    print("\n✅ ACCURACY CHECK:")
    for i, (pred, expected) in enumerate(zip(predictions, expected_elements)):
        status = "✓" if pred == expected else "✗"
        print(f"{i+1}. {pred} vs {expected} {status}")
        if pred == expected:
            correct += 1
    
    accuracy = correct / total
    print(f"\n🎯 Accuracy: {correct}/{total} = {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🚀 EXCELLENT! Your model is working perfectly!")
    else:
        print("⚠️  Some predictions need improvement")

if __name__ == "__main__":
    test_with_proper_data()