# debug_predictions.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def debug_predictions():
    """Debug why predictions are showing as numbers"""
    print("🔍 DEBUGGING PREDICTIONS")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('solar_spectrum_realistic.csv')
    print(f"Training data elements: {df['element'].unique()}")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[['wavelength', 'intensity']], df['element'])
    
    print(f"Model classes: {model.classes_}")
    print(f"Model classes type: {type(model.classes_[0])}")
    
    # Test with some sample data
    test_data = pd.DataFrame({
        'wavelength': [3968.5, 5172.7, 5889.95],
        'intensity': [0.25, 0.92, 0.85]
    })
    
    # Make predictions
    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)
    
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"Predictions: {predictions}")
    print(f"Predictions type: {type(predictions[0])}")
    print(f"Prediction probabilities shape: {probabilities.shape}")
    
    # Check each prediction
    for i, (wl, intensity, pred) in enumerate(zip(test_data['wavelength'], 
                                                 test_data['intensity'], 
                                                 predictions)):
        confidence = np.max(probabilities[i])
        print(f"{i+1}. {wl} Å -> {pred} (confidence: {confidence:.1%})")
    
    # Check if predictions are numeric
    if all(isinstance(p, (int, float, np.number)) for p in predictions):
        print("\n❌ PROBLEM: Predictions are numeric instead of strings!")
        print("This means the model is returning class indices instead of names")
    else:
        print("\n✅ Predictions are correctly returning element names")

if __name__ == "__main__":
    debug_predictions()