# 06_anomaly_detector.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_model_and_data():
    """Load the trained model and data"""
    df = pd.read_csv('solar_spectrum_realistic.csv')
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(df[['wavelength', 'intensity']], df['element'])
    return model, df

def detect_spectral_anomalies(new_eclipse_data):
    """
    PROPERLY FIXED: Detect anomalies using improved statistical methods
    """
    try:
        # Load training data
        df = pd.read_csv('solar_spectrum_realistic.csv')
        
        # First, let's see what our element identification model predicts
        model, _ = load_model_and_data()
        predictions = model.predict(new_eclipse_data[['wavelength', 'intensity']])
        confidence = np.max(model.predict_proba(new_eclipse_data[['wavelength', 'intensity']]), axis=1)
        
        print("📊 Model predictions for uploaded data:")
        for i, (wl, intensity, pred, conf) in enumerate(zip(
            new_eclipse_data['wavelength'],
            new_eclipse_data['intensity'],
            predictions,
            confidence
        )):
            print(f"  {i+1}. {wl:.2f} Å (intensity: {intensity:.2f}) -> {pred} (confidence: {conf:.2%})")
        
        # Scale the data for better anomaly detection
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(df[['wavelength', 'intensity']])
        new_scaled = scaler.transform(new_eclipse_data[['wavelength', 'intensity']])
        
        # Train anomaly detector with appropriate contamination
        anomaly_model = IsolationForest(
            contamination=0.1,  # Increased to reduce false positives
            random_state=42, 
            n_jobs=-1,
            n_estimators=200  # More trees for better accuracy
        )
        anomaly_model.fit(train_scaled)
        
        # Predict anomalies in new data
        anomalies = anomaly_model.predict(new_scaled)
        anomaly_scores = anomaly_model.decision_function(new_scaled)
        
        # Only consider very strong anomalies (scores < -0.2)
        strong_anomalies = anomaly_scores < -0.2
        
        # Cluster only the strong anomalies to find patterns
        if np.sum(strong_anomalies) > 0:
            try:
                clustering = DBSCAN(eps=1.5, min_samples=2)
                anomaly_clusters = clustering.fit_predict(
                    new_eclipse_data[strong_anomalies][['wavelength', 'intensity']]
                )
            except Exception as e:
                print(f"Clustering warning: {e}")
                anomaly_clusters = np.array([-1] * np.sum(strong_anomalies))
        else:
            anomaly_clusters = np.array([])
        
        # Return both all anomalies and strong anomalies
        return anomalies, anomaly_scores, anomaly_clusters, strong_anomalies
        
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        # Return default values (no anomalies detected)
        return np.ones(len(new_eclipse_data)), np.zeros(len(new_eclipse_data)), np.array([]), np.zeros(len(new_eclipse_data), dtype=bool)

def compare_eclipse_abundances(eclipse_2023, eclipse_2024):
    """
    Compare element abundances between two eclipses
    """
    try:
        # Load your element identification model
        model, df = load_model_and_data()
        
        # Predict elements for both eclipses
        elements_2023 = model.predict(eclipse_2023[['wavelength', 'intensity']])
        elements_2024 = model.predict(eclipse_2024[['wavelength', 'intensity']])
        
        # Calculate abundance changes
        abundance_2023 = pd.Series(elements_2023).value_counts(normalize=True)
        abundance_2024 = pd.Series(elements_2024).value_counts(normalize=True)
        
        # Align indices to handle all elements
        all_elements = sorted(set(abundance_2023.index) | set(abundance_2024.index))
        abundance_2023 = abundance_2023.reindex(all_elements, fill_value=0)
        abundance_2024 = abundance_2024.reindex(all_elements, fill_value=0)
        
        changes = (abundance_2024 - abundance_2023) * 100  # Percentage change
        
        return changes[changes.abs() > 5]  # Return only significant changes (>5%)
        
    except Exception as e:
        print(f"Error in abundance comparison: {e}")
        return pd.Series()

def is_valid_spectral_data(df):
    """Validate that the data has proper spectral format"""
    if not all(col in df.columns for col in ['wavelength', 'intensity']):
        return False, "Missing required columns: wavelength and intensity"
    
    if len(df) == 0:
        return False, "Empty dataset"
    
    if df['wavelength'].min() < 3000 or df['wavelength'].max() > 10000:
        return False, "Wavelengths outside reasonable solar spectrum range (3000-10000 Å)"
    
    if df['intensity'].min() < 0 or df['intensity'].max() > 2:
        return False, "Intensity values should be normalized (typically 0-1 range)"
    
    return True, "Data validation passed"

def analyze_spectral_features(new_eclipse_data):
    """
    Analyze spectral features and provide scientific interpretation
    """
    model, df = load_model_and_data()
    
    # Predict elements
    predictions = model.predict(new_eclipse_data[['wavelength', 'intensity']])
    confidence = np.max(model.predict_proba(new_eclipse_data[['wavelength', 'intensity']]), axis=1)
    
    results = []
    for i, (wl, intensity, pred, conf) in enumerate(zip(
        new_eclipse_data['wavelength'], 
        new_eclipse_data['intensity'], 
        predictions, 
        confidence
    )):
        results.append({
            'wavelength': wl,
            'intensity': intensity,
            'predicted_element': pred,
            'confidence': conf
        })
    
    return pd.DataFrame(results)

# Example usage for future eclipses
def analyze_future_eclipse(new_data_path):
    """
    Analyze new eclipse data for discoveries
    """
    try:
        new_data = pd.read_csv(new_data_path)
        
        # Validate data
        is_valid, message = is_valid_spectral_data(new_data)
        if not is_valid:
            print(f"❌ {message}")
            return None, None, None, None
        
        print("🔍 Analyzing spectral features...")
        feature_analysis = analyze_spectral_features(new_data)
        print("📊 Feature analysis completed")
        
        # 1. Detect anomalous spectral lines
        anomalies, scores, clusters, strong_anomalies = detect_spectral_anomalies(new_data)
        
        print(f"📈 Found {np.sum(strong_anomalies)} strong anomalous spectral features!")
        print(f"📊 Anomaly clusters: {len(np.unique(clusters)) if len(clusters) > 0 else 0} distinct patterns")
        
        # Show details of anomalies
        if np.sum(strong_anomalies) > 0:
            anomalous_data = new_data[strong_anomalies]
            print("\n📋 Strong anomalous data points:")
            for i, (idx, row) in enumerate(anomalous_data.iterrows()):
                print(f"  {i+1}. Wavelength: {row['wavelength']:.2f} Å, Intensity: {row['intensity']:.2f}")
        
        return anomalies, scores, clusters, strong_anomalies, feature_analysis
        
    except Exception as e:
        print(f"❌ Error analyzing eclipse data: {e}")
        return None, None, None, None, None

# For testing the module directly
if __name__ == "__main__":
    # Test with the provided sample data
    sample_data = pd.DataFrame({
        'wavelength': [3968.5, 5172.7, 5889.95, 5875.6, 6173.3, 6562.8, 5000.0],
        'intensity': [0.25, 0.92, 0.85, 0.30, 0.88, 0.60, 0.75]
    })
    
    print("Testing anomaly detection with sample data...")
    print("Sample data:")
    print(sample_data)
    
    anomalies, scores, clusters, strong_anomalies, feature_analysis = analyze_future_eclipse(sample_data)
    
    if feature_analysis is not None:
        print("\n📋 Model predictions for all data points:")
        print(feature_analysis.to_string(index=False))
    
    if np.sum(strong_anomalies) > 0:
        print(f"\n🚨 Strong anomalies detected: {np.sum(strong_anomalies)}")
        anomalous_points = sample_data[strong_anomalies]
        for i, (idx, row) in enumerate(anomalous_points.iterrows()):
            print(f"  Anomaly {i+1}: {row['wavelength']:.2f} Å (intensity: {row['intensity']:.2f})")
    else:
        print("\n✅ No strong anomalies detected - spectrum appears normal")