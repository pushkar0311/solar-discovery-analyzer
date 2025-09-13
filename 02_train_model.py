# 02_train_model.py
# This script trains a machine learning model to predict elements from spectral data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv('solar_spectrum_clean.csv')
print(f"Dataset shape: {df.shape}")
print(f"Elements: {df['element'].unique()}")

# 2. Prepare features (X) and target (y)
X = df[['wavelength', 'intensity']]  # Input features
y = df['element']                    # Output labels

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# 4. Create and train the Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.3f} ({accuracy:.2%})")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Feature importance
feature_importance = pd.DataFrame({
    'feature': ['wavelength', 'intensity'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(8, 4))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance in Element Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")

# 7. Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# 8. Test predictions on specific known wavelengths
print("\nTesting predictions on known spectral lines:")
test_cases = [
    (6562.8, 0.15, 'Hydrogen'),    # H-alpha line
    (5889.95, 0.85, 'Sodium'),     # Na D2 line
    (5172.7, 0.92, 'Magnesium'),   # Mg b triplet
    (6173.3, 0.88, 'Iron'),        # Fe line
    (3968.5, 0.25, 'Calcium'),     # Ca H line
    (5875.6, 0.80, 'Helium')       # He D3 line
]

print("\n{:>10} | {:>8} | {:>12} | {:>12}".format("Wavelength", "Intensity", "True", "Predicted"))
print("-" * 55)
for wl, intensity, true_element in test_cases:
    prediction = model.predict([[wl, intensity]])
    proba = model.predict_proba([[wl, intensity]])
    max_proba = np.max(proba)
    print(f"{wl:10.2f} | {intensity:8.2f} | {true_element:12} | {prediction[0]:12} ({max_proba:.2%})")

print("\nModel training and evaluation complete!")