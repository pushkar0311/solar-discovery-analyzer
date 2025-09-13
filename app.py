# app.py - SMART SOLAR SPECTRUM ANALYZER WITH FUTURE PREDICTION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Solar Discovery Analyzer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("☀️ Solar Discovery Analyzer")
st.markdown("""
**Element identification + Future anomaly detection** • **Trained on key spectral lines** • **Scientific prediction engine**
""")

# Load data and train model
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv('solar_spectrum_realistic.csv')
    
    # Create a simple mapping for elements to ensure proper display
    element_mapping = {
        'Calcium': 'Calcium',
        'Magnesium': 'Magnesium', 
        'Sodium': 'Sodium',
        'Helium': 'Helium',
        'Iron': 'Iron',
        'Hydrogen': 'Hydrogen'
    }
    
    # Ensure all elements are properly mapped
    df['element'] = df['element'].map(element_mapping).fillna(df['element'])
    
    # Train model directly on element names
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(df[['wavelength', 'intensity']], df['element'])
    
    return model, df

model, df = load_model_and_data()

# Element ranges
ELEMENT_RANGES = {
    'Calcium': (3964.0, 3974.0),
    'Magnesium': (5168.0, 5178.0),
    'Sodium': (5885.0, 5895.0),
    'Helium': (5871.0, 5881.0),
    'Iron': (6168.0, 6178.0),
    'Hydrogen': (6558.0, 6568.0)
}

# Simple element detection based on wavelength
def detect_element(wavelength):
    for element, (min_wl, max_wl) in ELEMENT_RANGES.items():
        if min_wl <= wavelength <= max_wl:
            return element
    return "Unknown"

def clean_dataframe(df):
    """
    Ensure dataframe is Arrow-compatible by fixing data types and handling NaN values
    """
    if df.empty:
        return df
        
    clean_df = df.copy()
    
    # Convert numeric columns, handling errors by coercing to NaN
    for col in clean_df.columns:
        if clean_df[col].dtype == object:
            try:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            except:
                pass
    
    # Fill NaN values with appropriate defaults
    for col in clean_df.columns:
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            if not clean_df[col].isna().all():
                fill_value = clean_df[col].median() if not np.isnan(clean_df[col].median()) else 0
                clean_df[col] = clean_df[col].fillna(fill_value)
            else:
                clean_df[col] = clean_df[col].fillna(0)
        else:
            if not clean_df[col].isna().all() and len(clean_df[col].mode()) > 0:
                clean_df[col] = clean_df[col].fillna(clean_df[col].mode().iloc[0])
            else:
                clean_df[col] = clean_df[col].fillna('')
    
    return clean_df

def get_suggested_ranges(requested_min, requested_max):
    """Suggest available ranges near the requested range"""
    suggestions = []
    
    for element, (elem_min, elem_max) in ELEMENT_RANGES.items():
        if (requested_max >= elem_min and requested_min <= elem_max):
            suggestions.append(f"{element} range: {elem_min:.1f}-{elem_max:.1f} Å")
        elif abs(requested_min - elem_min) < 100:
            suggestions.append(f"Near {element}: {elem_min:.1f}-{elem_max:.1f} Å")
    
    return suggestions

def detect_spectral_anomalies(new_data):
    """
    Detect anomalies in spectral data with proper handling of untrained wavelengths
    """
    try:
        # Get predictions directly as element names
        predictions = model.predict(new_data[['wavelength', 'intensity']])
        confidence_scores = np.max(model.predict_proba(new_data[['wavelength', 'intensity']]), axis=1)
        
        st.write("📊 **Detailed Analysis Results:**")
        
        # Create a detailed results table
        results = []
        anomalies = []
        
        for i, (wl, intensity, pred, conf) in enumerate(zip(
            new_data['wavelength'],
            new_data['intensity'],
            predictions,
            confidence_scores
        )):
            # Check if wavelength is in any trained range
            in_trained_range = False
            expected = "Unknown"
            
            for element, (min_wl, max_wl) in ELEMENT_RANGES.items():
                if min_wl <= wl <= max_wl:
                    in_trained_range = True
                    expected = element
                    break
            
            # Determine status
            if not in_trained_range:
                status = "❌"  # Outside trained range
                anomalies.append(i)
            elif pred == expected:
                status = "✅"  # Correct prediction
            else:
                status = "⚠️"  # Wrong prediction in trained range
                anomalies.append(i)
            
            results.append({
                'Wavelength': f"{wl:.2f} Å",
                'Intensity': intensity,
                'Expected': expected,
                'Predicted': pred,
                'Confidence': f"{conf:.1%}",
                'Status': status,
                'In Trained Range': "Yes" if in_trained_range else "No"
            })
        
        # Display the results
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, hide_index=True)
        
        # Explain what's happening
        st.info("""
        **📝 Interpretation Guide:**
        - ✅ = Prediction matches expected element for this wavelength
        - ⚠️ = Prediction differs from expected (potential anomaly)
        - ❌ = Wavelength outside trained range (model uncertainty)
        - Confidence < 60% = Uncertain prediction (needs investigation)
        """)
        
        st.success(f"🔍 Found {len(anomalies)} potential anomalies needing investigation")
        
        # For each anomaly, provide explanation
        if anomalies:
            st.write("**🧪 Anomaly Details:**")
            for i in anomalies:
                result = results[i]
                wl = new_data.iloc[i]['wavelength']
                pred = predictions[i]
                conf = confidence_scores[i]
                in_trained_range = result['In Trained Range'] == "Yes"
                
                st.write(f"- **{wl:.2f} Å**: Predicted as {pred} with {conf:.1%} confidence")
                
                if not in_trained_range:
                    st.write(f"  → ❗ Wavelength {wl:.2f} Å is outside trained ranges")
                    st.write(f"  → Model was only trained on specific spectral lines")
                    
                    # Find closest trained range
                    closest_range = None
                    min_distance = float('inf')
                    for element, (min_wl, max_wl) in ELEMENT_RANGES.items():
                        distance = min(abs(wl - min_wl), abs(wl - max_wl))
                        if distance < min_distance:
                            min_distance = distance
                            closest_range = (element, min_wl, max_wl)
                    
                    if closest_range:
                        element, min_wl, max_wl = closest_range
                        st.write(f"  → Closest trained range: {element} ({min_wl:.1f}-{max_wl:.1f} Å)")
                
                elif pred != result['Expected']:
                    st.write(f"  → Expected {result['Expected']} but predicted {pred}")
                    if conf < 0.6:
                        st.write(f"  → Low confidence ({conf:.1%}) suggests model uncertainty")
                
                elif conf < 0.6:
                    st.write(f"  → Low prediction confidence ({conf:.1%})")
        
        # Convert to anomaly scores for visualization
        anomaly_scores = np.zeros(len(new_data))
        for i in anomalies:
            anomaly_scores[i] = -1
        
        return np.where(anomaly_scores == -1, -1, 1), confidence_scores, np.array([])
        
    except Exception as e:
        st.error(f"❌ Error in analysis: {str(e)}")
        return np.ones(len(new_data)), np.zeros(len(new_data)), np.array([])

def validate_and_analyze_uploaded_data(uploaded_data):
    """
    Validate uploaded data and provide detailed analysis
    """
    st.write("🔍 **Data Validation Report:**")
    
    # Check basic structure
    if not all(col in uploaded_data.columns for col in ['wavelength', 'intensity']):
        st.error("❌ Missing required columns: wavelength and intensity")
        return False
    
    st.write(f"✅ Data shape: {uploaded_data.shape}")
    st.write(f"✅ Wavelength range: {uploaded_data['wavelength'].min():.2f} - {uploaded_data['wavelength'].max():.2f} Å")
    st.write(f"✅ Intensity range: {uploaded_data['intensity'].min():.2f} - {uploaded_data['intensity'].max():.2f}")
    
    # Check if data is within trained range
    DATA_MIN = df['wavelength'].min()
    DATA_MAX = df['wavelength'].max()
    out_of_range = uploaded_data[(uploaded_data['wavelength'] < DATA_MIN) | 
                                (uploaded_data['wavelength'] > DATA_MAX)]
    if len(out_of_range) > 0:
        st.warning(f"⚠️ {len(out_of_range)} data points outside trained range ({DATA_MIN:.1f}-{DATA_MAX:.1f} Å)")
    
    return True

def show_trained_ranges():
    """Display the trained wavelength ranges"""
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Trained Wavelength Ranges:**
    - Calcium: 3964.0-3974.0 Å
    - Magnesium: 5168.0-5178.0 Å  
    - Helium: 5871.0-5881.0 Å
    - Sodium: 5885.0-5895.0 Å
    - Iron: 6168.0-6178.0 Å
    - Hydrogen: 6558.0-6568.0 Å
    """)

# Sidebar
st.sidebar.header("🔧 Analysis Configuration")
analysis_mode = st.sidebar.radio("Analysis Mode:", 
                                ["Spectral Range Analysis", 
                                 "Element-Specific Analysis",
                                 "Future Eclipse Analysis"])

# Show trained ranges
show_trained_ranges()

# Main analysis area
if analysis_mode == "Spectral Range Analysis":
    st.header("🔍 Spectral Range Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        min_wl = st.number_input("Minimum Wavelength (Å)", value=6560.0, step=0.1)
    with col2:
        max_wl = st.number_input("Maximum Wavelength (Å)", value=6565.0, step=0.1)
    
    if st.button("🚀 Analyze Spectral Range", type="primary"):
        range_data = df[(df['wavelength'] >= min_wl) & (df['wavelength'] <= max_wl)]
        
        if len(range_data) == 0:
            st.error(f"❌ No spectral data in range {min_wl}-{max_wl} Å")
            
            # Suggest available ranges
            suggestions = get_suggested_ranges(min_wl, max_wl)
            if suggestions:
                st.info("💡 **Available spectral regions:**")
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
            else:
                st.warning("""
                **Your dataset contains isolated regions around these specific elements:**
                - Calcium H line: 3964-3974 Å
                - Magnesium b triplet: 5168-5178 Å  
                - Sodium D lines: 5885-5895 Å
                - Helium D3: 5871-5881 Å
                - Iron: 6168-6178 Å
                - Hydrogen H-alpha: 6558-6568 Å
                """)
        else:
            with st.spinner("Analyzing spectral patterns..."):
                # Get predictions
                predictions = model.predict(range_data[['wavelength', 'intensity']])
                
                # Create results table
                results = []
                for element in np.unique(predictions):
                    count = np.sum(predictions == element)
                    percentage = (count / len(predictions)) * 100
                    results.append({
                        'Element': element,
                        'Confidence': f"{percentage:.1f}%",
                        'Detections': count
                    })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, width='stretch', hide_index=True)
                
                # Simple visualization
                if len(results) > 0:
                    fig = px.pie(values=[r['Detections'] for r in results], 
                                names=[r['Element'] for r in results],
                                title='Element Distribution')
                    st.plotly_chart(fig, width='stretch')

elif analysis_mode == "Element-Specific Analysis":
    st.header("🎯 Element-Specific Analysis")
    
    selected_element = st.selectbox("Choose element to analyze:", list(ELEMENT_RANGES.keys()))
    elem_min, elem_max = ELEMENT_RANGES[selected_element]
    
    st.info(f"**{selected_element} spectral range:** {elem_min:.1f}-{elem_max:.1f} Å")
    
    col1, col2 = st.columns(2)
    with col1:
        analysis_min = st.number_input("Analysis start (Å)", value=float(elem_min), step=0.1)
    with col2:
        analysis_max = st.number_input("Analysis end (Å)", value=float(elem_max), step=0.1)
    
    if st.button(f"🔬 Analyze {selected_element} Region", type="primary"):
        range_data = df[(df['wavelength'] >= analysis_min) & (df['wavelength'] <= analysis_max)]
        
        if len(range_data) > 0:
            predictions = model.predict(range_data[['wavelength', 'intensity']])
            
            # Count predictions
            element_counts = {}
            for pred in predictions:
                element_counts[pred] = element_counts.get(pred, 0) + 1
            
            # Create results
            results = []
            for element, count in element_counts.items():
                percentage = (count / len(predictions)) * 100
                results.append({
                    'Element': element,
                    'Detection Count': count,
                    'Percentage': f"{percentage:.1f}%"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, width='stretch', hide_index=True)
            
            # Check if selected element is detected
            if selected_element in element_counts:
                percentage = (element_counts[selected_element] / len(predictions)) * 100
                if percentage > 80:
                    st.success(f"✅ Strong {selected_element} signature detected ({percentage:.1f}%)")
                else:
                    st.warning(f"⚠️ Moderate {selected_element} detection ({percentage:.1f}%)")
            else:
                st.error(f"❌ No {selected_element} detected in this region")
        else:
            st.error(f"❌ No data found in the range {analysis_min}-{analysis_max} Å")

else:  # Future Eclipse Analysis
    st.header("🔮 Future Eclipse Prediction Engine")
    
    st.success("""
    **🌟 Scientific Discovery Mode:**
    - Upload new eclipse data to detect anomalies
    - Find spectral features not in training data
    - Discover unusual solar activity patterns
    - Identify potential new physics!
    """)
    
    uploaded_file = st.file_uploader("Upload new eclipse spectrum data (CSV with wavelength, intensity columns)", 
                                   type=['csv', 'txt'])
    
    # Button to use sample test data
    if st.button("🎯 Use Sample Test Data", key="sample_data"):
        sample_data = pd.DataFrame({
            'wavelength': [3968.5, 5172.7, 5889.95, 5875.6, 6173.3, 6562.8],
            'intensity': [0.25, 0.92, 0.85, 0.80, 0.88, 0.15]
        })
        
        # Convert to CSV format for processing
        csv_data = sample_data.to_csv(index=False)
        
        # Create a file-like object
        uploaded_file = StringIO(csv_data)
        st.success("✅ Loaded sample test data!")
    
    if uploaded_file:
        try:
            # Read the uploaded file
            new_data = pd.read_csv(uploaded_file)
            
            # Clean the data to ensure Arrow compatibility
            new_data = clean_dataframe(new_data)
            
            # Validate the data
            if not validate_and_analyze_uploaded_data(new_data):
                st.stop()
            
            # Show training data info for reference
            DATA_MIN = df['wavelength'].min()
            DATA_MAX = df['wavelength'].max()
            st.info(f"**Training Data Reference:** Model was trained on {len(df)} samples covering {DATA_MIN:.1f}-{DATA_MAX:.1f} Å")
            
            if st.button("🔍 Analyze for Scientific Discoveries", type="primary"):
                with st.spinner("Scanning for anomalous spectral features..."):
                    try:
                        # Use the fixed anomaly detection
                        anomalies, scores, clusters = detect_spectral_anomalies(new_data)
                        
                        st.success("🎯 Future Eclipse Analysis Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Anomalous Features", f"{np.sum(anomalies == -1)}")
                        with col2:
                            st.metric("Distinct Patterns", f"{len(np.unique(clusters)) if len(clusters) > 0 else 0}")
                        with col3:
                            st.metric("Data Points", f"{len(new_data):,}")
                        
                        # Interpretation of results
                        if np.sum(anomalies == -1) > 0:
                            st.success("""
                            **🎉 Potential Discoveries Detected:**
                            - Unknown spectral lines or features
                            - Unusual ionization states
                            - Possible new solar phenomena
                            - Anomalous abundance patterns
                            """)
                            
                            # Show anomaly locations
                            fig = px.scatter(
                                x=new_data['wavelength'], 
                                y=scores,
                                color=(anomalies == -1),
                                title='Anomaly Detection Results',
                                labels={'x': 'Wavelength (Å)', 'y': 'Anomaly Score'},
                                color_discrete_map={True: 'red', False: 'blue'}
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="orange")
                            st.plotly_chart(fig, width='stretch')
                            
                            # Show cluster information
                            if len(np.unique(clusters)) > 0:
                                st.info(f"**Cluster Analysis:** {len(np.unique(clusters))} distinct anomaly patterns found")
                                
                        else:
                            st.info("""
                            **✅ No significant anomalies detected**
                            - Spectrum matches known patterns well
                            - No unusual features found in this data
                            - Consistent with previous observations
                            """)
                            
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            
# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset Information:**
- 6 key solar elements
- 10Å regions around each spectral line
- 30,006 total data points
- 3964-6568 Å total range
""")

st.sidebar.markdown("---")
st.sidebar.warning("""
**Future Analysis Mode:**
Upload new eclipse data to:
- Detect unknown spectral features
- Find anomalous patterns
- Make new discoveries!
""")

st.markdown("---")
st.caption("""
**Solar Discovery Engine** • Element identification + Anomaly detection • Ready for future eclipses
""")