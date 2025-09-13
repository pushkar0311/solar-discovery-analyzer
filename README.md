# ☀️ Solar Discovery Analyzer

AI-powered spectrum analyzer for solar science.  
Identify key solar elements, detect anomalies, and explore **future eclipse predictions** using machine learning.  
Built with **Python, Streamlit, and Random Forest** to make solar exploration interactive and insightful.  

---

## 🚀 Features
- 🔬 **Spectral Range Analysis** – Explore wavelength ranges & detect solar elements  
- 🎯 **Element-Specific Insights** – Focus on individual elements (Hydrogen, Helium, Calcium, etc.)  
- 🔮 **Future Eclipse Prediction Engine** – Upload new eclipse spectra to find anomalies & potential new discoveries  
- 📊 **Interactive Visuals** – Pie charts, anomaly scatter plots, and confidence metrics  
- 🧠 **Machine Learning** – Trained on key spectral lines with Random Forest Classifier  

---

## 📂 Dataset
- Covers **6 major solar elements**: Hydrogen, Helium, Calcium, Magnesium, Sodium, Iron  
- Spectral ranges: **3964–6568 Å**  
- ~30,000 total data points  

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **Streamlit** (web interface)  
- **Pandas & NumPy** (data handling)  
- **Plotly** (interactive charts)  
- **Scikit-learn** (Random Forest ML model)  

---

## 📸 Screenshots
*(Add screenshots of your Streamlit app output here – analysis tables, anomaly plots, etc.)*

---

## ⚡ How to Run
```bash
# Clone the repo
git clone https://github.com/<your-username>/solar-discovery-analyzer.git
cd solar-discovery-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
