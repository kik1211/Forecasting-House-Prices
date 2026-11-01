# Indian House Price Predictor

Interactive web application for exploring and forecasting residential real estate prices using regression models and Plotly analytical dashboards. Built with Python, Streamlit, and scikit-learn.

---

## Overview

This project provides an interactive dashboard to predict property prices based on square footage and inspect key market dynamics within the Chennai metropolitan housing market dataset.

### Key Capabilities

- **Price Inference:** Real-time property valuation based on built-up square footage using a trained scikit-learn regression model (`indian_house_model.joblib`), accompanied by test-set $R^2$ validation metrics.
- **Price vs. Area Analysis:** Interactive Plotly scatter distributions comparing predicted and actual transaction valuations.
- **Bedroom Distribution:** Breakdown of inventory across 1, 2, 3, and 4+ BHK configurations.
- **Location Analytics:** Geospatial property volume distribution across neighborhood clusters.

---

## Project Structure

```
Forecasting-House-Prices/
├── app.py                     # Streamlit web dashboard and visualization layouts
├── requirements.txt           # Python package dependencies
├── indian_house_model.joblib  # Serialized scikit-learn regression model artifact
├── viz_data.joblib            # Precomputed test evaluation and distribution metrics
└── Indian House Prices.xls    # Source tabular housing dataset
```

---

## Tech Stack

- **Application Framework:** Streamlit 1.29
- **Data Manipulation:** pandas 2.0.3
- **Machine Learning:** scikit-learn 1.3.0, joblib 1.3.2
- **Visualization:** Plotly Express, Matplotlib 3.7.2, Seaborn 0.12.2

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kik1211/Forecasting-House-Prices.git
   cd Forecasting-House-Prices
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser at `http://localhost:8501`.
