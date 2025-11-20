
# Real Estate Price Prediction - Streamlit Demo

This is a ready-to-run demo project (synthetic dataset) that includes:
- `data/sample_houses.csv` - synthetic sample dataset
- `train.py` - training script that trains an XGBoost model and saves it to `models/house_price_model.joblib`
- `app.py` - Streamlit app that loads the saved model and predicts prices and shows comparables on a map
- `requirements.txt` - Python dependencies
- `data/locality_coords.json` - mapping of localities to coordinates

## Quick start (local)
1. Create a virtual environment (recommended) and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Train the model (will produce `models/house_price_model.joblib`):
   ```bash
   python train.py
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes
- This uses a synthetic dataset for demo purposes. Replace `data/sample_houses.csv` with a real dataset (e.g., Bengaluru house prices or Ames dataset) and adjust columns if needed.
- The training script produces a pipeline which includes preprocessing; the Streamlit app expects the saved pipeline.
- To use geocoding for real localities, replace `data/locality_coords.json` with accurate lat/lon pairs.

