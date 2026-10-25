# SuperKart Sales Predictor

Streamlit front end for the SuperKart sales-prediction API.

Set the environment variable `SUPERKART_API_URL` to the URL of the backend API.
Without it, the app calls `http://localhost:7860`. In the local Docker setup,
the frontend container reaches the backend at `http://superkart-full-backend:7860`.

- **Single product:** enter one product-store record and get a predicted
  `Product_Store_Sales_Total`.
- **Batch CSV:** upload a CSV with the 10 model feature columns to get
  predictions for many rows.
