# MarketMind AI

MarketMind AI is an AI-assisted business intelligence dashboard built with Streamlit and the Olist Brazilian E-Commerce dataset.

## Features
- Executive KPIs: total orders, GMV, delivered GMV, delivered AOV, delivery rate, customers and average review rating.
- Sales analytics: monthly GMV trends and top product categories.
- Customer geography: order distribution by state.
- Delivery and review analysis.
- Exploratory three-month sales forecasting using Random Forest.
- Gemini-powered natural-language business explanations based on calculated analytical results.

## Project structure
```text
MarketMind_AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── raw/
        ├── olist_customers_dataset.csv
        ├── olist_orders_dataset.csv
        ├── olist_order_items_dataset.csv
        ├── olist_order_reviews_dataset.csv
        ├── olist_products_dataset.csv
        └── product_category_name_translation.csv
```

## Run locally
1. Create and activate a Python environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Place the six required CSV files in `data/raw/`.
4. Configure your Gemini key (optional for non-AI dashboard features):
   - Windows PowerShell: `$env:GEMINI_API_KEY="YOUR_NEW_KEY"`
   - Or create `.streamlit/secrets.toml` containing:
     `GEMINI_API_KEY = "YOUR_NEW_KEY"`
5. Run:
   `streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud and create an app from that repository.
3. Select `app.py` as the entry point.
4. Add `GEMINI_API_KEY` in the app's Secrets settings.
5. Deploy.

## Data and interpretation notes
- GMV is calculated as item price plus freight value. It is not profit or accounting net revenue.
- The forecasting model is an exploratory demonstration on historical Olist data, not a live forecast.
- Keep API keys out of notebooks, source files, screenshots and GitHub. Store secrets in Streamlit's Secrets settings or environment variables.
