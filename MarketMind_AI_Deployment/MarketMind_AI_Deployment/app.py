import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="MarketMind AI", page_icon="📊", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

@st.cache_data
def load_data():
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files in data/raw: " + ", ".join(missing)
        )
    data = {name: pd.read_csv(os.path.join(DATA_DIR, filename))
            for name, filename in FILES.items()}
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in date_cols:
        if col in data["orders"].columns:
            data["orders"][col] = pd.to_datetime(data["orders"][col], errors="coerce")
    for col in ["review_creation_date", "review_answer_timestamp"]:
        if col in data["reviews"].columns:
            data["reviews"][col] = pd.to_datetime(data["reviews"][col], errors="coerce")
    return data

@st.cache_data
def prepare_analysis():
    data = load_data()
    orders = data["orders"].copy()
    items = data["items"].copy()
    customers = data["customers"].copy()
    products = data["products"].copy()
    reviews = data["reviews"].copy()
    translation = data["translation"].copy()

    items["item_value"] = items["price"].fillna(0) + items["freight_value"].fillna(0)
    order_values = items.groupby("order_id").agg(
        order_value=("item_value", "sum"),
        units=("order_item_id", "count")
    ).reset_index()
    sales = orders.merge(order_values, on="order_id", how="left")
    sales["order_value"] = sales["order_value"].fillna(0)
    sales["units"] = sales["units"].fillna(0)

    category_data = items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id", how="left"
    ).merge(translation, on="product_category_name", how="left")
    category_data["category"] = (
        category_data.get("product_category_name_english", pd.Series(index=category_data.index))
        .fillna(category_data["product_category_name"])
        .fillna("unknown")
    )
    category_analysis = category_data.groupby("category").agg(
        revenue=("item_value", "sum"),
        units_sold=("order_item_id", "count")
    ).reset_index().sort_values("revenue", ascending=False)

    monthly = sales.dropna(subset=["order_purchase_timestamp"]).copy()
    monthly["month"] = monthly["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly_sales = monthly.groupby("month").agg(
        revenue=("order_value", "sum"),
        orders=("order_id", "nunique")
    ).reset_index().sort_values("month")
    monthly_sales["growth_percentage"] = monthly_sales["revenue"].pct_change() * 100

    customer_orders = sales.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id", how="left"
    )
    state_analysis = customer_orders.groupby("customer_state").agg(
        orders=("order_id", "nunique"),
        customers=("customer_unique_id", "nunique")
    ).reset_index().sort_values("orders", ascending=False)

    review_analysis = reviews.groupby("review_score").size().reset_index(name="review_count")
    average_rating = float(reviews["review_score"].mean())

    delivery = sales[
        sales["order_delivered_customer_date"].notna() &
        sales["order_estimated_delivery_date"].notna() &
        sales["order_purchase_timestamp"].notna()
    ].copy()
    delivery["delivery_days"] = (
        delivery["order_delivered_customer_date"] -
        delivery["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivery["is_delayed"] = (
        delivery["order_delivered_customer_date"] >
        delivery["order_estimated_delivery_date"]
    )
    delivery_metrics = {
        "Average Delivery Days": float(delivery["delivery_days"].mean()) if len(delivery) else 0,
        "Delayed Orders": int(delivery["is_delayed"].sum()),
        "Delayed Order Percentage": float(delivery["is_delayed"].mean() * 100) if len(delivery) else 0
    }

    delivered = sales[sales["order_status"] == "delivered"]
    total_orders = int(sales["order_id"].nunique())
    delivered_orders = int(delivered["order_id"].nunique())
    delivered_gmv = float(delivered["order_value"].sum())
    kpis = {
        "Total Orders": total_orders,
        "Total GMV": float(sales["order_value"].sum()),
        "Delivered GMV": delivered_gmv,
        "Delivered AOV": delivered_gmv / delivered_orders if delivered_orders else 0,
        "Delivered Order Percentage": delivered_orders / total_orders * 100 if total_orders else 0,
        "Unique Customers": int(customers["customer_unique_id"].nunique()),
        "Average Review Rating": average_rating,
    }
    return {
        "data": data, "sales": sales, "kpis": kpis,
        "category": category_analysis, "monthly": monthly_sales,
        "states": state_analysis, "reviews": review_analysis,
        "average_rating": average_rating, "delivery": delivery,
        "delivery_metrics": delivery_metrics
    }

def get_gemini_client():
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    key = key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)

def ask_marketmind(question, analysis_type, results):
    client = get_gemini_client()
    if client is None:
        return ("Gemini is not configured. Add GEMINI_API_KEY to Streamlit secrets "
                "or the environment to enable AI-generated explanations.")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    prompt = f"""
You are MarketMind AI, a business intelligence analyst.
Answer the user's business question using ONLY the supplied calculated data.
Do not invent figures. Distinguish observed facts from hypotheses.
Give a concise response with: Direct Answer, Key Findings, Business Interpretation,
and Practical Recommendations.

Question: {question}
Analysis category: {analysis_type}
Calculated data: {json.dumps(results, default=str)}
"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def forecast_sales(monthly):
    fd = monthly[["month", "revenue"]].rename(columns={"revenue": "sales"}).copy()
    fd = fd.sort_values("month").set_index("month").asfreq("MS", fill_value=0).reset_index()
    if len(fd) < 8:
        return None, None, "Not enough monthly observations to train a forecast model."
    fd["lag_1"] = fd["sales"].shift(1)
    fd["lag_2"] = fd["sales"].shift(2)
    fd["lag_3"] = fd["sales"].shift(3)
    fd["rolling_mean_3"] = fd["sales"].shift(1).rolling(3).mean()
    fd["month_number"] = fd["month"].dt.month
    model_data = fd.dropna().copy()
    features = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "month_number"]
    if len(model_data) < 6:
        return None, None, "Not enough rows after feature engineering."
    test_n = min(3, max(1, len(model_data)//5))
    X, y = model_data[features], model_data["sales"]
    model = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=2, random_state=42
    )
    model.fit(X.iloc[:-test_n], y.iloc[:-test_n])
    pred = model.predict(X.iloc[-test_n:])
    comparison = pd.DataFrame({
        "Month": model_data["month"].iloc[-test_n:].values,
        "Actual Sales": y.iloc[-test_n:].values,
        "Predicted Sales": pred
    })
    mae = mean_absolute_error(comparison["Actual Sales"], comparison["Predicted Sales"])
    rmse = np.sqrt(mean_squared_error(comparison["Actual Sales"], comparison["Predicted Sales"]))
    # Refit on all available historical observations for future predictions
    model.fit(X, y)
    history = fd["sales"].tolist()
    last_date = fd["month"].max()
    future = []
    for i in range(1, 4):
        next_month = last_date + pd.DateOffset(months=i)
        row = pd.DataFrame([{
            "lag_1": history[-1], "lag_2": history[-2], "lag_3": history[-3],
            "rolling_mean_3": float(np.mean(history[-3:])),
            "month_number": next_month.month
        }])
        value = max(0, float(model.predict(row[features])[0]))
        future.append({"month": next_month, "predicted_sales": value})
        history.append(value)
    return comparison, pd.DataFrame(future), {"MAE": mae, "RMSE": rmse}

try:
    result = prepare_analysis()
except Exception as exc:
    st.error(f"Could not load the Olist data: {exc}")
    st.info("Place all six required CSV files inside the project's data/raw folder.")
    st.stop()

st.title("📊 MarketMind AI")
st.caption("AI-powered e-commerce business intelligence | Olist Brazilian E-Commerce Dataset")

nav = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Sales Analytics", "Customer Analytics",
     "Delivery & Reviews", "Sales Forecasting", "AI Business Analyst"]
)

k = result["kpis"]
money = lambda x: f"R$ {x:,.2f}"

if nav == "Executive Overview":
    st.subheader("Business Performance at a Glance")
    cols = st.columns(4)
    metrics = [
        ("Total Orders", f'{k["Total Orders"]:,}'),
        ("Total GMV", money(k["Total GMV"])),
        ("Delivered AOV", money(k["Delivered AOV"])),
        ("Delivered Orders", f'{k["Delivered Order Percentage"]:.1f}%')
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    cols2 = st.columns(3)
    cols2[0].metric("Delivered GMV", money(k["Delivered GMV"]))
    cols2[1].metric("Unique Customers", f'{k["Unique Customers"]:,}')
    cols2[2].metric("Average Review Rating", f'{k["Average Review Rating"]:.2f}/5')
    st.plotly_chart(px.line(result["monthly"], x="month", y="revenue",
                            title="Monthly GMV Trend", markers=True), use_container_width=True)
    st.caption("GMV represents item price plus freight value; it is not net profit.")

elif nav == "Sales Analytics":
    st.subheader("Sales & Product Performance")
    st.plotly_chart(px.bar(result["category"].head(10), x="revenue", y="category",
                           orientation="h", title="Top 10 Categories by GMV",
                           labels={"revenue": "GMV", "category": "Category"}),
                    use_container_width=True)
    st.dataframe(result["category"].head(15), use_container_width=True)
    st.plotly_chart(px.line(result["monthly"], x="month", y="revenue",
                            title="Monthly GMV", markers=True), use_container_width=True)
    st.dataframe(result["monthly"].tail(12), use_container_width=True)

elif nav == "Customer Analytics":
    st.subheader("Customer Geography")
    top_states = result["states"].head(10)
    st.plotly_chart(px.bar(top_states, x="customer_state", y="orders",
                           title="Top 10 States by Order Volume",
                           labels={"customer_state": "State", "orders": "Orders"}),
                    use_container_width=True)
    st.dataframe(top_states, use_container_width=True)

elif nav == "Delivery & Reviews":
    st.subheader("Delivery Performance")
    dm = result["delivery_metrics"]
    a, b, c = st.columns(3)
    a.metric("Average Delivery Time", f'{dm["Average Delivery Days"]:.1f} days')
    b.metric("Delayed Orders", f'{dm["Delayed Orders"]:,}')
    c.metric("Delayed Order Rate", f'{dm["Delayed Order Percentage"]:.1f}%')
    st.plotly_chart(px.histogram(result["delivery"], x="delivery_days", nbins=40,
                                 title="Distribution of Delivery Time"),
                    use_container_width=True)
    st.subheader("Customer Reviews")
    st.metric("Average Review Score", f'{result["average_rating"]:.2f}/5')
    st.plotly_chart(px.bar(result["reviews"], x="review_score", y="review_count",
                           title="Review Score Distribution"), use_container_width=True)

elif nav == "Sales Forecasting":
    st.subheader("Three-Month Sales Forecast")
    comparison, future, evaluation = forecast_sales(result["monthly"])
    if comparison is None:
        st.warning(evaluation)
    else:
        st.caption("Exploratory model based on the historical Olist dataset; not a live-market forecast.")
        a, b = st.columns(2)
        a.metric("Validation MAE", money(evaluation["MAE"]))
        b.metric("Validation RMSE", money(evaluation["RMSE"]))
        st.write("Holdout comparison")
        st.dataframe(comparison, use_container_width=True)
        st.write("Future estimates")
        st.dataframe(future, use_container_width=True)
        hist = result["monthly"][["month", "revenue"]].rename(columns={"revenue": "value"})
        hist["series"] = "Historical GMV"
        fut = future.rename(columns={"predicted_sales": "value"})
        fut["series"] = "Forecast GMV"
        chart_data = pd.concat([hist[["month", "value", "series"]],
                                fut[["month", "value", "series"]]], ignore_index=True)
        st.plotly_chart(px.line(chart_data, x="month", y="value", color="series",
                                markers=True, title="Historical GMV and Three-Month Forecast"),
                        use_container_width=True)

elif nav == "AI Business Analyst":
    st.subheader("Ask MarketMind AI")
    st.write("Ask a question about sales, products, customers, delivery, reviews or overall KPIs.")
    examples = [
        "What is the overall business performance?",
        "Which product categories generate the highest GMV?",
        "How are monthly sales changing?",
        "Which states have the highest customer concentration?",
        "How satisfied are customers?",
        "Are deliveries frequently delayed?"
    ]
    question = st.selectbox("Choose a sample question", [""] + examples)
    custom = st.text_input("Or enter your own business question")
    if st.button("Analyse", type="primary"):
        q = custom.strip() or question.strip()
        if not q:
            st.warning("Enter or select a question first.")
        else:
            # Route using simple keywords first, then Gemini for ambiguous questions.
            ql = q.lower()
            if any(x in ql for x in ["categor", "product", "item"]):
                kind, payload = "categories", result["category"].head(10).to_dict("records")
            elif any(x in ql for x in ["month", "trend", "growth", "sales"]):
                kind, payload = "monthly", result["monthly"].tail(12).to_dict("records")
            elif any(x in ql for x in ["state", "customer", "location"]):
                kind, payload = "customers", result["states"].head(10).to_dict("records")
            elif any(x in ql for x in ["review", "rating", "satisf"]):
                kind, payload = "reviews", {
                    "average_rating": result["average_rating"],
                    "distribution": result["reviews"].to_dict("records")}
            elif any(x in ql for x in ["deliver", "delay", "shipping"]):
                kind, payload = "delivery", result["delivery_metrics"]
            else:
                kind, payload = "kpis", k
            with st.spinner("MarketMind AI is analysing the calculated business data..."):
                try:
                    answer = ask_marketmind(q, kind, payload)
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"AI request failed: {exc}")
                    st.info("Check the Gemini API key and model setting in Streamlit secrets.")
