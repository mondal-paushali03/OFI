import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NexGen Logistics – Predictive Risk Engine",
    layout="wide"
)

st.title("📦 NexGen Logistics – Predictive Delivery Risk Engine")
st.markdown("""
**Goal:**  
Move from *reactive firefighting* to **predictive, cost-aware logistics decisions**  
by identifying delivery delay risks **before dispatch** and recommending actions.
""")

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv("data/orders.csv")
    delivery = pd.read_csv("data/delivery_performance.csv")
    routes = pd.read_csv("data/routes_distance.csv")
    warehouse = pd.read_csv("data/warehouse_inventory.csv")
    costs = pd.read_csv("data/cost_breakdown.csv")
    return orders, delivery, routes, warehouse, costs

orders, delivery, routes, warehouse, costs = load_data()

# --------------------------------------------------
# OPERATIONAL SNAPSHOT
# --------------------------------------------------
st.subheader("📊 Operational Snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Orders", len(orders))
c2.metric("Completed Deliveries", len(delivery))
warehouse.rename(columns=lambda x: x.lower(), inplace=True)
c3.metric("Warehouses", warehouse["warehouse_id"].nunique())

st.divider()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
delivery["delay_days"] = (
    delivery["Actual_Delivery_Days"] -
    delivery["Promised_Delivery_Days"]
)
delivery["delay_flag"] = (delivery["delay_days"] > 0).astype(int)

delivery.rename(columns={
    "Order_ID": "order_id",
    "Carrier": "carrier_id",
    "Customer_Rating": "customer_rating"
}, inplace=True)

orders.rename(columns=lambda x: x.lower(), inplace=True)
routes.rename(columns=lambda x: x.lower(), inplace=True)
warehouse.rename(columns=lambda x: x.lower(), inplace=True)
costs.rename(columns=lambda x: x.lower(), inplace=True)

carrier_perf = (
    delivery.groupby("carrier_id")
    .agg(carrier_delay_rate=("delay_flag", "mean"))
    .reset_index()
)

weather_map = {"Light_Rain": 0.3, "Fog": 0.6, "Heavy_Rain": 1.0}
routes["weather_impact_num"] = routes["weather_impact"].map(weather_map).fillna(0.5)

routes["route_risk_score"] = (
    0.4 * (routes["traffic_delay_minutes"] / routes["traffic_delay_minutes"].max()) +
    0.3 * routes["weather_impact_num"] +
    0.3 * (routes["distance_km"] / routes["distance_km"].max())
)

warehouse["inventory_pressure"] = (
    warehouse["current_stock_units"] / warehouse["reorder_level"]
)

costs["total_cost"] = (
    costs["fuel_cost"] +
    costs["labor_cost"] +
    costs["vehicle_maintenance"] +
    costs["insurance"] +
    costs["packaging_cost"] +
    costs["technology_platform_fee"] +
    costs["other_overhead"]
)

# --------------------------------------------------
# MERGE MASTER DATASET
# --------------------------------------------------
df = orders.merge(delivery, on="order_id", how="left")
df = df.merge(routes[["order_id", "route_risk_score", "distance_km"]], on="order_id", how="left")
df = df.merge(costs[["order_id", "total_cost"]], on="order_id", how="left")
df = df.merge(carrier_perf, on="carrier_id", how="left")
df = df.merge(
    warehouse[["location", "product_category", "inventory_pressure"]],
    left_on=["origin", "product_category"],
    right_on=["location", "product_category"],
    how="left"
)

# --------------------------------------------------
# MODELING
# --------------------------------------------------
features = [
    "route_risk_score",
    "carrier_delay_rate",
    "inventory_pressure",
    "distance_km",
    "total_cost"
]

df_model = df[features + ["delay_flag"]].copy()
df_model.fillna(df_model.median(numeric_only=True), inplace=True)
df_model.dropna(subset=["delay_flag"], inplace=True)

X = df_model[features]
y = df_model["delay_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:, 1])

rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

best_model = rf_model if rf_auc > log_auc else log_model

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.subheader("📈 Model Performance")
st.write(f"Logistic Regression AUC: **{log_auc:.3f}**")
st.write(f"Random Forest AUC: **{rf_auc:.3f}**")
st.success(f"Selected Model: **{type(best_model).__name__}**")

st.divider()

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
st.subheader("Feature Importance")

fi = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

st.plotly_chart(
    px.bar(fi, x="Importance", y="Feature", orientation="h"),
    use_container_width=True
)
st.caption("Shows how strongly each feature influences the model’s delay prediction (higher means greater impact).")

# --------------------------------------------------
# RISK & COST INSIGHTS
# --------------------------------------------------
st.subheader("📈 Risk & Cost Insights")

c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(
        px.histogram(
            df_model,
            x="route_risk_score",
            color="delay_flag",
            nbins=20,
            title="Route Risk Distribution vs Delay"
        ),
        use_container_width=True
    )
    st.caption("Compares route risk scores for delayed versus on-time deliveries to reveal risk separation.")

with c2:
    st.plotly_chart(
        px.box(
            df_model,
            x="delay_flag",
            y="total_cost",
            title="Cost Distribution: On-Time vs Delayed",
            labels={"delay_flag": "Delayed (0=No, 1=Yes)"}
        ),
        use_container_width=True
    )
    st.caption("Shows how total delivery costs differ between delayed and on-time orders.")

# --------------------------------------------------
# CARRIER PERFORMANCE
# --------------------------------------------------
st.subheader("🚚 Carrier-wise Delay Risk")

carrier_view = (
    df.groupby("carrier_id", as_index=False)
    .agg(delay_rate=("delay_flag", "mean"))
    .sort_values("delay_rate", ascending=False)
)

st.plotly_chart(
    px.bar(
        carrier_view,
        x="carrier_id",
        y="delay_rate",
        color="delay_rate",
        color_continuous_scale="Reds",
        title="Delay Rate by Carrier"
    ),
    use_container_width=True
)
st.caption("Displays average delay rate per carrier to identify high-risk logistics partners.")

st.divider()

# --------------------------------------------------
# PREDICTIVE SIMULATOR
# --------------------------------------------------
st.subheader("Predictive Risk Simulator")

with st.sidebar:
    st.header("Simulation Inputs")
    route_risk = st.slider("Route Risk Score", 0.0, 1.0, 0.3)
    carrier_rate = st.slider("Carrier Delay Rate", 0.0, 1.0, 0.2)
    inventory_pressure = st.slider("Inventory Pressure", 0.5, 2.0, 1.0)
    distance = st.number_input("Distance (km)", 50, 3000, 200)
    cost = st.number_input("Total Cost (₹)", 500, 50000, 8000)

input_df = pd.DataFrame([{
    "route_risk_score": route_risk,
    "carrier_delay_rate": carrier_rate,
    "inventory_pressure": inventory_pressure,
    "distance_km": distance,
    "total_cost": cost
}])

if best_model == log_model:
    risk_prob = log_model.predict_proba(scaler.transform(input_df))[0][1]
else:
    risk_prob = rf_model.predict_proba(input_df)[0][1]

risk_prob = float(np.clip(risk_prob, 0.05, 0.95))
expected_cost = risk_prob * cost

st.metric("Predicted Delay Risk", f"{risk_prob*100:.1f}%")
st.metric("Expected Delay Cost", f"₹{expected_cost:,.0f}")
st.caption("Estimated delay probability and financial exposure based on simulated operational conditions.")

if risk_prob > 0.7:
    st.error("⚠️ High Risk – Upgrade carrier or reroute recommended")
elif risk_prob > 0.4:
    st.warning("⚠️ Medium Risk – Monitor closely")
else:
    st.success("✅ Low Risk – Proceed as planned")
