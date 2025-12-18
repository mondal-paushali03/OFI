# 📦 NexGen Logistics – Predictive Delivery Risk Engine

[Streamlit App](https://paushali-nexgen-logistics-risk-engine.streamlit.app/)

NexGen Logistics is a data-driven decision support tool designed to transform logistics from **reactive firefighting** to **predictive, cost-aware management**. By analyzing historical delivery data, warehouse inventory, and route conditions, it identifies delivery risks **before dispatch**.

---

## 🚀 Problem Statement
Traditional logistics often suffer from unexpected delays due to carrier inefficiency, weather, or inventory shortages. These delays lead to:
* **Increased operational costs** due to emergency rerouting.
* **Poor customer satisfaction** scores from missed delivery windows.
* **Inefficient resource allocation** across warehouse hubs.

---

## 🛠 The Solution Architecture
The engine integrates multiple data streams—Orders, Carriers, Routes, and Warehouses—to create a unified **"Risk Profile"** for every shipment.



### 1. Data Fusion
We merge five distinct datasets to gain a 360-degree view of the supply chain:
* **Orders & Delivery:** Historical performance and delay flags.
* **Routes:** Real-time distance and weather impact data.
* **Warehouse:** Inventory pressure (Current Stock vs. Reorder Levels).
* **Costs:** Granular breakdown of fuel, labor, and maintenance.

### 2. Feature Engineering
We calculate critical KPIs that serve as model inputs:
* **Route Risk Score:** A weighted metric combining traffic delay, weather impact, and distance.
* **Inventory Pressure:** Identifying if a warehouse is overextended.
* **Carrier Delay Rate:** A historical reliability score for each partner.

---

## ✨ Key Features
* **📊 Operational Dashboard:** Real-time metrics on orders, deliveries, and warehouse status.
* **📈 Model Comparison:** Evaluates **Logistic Regression vs. Random Forest** to choose the most accurate predictor based on AUC scores.
* **🔮 Interactive Simulator:** Allows logistics managers to input variables (distance, cost, carrier) and see an instant risk probability.
* **💰 Financial Exposure:** Calculates "Expected Delay Cost" to help prioritize high-value shipments.

---

## 💻 Tech Stack
* **Language:** Python 3.9+
* **Frontend:** Streamlit
* **Data Science:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Plotly Express

---

## ⚙️ Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/mondal-paushali03/nexgen-logistics-risk-engine.git](https://github.com/mondal-paushali03/nexgen-logistics-risk-engine.git)
   cd nexgen-logistics-risk-engine
2. **Install dependencies**
   ```bash
   pip install pandas numpy streamlit plotly scikit-learn
3. **Run the Application**
   ```bash
   streamlit run app.py
   
## 🧠 Machine Learning Logic
The system uses a **Random Forest Classifier** as its primary engine (selected dynamically based on the highest AUC score ). 

The predictive engine calculates the risk using the following weighted formula:

$$\text{Route Risk Score} = (0.4 \times \text{Traffic}) + (0.3 \times \text{Weather}) + (0.3 \times \text{Distance})$$

The model evaluates operational inputs to predict a binary outcome:  
* **Delay (1):** High probability of missing the delivery window.  
* **On-Time (0):** High probability of meeting the promised delivery date.



---

## 📩 Contact
I am always looking to improve the engine! If you have suggestions for new features, code optimizations, or partnership ideas, feel free to reach out:

* **Email:** [mondal.paushali384@gmail.com](mailto:mondal.paushali384@gmail.com)
* **GitHub:** [@mondal-paushali03](https://github.com/mondal-paushali03)

---

## ⚖️ License
This project is licensed under the **MIT License**. 
*Copyright (c) 2025 Paushali Mondal*
