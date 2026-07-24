import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("SuperStore_Sales_Dataset.csv")

df.rename(columns={"Row ID+O6G3A1:R6":"Row ID"}, inplace=True)

for col in ["Returns","ind1","ind2"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
model = joblib.load("sales_model.pkl")

df["Month"] = df["Order Date"].dt.month
df["Year"] = df["Order Date"].dt.year

# ---------------- TITLE ----------------
st.markdown(
"""
<h1 style='text-align:center;color:#4CAF50;'>
📊 Sales Forecasting Dashboard
</h1>
""",
unsafe_allow_html=True
)

st.write("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Filters")

category = st.sidebar.multiselect(
    "Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

region = st.sidebar.multiselect(
    "Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

segment = st.sidebar.multiselect(
    "Segment",
    df["Segment"].unique(),
    default=df["Segment"].unique()
)

state = st.sidebar.multiselect(
    "State",
    df["State"].unique(),
    default=df["State"].unique()
)

payment = st.sidebar.multiselect(
    "Payment Mode",
    df["Payment Mode"].unique(),
    default=df["Payment Mode"].unique()
)

# ---------------- FILTER ----------------
df = df[
(df["Category"].isin(category)) &
(df["Region"].isin(region)) &
(df["Segment"].isin(segment)) &
(df["State"].isin(state)) &
(df["Payment Mode"].isin(payment))
]

# ---------------- KPI ----------------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
orders = df["Order ID"].nunique()
qty = df["Quantity"].sum()

c1,c2,c3,c4 = st.columns(4)

c1.metric("💰 Total Sales",f"${total_sales:,.2f}")
c2.metric("📈 Total Profit",f"${total_profit:,.2f}")
c3.metric("📦 Orders",orders)
c4.metric("🛒 Quantity",qty)

st.write("---")

# ---------------- MONTHLY SALES ----------------
monthly = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
monthly["Order Date"]=monthly["Order Date"].astype(str)

fig1=px.line(
monthly,
x="Order Date",
y="Sales",
markers=True,
title="📈 Monthly Sales Trend"
)

st.plotly_chart(fig1,use_container_width=True)

# ---------------- CATEGORY + REGION ----------------
left,right=st.columns(2)

cat=df.groupby("Category")["Sales"].sum().reset_index()

fig2=px.bar(
cat,
x="Category",
y="Sales",
color="Category",
title="Sales by Category"
)

left.plotly_chart(fig2,use_container_width=True)

reg=df.groupby("Region")["Sales"].sum().reset_index()

fig3=px.pie(
reg,
names="Region",
values="Sales",
title="Sales by Region"
)

right.plotly_chart(fig3,use_container_width=True)

# ---------------- TOP 10 STATES ----------------
st.write("---")
st.subheader("🏆 Top 10 States by Sales")

state_sales = (
    df.groupby("State")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig4 = px.bar(
    state_sales,
    x="State",
    y="Sales",
    color="Sales",
    text_auto=".2s",
    title="Top 10 States"
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------- PROFIT BY CATEGORY ----------------
st.write("---")
st.subheader("💰 Profit by Category")

profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .reset_index()
)

fig5 = px.bar(
    profit,
    x="Category",
    y="Profit",
    color="Category",
    text_auto=".2s"
)

st.plotly_chart(fig5, use_container_width=True)

# ---------------- PAYMENT MODE ----------------
st.write("---")
left,right = st.columns(2)

payment = (
    df.groupby("Payment Mode")["Sales"]
    .sum()
    .reset_index()
)

fig6 = px.pie(
    payment,
    names="Payment Mode",
    values="Sales",
    hole=0.5,
    title="Payment Mode Analysis"
)

left.plotly_chart(fig6, use_container_width=True)

# ---------------- TOP PRODUCTS ----------------
products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig7 = px.bar(
    products,
    x="Sales",
    y="Product Name",
    orientation="h",
    color="Sales",
    title="Top 10 Products"
)

right.plotly_chart(fig7, use_container_width=True)

# ---------------- DATA TABLE ----------------
st.write("---")
st.subheader("📄 Sales Data")

st.dataframe(df)

# ---------------- DOWNLOAD BUTTON ----------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="Filtered_Sales_Data.csv",
    mime="text/csv"
)
st.write("---")
st.header("🤖 Sales Prediction")

col1, col2 = st.columns(2)

with col1:
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=20,
        value=2
    )

    profit = st.number_input(
        "Profit",
        value=100.0
    )

with col2:
    month = st.selectbox(
        "Month",
        list(range(1, 13))
    )

    year = st.selectbox(
        "Year",
        [2019, 2020, 2021]
    )

if st.button("Predict Sales"):

    prediction = model.predict([[quantity, profit, month, year]])

    st.success(
        f"💰 Predicted Sales : ${prediction[0]:.2f}"
    )

# ---------------- FOOTER ----------------
st.write("---")

st.markdown(
"""
<div style='text-align:center;'>

### ✅ Sales Forecasting Dashboard

Developed by **Abhinav Yadav**

Python | Streamlit | Plotly | Machine Learning

</div>
""",
unsafe_allow_html=True
)