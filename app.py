import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("👥 Customer Segmentation using K-Means")
st.markdown(
    "Analyze customer demographics, income, and spending behavior "
    "using K-Means Clustering."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/Mall_Customers.csv")


df = load_data()

# --------------------------------------------------
# K-Means Clustering
# --------------------------------------------------

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# --------------------------------------------------
# Customer Segment Names
# --------------------------------------------------

segment_names = {
    0: "Average Customers",
    1: "High-Value Customers",
    2: "Young High-Spending Customers",
    3: "High-Income Low-Spending Customers",
    4: "Low-Income Low-Spending Customers"
}

df["Customer Segment"] = df["Cluster"].map(segment_names)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("🔎 Filters")

selected_segment = st.sidebar.selectbox(
    "Select Customer Segment",
    ["All"] + sorted(df["Customer Segment"].unique().tolist())
)

if selected_segment != "All":
    filtered_df = df[df["Customer Segment"] == selected_segment]
else:
    filtered_df = df

# --------------------------------------------------
# Key Metrics
# --------------------------------------------------

st.subheader("📊 Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(filtered_df))
col2.metric(
    "Average Age",
    f"{filtered_df['Age'].mean():.1f}"
)
col3.metric(
    "Avg. Annual Income",
    f"${filtered_df['Annual Income (k$)'].mean():.1f}k"
)
col4.metric(
    "Avg. Spending Score",
    f"{filtered_df['Spending Score (1-100)'].mean():.1f}"
)

# --------------------------------------------------
# Segment Distribution
# --------------------------------------------------

st.subheader("📊 Customer Segment Distribution")

segment_counts = (
    filtered_df["Customer Segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["Customer Segment", "Customers"]

fig1, ax1 = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=segment_counts,
    x="Customer Segment",
    y="Customers",
    ax=ax1
)

ax1.set_xlabel("Customer Segment")
ax1.set_ylabel("Number of Customers")
ax1.tick_params(axis="x", rotation=25)

plt.tight_layout()

st.pyplot(fig1)

# --------------------------------------------------
# Income vs Spending Score
# --------------------------------------------------

st.subheader("💰 Annual Income vs Spending Score")

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=filtered_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Customer Segment",
    style="Genre",
    s=100,
    ax=ax2
)

ax2.set_title("Annual Income vs Spending Score")

plt.tight_layout()

st.pyplot(fig2)

# --------------------------------------------------
# Gender Distribution by Segment
# --------------------------------------------------

st.subheader("🚻 Gender Distribution by Segment")

fig3, ax3 = plt.subplots(figsize=(12, 6))

sns.countplot(
    data=filtered_df,
    x="Customer Segment",
    hue="Genre",
    ax=ax3
)

ax3.set_xlabel("Customer Segment")
ax3.set_ylabel("Number of Customers")
ax3.tick_params(axis="x", rotation=25)

plt.tight_layout()

st.pyplot(fig3)

# --------------------------------------------------
# Cluster Analysis
# --------------------------------------------------

st.subheader("🔍 Customer Segment Analysis")

cluster_summary = (
    df.groupby("Cluster")[
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ]
    .mean()
    .round(2)
)

cluster_summary["Customer Segment"] = (
    cluster_summary.index.map(segment_names)
)

cluster_summary = cluster_summary[
    [
        "Customer Segment",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

st.dataframe(
    cluster_summary,
    use_container_width=True
)

# --------------------------------------------------
# Customer Data
# --------------------------------------------------

st.subheader("👥 Customer Data")

display_columns = [
    "CustomerID",
    "Genre",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Customer Segment"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)

# --------------------------------------------------
# Business Insights
# --------------------------------------------------

st.subheader("💡 Business Insights")

st.markdown("""
### High-Value Customers
Customers with relatively high income and high spending scores.
Businesses can target them with premium products, loyalty programs,
and personalized offers.

### High-Income Low-Spending Customers
These customers have strong purchasing power but relatively low spending.
Special promotions and personalized recommendations may encourage
higher engagement.

### Low-Income Low-Spending Customers
These customers have lower income and lower spending.
Budget-friendly products and discounts may be more effective.

### Young High-Spending Customers
Younger customers with high spending scores.
Social media campaigns, trends, and personalized promotions can help
engage this group.

### Average Customers
Customers with relatively balanced income and spending behavior.
Businesses can use regular promotions and loyalty programs to retain
these customers.
""")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "**Customer Segmentation using K-Means | "
    "Developed by Sanjay Kumar Bejjanki**"
)