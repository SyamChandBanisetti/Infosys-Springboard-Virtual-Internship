import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Titanic Data Analysis", layout="wide")
sns.set_style("whitegrid")

# Title
st.title("🚢 Titanic Dataset Analysis")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("titanic.csv")
    return data

df = load_data()

# Sidebar filters
st.sidebar.header("⚙️ Filter & Customize")

# Filter by Passenger Class
pclass_filter = st.sidebar.multiselect(
    "Select Passenger Class(es):",
    options=sorted(df["Pclass"].unique()),
    default=sorted(df["Pclass"].unique())
)

# Filter by Survival
survival_filter = st.sidebar.multiselect(
    "Select Survival Status:",
    options=[0, 1],
    format_func=lambda x: "✅ Survived" if x == 1 else "❌ Did not survive",
    default=[0, 1]
)

# Filter by Gender
if "Sex" in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Select Gender:",
        options=df["Sex"].unique(),
        default=list(df["Sex"].unique())
    )
else:
    gender_filter = df["Sex"].unique()

# Apply filters
filtered_df = df[
    (df["Pclass"].isin(pclass_filter)) &
    (df["Survived"].isin(survival_filter)) &
    (df["Sex"].isin(gender_filter))
]

# Data Preview
if st.sidebar.checkbox("Show Filtered Data Preview", value=True):
    st.subheader("📋 Filtered Data Preview")
    st.write(filtered_df.head())
else:
    st.subheader("📋 Original Data Preview")
    st.write(df.head())

# Summary statistics
if st.sidebar.checkbox("Show Data Summary Statistics", value=True):
    st.subheader("📊 Summary Statistics")
    st.write(filtered_df.describe(include="all"))

# Survival Count plot
st.subheader("🧮 Survival Count")
fig1, ax1 = plt.subplots()
sns.countplot(x="Survived", data=filtered_df, palette="Set2", ax=ax1)
ax1.set_xticklabels(["Did not survive", "Survived"])
st.pyplot(fig1)

# Numerical column histogram
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.to_list()
hist_col = st.sidebar.selectbox("Select column for histogram:", numerical_columns, index=numerical_columns.index("Age"))

st.subheader(f"📈 {hist_col} Distribution")
show_kde = st.sidebar.checkbox("Show KDE plot", value=True)
bins = st.sidebar.slider("Number of bins:", min_value=10, max_value=100, value=30, step=5)
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df[hist_col].dropna(), kde=show_kde, bins=bins, color="skyblue", ax=ax2)
st.pyplot(fig2)

# Survival by Passenger Class
st.subheader("🚻 Survival by Passenger Class")
fig3, ax3 = plt.subplots()
sns.countplot(x="Pclass", hue="Survived", data=filtered_df, palette="Set1", ax=ax3)
ax3.legend(title="Survived", labels=["No", "Yes"])
st.pyplot(fig3)

# Correlation Heatmap
st.subheader("🔗 Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax4, fmt=".2f")
st.pyplot(fig4)
