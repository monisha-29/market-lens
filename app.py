import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Stock Market Information Gain Dashboard",
    layout="wide"
)

st.title("📊 Stock Market Analysis using Information Gain")
st.markdown("Entropy-based feature importance analysis for Tata, Reliance, and Adani stocks.")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("stock_market_dataset.csv")
    return df

df = load_data()

# -----------------------------
# Utility Functions
# -----------------------------
def entropy(y):
    probs = y.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))


def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    weighted_entropy = (
        data.groupby(feature)[target]
        .apply(lambda x: (len(x) / len(data)) * entropy(x))
        .sum()
    )
    return total_entropy - weighted_entropy


def discretize_features(data, features, bins=4):
    data = data.copy()
    for f in features:
        try:
            data[f] = pd.qcut(data[f], q=bins, duplicates='drop')
        except Exception:
            data[f] = pd.cut(data[f], bins)
    return data


def analyze_company(df, company):
    features = ['AveragePrice', 'HighPrice', 'LowPrice', 'Volume']
    target = 'PerformanceLabel'

    temp = df[df['Company'] == company][features + [target]].copy()

    for col in features + [target]:
        temp[col] = pd.to_numeric(temp[col], errors='coerce')

    temp.dropna(inplace=True)
    temp = discretize_features(temp, features)

    ig_scores = {
        f: information_gain(temp, f, target)
        for f in features
    }

    return ig_scores

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("🔍 Select Company")

company = st.sidebar.selectbox(
    "Choose a Company",
    df['Company'].unique()
)

# -----------------------------
# Run Analysis
# -----------------------------
ig_scores = analyze_company(df, company)

ig_df = pd.DataFrame(
    ig_scores.items(),
    columns=["Feature", "Information Gain"]
)

# -----------------------------
# Display Results
# -----------------------------
st.subheader(f"📌 Information Gain for {company}")
st.dataframe(ig_df, use_container_width=True)

# -----------------------------
# Bar Chart
# -----------------------------
st.subheader("📊 Feature Importance (Information Gain)")

fig, ax = plt.subplots()
ax.bar(ig_df["Feature"], ig_df["Information Gain"])
ax.set_xlabel("Feature")
ax.set_ylabel("Information Gain")
ax.set_title(f"Information Gain – {company}")

st.pyplot(fig)

# -----------------------------
# Best Feature Highlight
# -----------------------------
best_feature = ig_df.loc[ig_df["Information Gain"].idxmax()]

st.success(
    f"✅ Most Influential Feature for {company}: **{best_feature['Feature']}** "
    f"(IG = {best_feature['Information Gain']:.4f})"
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🎓 **Academic Project | Machine Learning | Feature Selection using Information Gain**")
