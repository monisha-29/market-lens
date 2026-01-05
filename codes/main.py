import pandas as pd
import numpy as np

# --------------------------------
# Load CSV Dataset
# --------------------------------

df = pd.read_csv("stock_market_dataset.csv")
df.columns = df.columns.str.strip()

# --------------------------------
# Utility Functions
# --------------------------------

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


# --------------------------------
# Analyze One Company
# --------------------------------

def analyze_company(df, company):
    features = ['AveragePrice', 'HighPrice', 'LowPrice', 'Volume']
    target = 'PerformanceLabel'

    temp = df[df['Company'] == company][features + [target]].copy()

    # Ensure numeric
    for col in features + [target]:
        temp[col] = pd.to_numeric(temp[col], errors='coerce')

    temp.dropna(inplace=True)

    # Discretize continuous features
    temp = discretize_features(temp, features)

    ig_scores = {
        f: information_gain(temp, f, target)
        for f in features
    }

    avg_ig = np.mean(list(ig_scores.values()))

    print(f"\nInformation Gain – {company}")
    print("-" * 40)
    for f, score in ig_scores.items():
        print(f"{f:<15}: {score:.4f}")

    return avg_ig


# --------------------------------
# Run Analysis for All Companies
# --------------------------------

companies = df['Company'].unique()
final_scores = {}

for company in companies:
    final_scores[company] = analyze_company(df, company)

# --------------------------------
# Final Ranking
# --------------------------------

print("\nAverage Information Gain Scores")
print("=" * 40)

for company, score in final_scores.items():
    print(f"{company:<10}: {score:.4f}")

best_company = max(final_scores, key=final_scores.get)
print(f"\nBest Stock Based on Information Gain: {best_company}")
