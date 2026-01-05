import pandas as pd
import random

data = []

companies = ['Tata', 'Reliance', 'Adani']
years = range(2010, 2025)   # 15 years → 15 * 12 * 3 = 540 rows
months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

base_price = {
    'Tata': 150,
    'Reliance': 300,
    'Adani': 1200
}

base_volume = {
    'Tata': 500000,
    'Reliance': 1000000,
    'Adani': 2000000
}

for company in companies:
    price = base_price[company]
    volume = base_volume[company]

    for year in years:
        for month in months:
            avg_price = price + random.randint(-20, 40)
            high_price = avg_price + random.randint(5, 25)
            low_price = avg_price - random.randint(5, 25)
            vol = volume + random.randint(-100000, 150000)

            # Simple rule-based performance label
            performance = 1 if avg_price > price else 0

            data.append([
                company,
                f"{month}-{year}",
                avg_price,
                high_price,
                low_price,
                vol,
                performance
            ])

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        'Company',
        'Month',
        'AveragePrice',
        'HighPrice',
        'LowPrice',
        'Volume',
        'PerformanceLabel'
    ]
)

# Save to CSV
df.to_csv("stock_market_dataset.csv", index=False)

print("✅ CSV file created: stock_market_dataset.csv")
print("Total rows:", len(df))
