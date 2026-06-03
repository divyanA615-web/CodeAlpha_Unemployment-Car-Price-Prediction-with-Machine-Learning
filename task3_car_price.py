# ============================================================
# TASK 3: CAR PRICE PREDICTION — EDA + REGRESSION
# ============================================================
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # type: ignore

# ── 1. LOAD ──────────────────────────────────────────────────
df = pd.read_csv("D:\data science related\CodeAlpha_Internship\CodeAlpha_Unemployment-Car-Price-Prediction-with-Machine-Learning\car_data.csv") # type: ignore
print(f"Shape: {df.shape}")
print(df.head())
print(f"\nMissing:\n{df.isnull().sum()}")
print(f"\nDescriptive Stats:\n{df.describe().round(2)}")

# ── 2. FEATURE ENGINEERING ───────────────────────────────────
df['Car_Age'] = 2024 - df['Year']
df.drop(columns=['Car_Name', 'Year'], inplace=True)

# ── 3. ENCODE CATEGORICALS ───────────────────────────────────
le = LabelEncoder()
for col in ['Fuel_Type', 'Selling_type', 'Transmission']:
    df[col] = le.fit_transform(df[col]) # type: ignore

# ── 4. CHART 1 — Correlation Heatmap ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 7)) # type: ignore
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', # type: ignore
            mask=mask, linewidths=0.5, ax=ax,
            annot_kws={'size': 10, 'weight': 'bold'})
ax.set_title('🚗 Car Price — Feature Correlation Heatmap', # type: ignore
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('car_correlation_heatmap.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: car_correlation_heatmap.png")

# ── 5. CHART 2 — Selling Price Distribution ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5)) # type: ignore
fig.suptitle('🚗 Selling Price Distribution', fontsize=14, fontweight='bold') # type: ignore
axes[0].hist(df['Selling_Price'], bins=30, color='#4C72B0',
             edgecolor='white', alpha=0.85)
axes[0].set_title('Original Scale')
axes[0].set_xlabel('Selling Price (Lakhs)')
axes[0].set_ylabel('Count')
axes[0].grid(alpha=0.3)
axes[1].hist(np.log1p(df['Selling_Price']), bins=30, color='#DD8452',
             edgecolor='white', alpha=0.85)
axes[1].set_title('Log Scale')
axes[1].set_xlabel('log(Selling Price)')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('car_price_distribution.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: car_price_distribution.png")
