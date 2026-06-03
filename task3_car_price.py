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


# ── 6. CHART 3 — Price by Fuel Type ─────────────────────────
# Decode for visualization
df_vis = df.copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # type: ignore
fig.suptitle('🚗 Price Analysis by Category', fontsize=14, fontweight='bold') # type: ignore
# Price by owner count
sns.boxplot(data=df_vis, x='Owner', y='Selling_Price',
            ax=axes[0], palette='Set2')
axes[0].set_title('Price by Number of Owners')
axes[0].set_xlabel('Owners')
axes[0].set_ylabel('Selling Price (Lakhs)')
axes[0].grid(alpha=0.3)
# Price by transmission
sns.boxplot(data=df_vis, x='Transmission', y='Selling_Price',
            ax=axes[1], palette='Set3')
axes[1].set_title('Price by Transmission Type (0=Auto, 1=Manual)')
axes[1].set_xlabel('Transmission')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('car_price_by_category.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: car_price_by_category.png")

# ── 7. TRAIN / TEST SPLIT ────────────────────────────────────
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split( # type: ignore
    X, y, test_size=0.2, random_state=42)

# ── 8. TRAIN MODELS ──────────────────────────────────────────
models = { # type: ignore
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
}
results = {}
for name, model in models.items(): # type: ignore
    model.fit(X_train, y_train) # type: ignore
    pred = model.predict(X_test) # type: ignore
    results[name] = {
        'model': model, 'pred': pred,
        'R2':   round(r2_score(y_test, pred), 4), # type: ignore
        'MAE':  round(mean_absolute_error(y_test, pred), 4), # type: ignore
        'RMSE': round(np.sqrt(mean_squared_error(y_test, pred)), 4), # type: ignore
    }
    print(f"\n{'='*40}")
    print(f"  Model: {name}")
    print(f"  R²   : {results[name]['R2']}")
    print(f"  MAE  : {results[name]['MAE']}")
    print(f"  RMSE : {results[name]['RMSE']}")

# Best model = Random Forest
best_pred = results['Random Forest']['pred'] # type: ignore
best_model = results['Random Forest']['model'] # type: ignore

# ── 9. CHART 4 — Actual vs Predicted ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 6)) # type: ignore
ax.scatter(y_test, best_pred, alpha=0.7, color='#4C72B0', # type: ignore
           edgecolors='white', linewidth=0.5, s=60)
mn, mx = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max()) # type: ignore
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction') # type: ignore
ax.set_title('🚗 Actual vs Predicted — Random Forest', # type: ignore
             fontsize=13, fontweight='bold')
ax.set_xlabel('Actual Selling Price (Lakhs)') # type: ignore
ax.set_ylabel('Predicted Selling Price (Lakhs)') # type: ignore
ax.legend() # type: ignore
ax.grid(alpha=0.3) # type: ignore
r2 = results['Random Forest']['R2'] # type: ignore
ax.text(0.05, 0.92, f'R² = {r2}', transform=ax.transAxes, # type: ignore
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('car_actual_vs_predicted.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: car_actual_vs_predicted.png")

# ── 10. CHART 5 — Feature Importance ─────────────────────────
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True) # type: ignore
fig, ax = plt.subplots(figsize=(9, 5)) # type: ignore
colors = ['#C44E52' if v > importances.mean() else '#4C72B0'
          for v in importances.values]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('🚗 Feature Importance (Random Forest)', # type: ignore
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score') # type: ignore
for i, v in enumerate(importances.values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9) # type: ignore
ax.axvline(importances.mean(), color='black', linestyle='--', # type: ignore
           linewidth=1.2, label='Mean importance')
ax.legend() # type: ignore
ax.grid(axis='x', alpha=0.3) # type: ignore
plt.tight_layout()
plt.savefig('car_feature_importance.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: car_feature_importance.png")
