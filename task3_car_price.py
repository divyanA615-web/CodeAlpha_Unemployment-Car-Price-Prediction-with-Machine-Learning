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