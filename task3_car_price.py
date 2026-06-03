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