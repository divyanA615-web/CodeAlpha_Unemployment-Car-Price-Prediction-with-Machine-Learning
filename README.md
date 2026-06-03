🚗 Car Price Prediction with Machine Learning

End-to-end Exploratory Data Analysis (EDA) + Regression Modelling to predict used car selling prices.
Built with Python · Pandas · Seaborn · Scikit-Learn


📁 Repository Structure
CodeAlpha_Car-Price-Prediction-with-Machine-Learning/
│
├── car_data.csv                     # Dataset (301 rows × 9 columns)
├── task3_car_price.py               # Full EDA + ML pipeline
│
├── car_price_distribution.png       # Selling price histogram (original + log scale)
├── car_correlation_heatmap.png      # Feature correlation heatmap
├── car_price_by_category.png        # Price by Owners & Transmission type
├── car_actual_vs_predicted.png      # Actual vs Predicted scatter plot
├── car_feature_importance.png       # Random Forest feature importance
├── car_residuals.png                # Residuals vs Predicted + distribution
├── car_model_comparison.png         # Model R² and MAE comparison bar chart
│
└── README.md

📋 Dataset Overview
PropertyValueSourceCodeAlpha Internship DatasetRows301Columns9Missing ValuesNoneTargetSelling_Price (in Lakhs ₹)
Column Details
ColumnTypeRoleCar_NameStringDropped (too many unique values)YearIntegerEngineered → Car_AgeSelling_PriceFloat🎯 Target VariablePresent_PriceFloatFeatureDriven_kmsIntegerFeatureFuel_TypeStringLabel Encoded (Petrol/Diesel/CNG)Selling_typeStringLabel Encoded (Dealer/Individual)TransmissionStringLabel Encoded (Manual/Auto)OwnerIntegerFeature

⚙️ Feature Engineering
TransformationDetailCar_Age2024 - Year — more meaningful than raw yearDrop Car_NameToo many unique values, no signalLabel EncodingApplied to Fuel_Type, Selling_type, Transmission

🔍 Exploratory Data Analysis (EDA)
1. Selling Price Distribution
Show Image

The distribution is right-skewed — most cars are priced under ₹10 Lakhs, with a few luxury outliers above ₹30 Lakhs. The log-transformed version shows a near-normal distribution, confirming the skew.


2. Feature Correlation Heatmap
Show Image

Present_Price has the strongest positive correlation with Selling_Price (r ≈ 0.88), making it the most influential raw feature. Driven_kms shows a slight negative correlation — more kilometres driven means lower resale value.


3. Price by Category
Show Image

Cars with 0 previous owners command significantly higher prices. Automatic transmission cars have a wider and higher price range compared to manual cars, reflecting premium positioning.


🤖 Machine Learning Models
Model Configuration
ParameterValueTrain / Test80% / 20%Test Samples~60 carsrandom_state42RF n_estimators100
Models Trained
ModelPurposeLinear RegressionBaseline model, interpretableRandom Forest ✅Best performer, handles non-linearity
Model Results
ModelR² ScoreMAERMSELinear Regression~0.85~1.20~2.10Random Forest~0.96~0.65~1.10

4. Actual vs Predicted
Show Image

Points closely follow the perfect prediction line (red dashed), confirming the Random Forest model captures price patterns well. Slight scatter exists for high-value cars (sparse training samples).


5. Feature Importance
Show Image

Present_Price and Car_Age are the two dominant predictors — together accounting for over 85% of the model's decision making. Fuel_Type and Transmission contribute less but still add value.


6. Residuals Analysis
Show Image

Residuals are randomly scattered around zero — no systematic bias. The residual distribution is approximately bell-shaped, confirming good model fit and no major violations of regression assumptions.


7. Model Comparison
Show Image

Random Forest outperforms Linear Regression significantly — R² improves from ~0.85 to ~0.96, and MAE drops by ~45%. This confirms that car pricing has non-linear relationships that tree-based models capture effectively.


💡 Key Insights
#Insight1🏷️ Present_Price is the strongest predictor of Selling_Price (r ≈ 0.88)2📅 Car_Age is the second most important feature — older cars sell for less3🔑 First-owner cars command a premium over multi-owner vehicles4⚙️ Automatic transmission cars have a higher and wider price range5🌲 Random Forest achieves ~96% R², far better than Linear Regression6📉 Residuals are normally distributed — model assumptions are well-satisfied

🚀 How to Run
1. Clone the Repository
bashgit clone https://github.com/divyanA615-web/CodeAlpha_Car-Price-Prediction-with-Machine-Learning.git
cd CodeAlpha_Car-Price-Prediction-with-Machine-Learning
2. Install Dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn
3. Run the Analysis
bashpython task3_car_price.py
4. Output
All 7 charts saved as .png files + model metrics printed in terminal.

🛠️ Technologies Used
ToolPurposePython 3.xCore languagePandasData loading, cleaning, engineeringNumPyNumerical computationMatplotlibChart renderingSeabornStatistical visualizationsScikit-LearnML models, metrics, preprocessing

🌐 Project Context
This project was completed as Task 3 of the CodeAlpha Data Science Internship.
TaskProjectLinkTask 1Iris Flower Classification🔗 View RepoTask 2Unemployment Analysis🔗 View RepoTask 3Car Price Prediction✅ This RepoTask 4Sales Prediction🔗 View Repo

👤 Author
DivyanA615-web
GitHub: github.com/divyanA615-web

⭐ If you found this useful, please star the repository!
