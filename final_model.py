import pandas as pd

# Load the dataset
data = pd.read_csv('D:/DS-Intern-Assignment-main/data/data.csv')

# Display shape, data types, and missing values
print("Shape of the dataset:", data.shape)
print("Data types:\n", data.dtypes)

# Convert timestamp column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# Check for any parsing issues
print("Null timestamps after conversion:", data['timestamp'].isnull().sum())

# List of columns to convert to float
columns_to_convert = [
    'equipment_energy_consumption',
    'lighting_energy',
    'zone1_temperature',
    'zone1_humidity',
    'zone2_temperature'
]

# Convert the columns to numeric, coercing any errors (invalid parsing) to NaN
data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')


print("Missing values:\n", data.isnull().sum())

# # Drop rows where equipment_energy_consumption is missing
# data = data.dropna(subset=['equipment_energy_consumption'])

# Remove rows with values < 5
data = data[data['equipment_energy_consumption'] >= 5]

# Basic statistical summary
print("Statistical Summary:\n", data.describe())

data.head()

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# Handle missing values
data = data.dropna()  # Alternatively, you can impute missing values if needed

# Outlier detection (Z-score)
z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
data = data[(z_scores < 3).all(axis=1)]  # Removing rows with z-scores > 3

# Drop rows where either variable is negative
data = data[(data['random_variable1'] >= 0) & (data['random_variable2'] >= 0)]

# Convert timestamp to useful features (hour, weekday, etc.)
data['hour'] = data['timestamp'].dt.hour
data['weekday'] = data['timestamp'].dt.weekday

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize target variable distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['equipment_energy_consumption'], kde=True)
plt.title('Distribution of Equipment Energy Consumption')
plt.xlabel('Energy Consumption')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(16,12))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Time-series trends (e.g., energy usage by hour/day)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.dayofweek

# Energy usage by hour
plt.figure(figsize=(10, 6))
sns.boxplot(x='hour', y='equipment_energy_consumption', data=data)
plt.title('Energy Usage by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Energy Consumption')
plt.show()

# Compare random_variable1/2 with the target
plt.figure(figsize=(10, 6))
sns.scatterplot(x='random_variable1', y='equipment_energy_consumption', data=data)
plt.title('Random Variable 1 vs Energy Consumption')
plt.xlabel('Random Variable 1')
plt.ylabel('Energy Consumption')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='random_variable2', y='equipment_energy_consumption', data=data)
plt.title('Random Variable 2 vs Energy Consumption')
plt.xlabel('Random Variable 2')
plt.ylabel('Energy Consumption')
plt.show()


# Create a copy
data_scaled = data.copy()

# Select columns that contain temperature, humidity, and the random variables
cols_to_scale = [col for col in data.columns if 
                 'temperature' in col or 
                 'humidity' in col or 
                 'random_variable' in col]

# Apply StandardScaler
scaler = StandardScaler()
data_scaled[cols_to_scale] = scaler.fit_transform(data_scaled[cols_to_scale])

# Check scaled data
print(data_scaled[cols_to_scale].describe())


# Feature selection based on correlation or feature importance
correlation_matrix = data.corr()
top_features = correlation_matrix['equipment_energy_consumption'].abs().sort_values(ascending=False).head(10)
print("Top features based on correlation:\n", top_features)

X = data_scaled[top_features.index]
Y = data_scaled['equipment_energy_consumption']

import numpy as np

y = np.log1p(Y)  # log(1 + y) to avoid log(0)

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define models and their hyperparameter grids
models = {
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    "XGBoost": {
        "estimator": XGBRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0],
            'gamma': [0, 1, 5],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
    },
    "Linear Regression": {
        "estimator": LinearRegression(),
        "params": {}  # No hyperparameters to tune here
    }
}

#Evaluate each model
results = []

for name, config in models.items():
    print(f"\n🔍 Tuning and evaluating: {name}")
    
    best_params_dict = {}

    if config["params"]:  # Tune if params exist
        search = RandomizedSearchCV(
            estimator=config["estimator"],
            param_distributions=config["params"],
            n_iter=10,
            scoring='r2',
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_params_dict[name] = best_params

    else:  # No tuning for LinearRegression
        best_model = config["estimator"]
        best_model.fit(X_train, y_train)
        best_params = "N/A"
        best_params_dict[name] = best_params


    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("\n🛠️ Best Hyperparameters for Each Model:")
    for model_name, params in best_params_dict.items():
        print(f"{model_name}: {params}")


    results.append({
        "Model": name,
        "R²": r2,
        "RMSE": rmse,
    })
    
#Display results
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Results:")
print(results_df.sort_values(by="R²", ascending=False))