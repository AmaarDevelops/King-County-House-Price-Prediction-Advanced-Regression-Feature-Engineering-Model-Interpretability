import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import shap

warnings.filterwarnings('ignore')

#  Load Data
df = pd.read_csv('kc_house_data.csv', encoding='latin1')

#  Initial Data Inspection & Basic Cleaning (EDA) 
df.info()
print("\n--- DataFrame Description ---")
print(df.describe())
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# --- 3. Feature Engineering ---
df['date'] = pd.to_datetime(df['date'])
df['sale_year'] = df['date'].dt.year
df['sale_month'] = df['date'].dt.month
df['sale_day_of_week'] = df['date'].dt.dayofweek
df['sale_quarter'] = df['date'].dt.quarter

current_year = 2025
df['house_age'] = current_year - df['yr_built']
df['years_since_renovation'] = np.where(
    df['yr_renovated'] == 0,
    0,
    current_year - df['yr_renovated']
)
df['years_since_renovation'] = np.where(
    df['years_since_renovation'] < 0,
    0,
    df['years_since_renovation']
)

central_lat = df['lat'].mean()
central_lon = df['long'].mean()
df['distance_to_center'] = np.sqrt(
    (df['lat'] - central_lat)**2 + (df['long'] - central_lon)**2
)

zip_coords = df.groupby('zipcode')[['lat','long']].mean().reset_index()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
zip_coords['region'] = kmeans.fit_predict(zip_coords[['lat','long']])
zip_coords_map = dict(zip(zip_coords['zipcode'], zip_coords['region']))
df['region'] = df['zipcode'].map(zip_coords_map)

df['sqft_living_per_lot'] = df['sqft_living'] / (df['sqft_lot'] + 1e-6)
df['bathrooms_per_bedroom'] = df['bathrooms'] / (df['bedrooms'].replace(0, np.nan))
df['bathrooms_per_bedroom'] = df['bathrooms_per_bedroom'].fillna(0)
df['total_square_footage'] = df['sqft_above'] + df['sqft_basement']

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = ['sqft_living', 'grade', 'house_age']
df_poly_features = poly.fit_transform(df[poly_features])
poly_feature_names = poly.get_feature_names_out(poly_features)
df_poly = pd.DataFrame(df_poly_features, columns=poly_feature_names, index=df.index)
df = pd.concat([df, df_poly], axis=1)

#  Prepare Data for Modeling 
df = df.drop(['id', 'date', 'yr_built', 'yr_renovated', 'lat', 'long'], axis=1)

x = df.drop('price', axis=1)
y = df['price']

categorical_cols_ohe = ['zipcode', 'waterfront', 'region']
categorical_cols_ohe = [col for col in categorical_cols_ohe if col in x.columns]

ordinal_features = ['view', 'condition', 'grade']
ordinal_features = [col for col in ordinal_features if col in x.columns]

if ordinal_features:
    encoder = OrdinalEncoder(categories='auto')
    x[ordinal_features] = encoder.fit_transform(x[ordinal_features])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

if categorical_cols_ohe:
    combined_ohe_data = pd.concat([x_train[categorical_cols_ohe], x_test[categorical_cols_ohe]], axis=0)
    combined_ohe_encoded = pd.get_dummies(combined_ohe_data, columns=categorical_cols_ohe, drop_first=True)
    x_train_ohe = combined_ohe_encoded.loc[x_train.index]
    x_test_ohe = combined_ohe_encoded.loc[x_test.index]
    x_train = x_train.drop(columns=categorical_cols_ohe)
    x_test = x_test.drop(columns=categorical_cols_ohe)
    x_train_final = pd.concat([x_train, x_train_ohe], axis=1)
    x_test_final = pd.concat([x_test, x_test_ohe], axis=1)
else:
    x_train_final = x_train
    x_test_final = x_test

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_final)
x_test_scaled = scaler.transform(x_test_final)

feature_names = x_train_final.columns

# --- 5. Model Training & Evaluation ---

print("\n--- Linear Regression ---")
lg = LinearRegression()
lg.fit(x_train_scaled, y_train)
lg_y_pred = lg.predict(x_test_scaled)
rmse_lg = np.sqrt(mean_squared_error(y_test, lg_y_pred))
mae_lg = mean_absolute_error(y_test, lg_y_pred)
r2_lg = r2_score(y_test, lg_y_pred)
print(f'RMSE: {rmse_lg:.2f}')
print(f'MAE: {mae_lg:.2f}')
print(f'R-squared: {r2_lg:.4f}')

print("\n--- Ridge Regression ---")
rr = Ridge(alpha=1.0)
rr.fit(x_train_scaled, y_train)
rr_y_pred = rr.predict(x_test_scaled)
rmse_rr = np.sqrt(mean_squared_error(y_test, rr_y_pred))
mae_rr = mean_absolute_error(y_test, rr_y_pred)
r2_rr = r2_score(y_test, rr_y_pred)
print(f'RMSE: {rmse_rr:.2f}')
print(f'MAE: {mae_rr:.2f}')
print(f'R-squared: {r2_rr:.4f}')

print("\n--- Random Forest Regressor (Initial) ---")
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train_scaled, y_train)
y_pred_rf = rf.predict(x_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'RMSE: {rmse_rf:.2f}')
print(f'MAE: {mae_rf:.2f}')
print(f'R-squared: {r2_rf:.4f}')

print("\n--- XGBoost Regressor ---")
xg_boost = XGBRegressor(random_state=42)
xg_boost.fit(x_train_scaled, y_train)
xg_y_pred = xg_boost.predict(x_test_scaled)
rmse_xg = np.sqrt(mean_squared_error(y_test, xg_y_pred))
mae_xg = mean_absolute_error(y_test, xg_y_pred)
r2_xg = r2_score(y_test, xg_y_pred)
print(f'RMSE: {rmse_xg:.2f}')
print(f'MAE: {mae_xg:.2f}')
print(f'R-squared: {r2_xg:.4f}')

print("\n--- Gradient Boosting Regressor ---")
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(x_train_scaled, y_train)
y_pred_gbr = gbr.predict(x_test_scaled)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)
print(f'RMSE: {rmse_gbr:.2f}')
print(f'MAE: {mae_gbr:.2f}')
print(f'R-squared: {r2_gbr:.4f}')

print("\n--- Hyperparameter Tuning (RandomForestRegressor) ---")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                              param_grid=param_grid_rf,
                              cv=3,
                              n_jobs=-1,
                              scoring='neg_mean_squared_error',
                              verbose=1)

grid_search_rf.fit(x_train_scaled, y_train)

best_rf = grid_search_rf.best_estimator_
print(f"Best parameters for RandomForest: {grid_search_rf.best_params_}")

best_rf_y_pred = best_rf.predict(x_test_scaled)

rmse_best_rf = np.sqrt(mean_squared_error(y_test, best_rf_y_pred))
mae_best_rf = mean_absolute_error(y_test, best_rf_y_pred)
r2_best_rf = r2_score(y_test, best_rf_y_pred)

print(f'Tuned RandomForest RMSE: {rmse_best_rf:.2f}')
print(f'Tuned RandomForest MAE: {mae_best_rf:.2f}')
print(f'Tuned RandomForest R-squared: {r2_best_rf:.4f}')

# --- Model Interpretability with SHAP (for Tuned RandomForest) ---
print("\n--- Model Interpretability with SHAP (for Tuned RandomForest) ---")
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(x_test_scaled)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, x_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Average Absolute SHAP Value)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, x_test_scaled, feature_names=feature_names, show=False)
plt.title("SHAP Feature Impact (Beeswarm Plot)")
plt.tight_layout()
plt.show()


# --- Visualizations (Expanded) ---
print("\n--- Visualizing Distributions of Numerical Features ---")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if col != 'price':
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()


print("\n--- Visualizing Feature Correlations with Price ---")
final_numerical_features_for_corr = df.select_dtypes(include=np.number).columns.tolist()
if 'id' in final_numerical_features_for_corr:
    final_numerical_features_for_corr.remove('id')

plt.figure(figsize=(12, 10))
sns.heatmap(df[final_numerical_features_for_corr].corr(),
            annot=False,
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Features (and Price)")
plt.show()


print("\n--- Residual Plot for Tuned RandomForest Regressor ---")
plt.figure(figsize=(10, 6))
residuals = y_test - best_rf_y_pred
plt.scatter(best_rf_y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Values (Tuned Random Forest)')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True)
plt.show()


print("\n--- Predicted vs. Actual Prices Plot (Tuned RandomForest Regressor) ---")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_rf_y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs. Actual Prices (Tuned Random Forest)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.grid(True)
plt.show()


print("\n Project tasks complete! Review plots and numerical results.")