import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle

# 1. Load the dataset
file_path = 'housing_prices.csv' 
df = pd.read_csv(file_path)

# Display the data to check its structure
print(df.head())

# 2. Data processing and checking for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# 3. Analyze correlation and select features
correlation_matrix = df.corr()

# Plot the heatmap to visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# 4. Split the data into training, validation, and test sets
X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the sizes of the datasets
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# 5. Standardize the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. Hyperparameter Tuning for Random Forest

# Define a parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize a Random Forest model
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model on the training set
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model
best_rf_model = grid_search.best_estimator_

# Test the best model
y_test_pred = best_rf_model.predict(X_test_scaled)

# Calculate MSE and R²
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the results
print(f"Mean Squared Error (MSE) on Test Set: {mse_test}")
print(f"R² Score on Test Set: {r2_test * 100:.2f}%")

# 7. Save the model and the scaler

# Save the trained model
with open('your_rf_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

# Save the scaler
with open('your_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and Scaler have been saved successfully!")