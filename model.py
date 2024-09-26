import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
aqi_df = pd.read_csv('Air_quality_index.csv')

# Handle missing values (for simplicity, we will drop them)
aqi_df = aqi_df.dropna()

# Perform label encoding on categorical columns
categorical_columns = ['country', 'state', 'city', 'station', 'pollutant_id']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    aqi_df[column] = le.fit_transform(aqi_df[column])
    label_encoders[column] = le  # Store the encoder if needed for inverse transformation later

# Select relevant features for clustering
features_for_clustering = aqi_df[['pollutant_avg', 'pollutant_min', 'pollutant_max']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_clustering)

# Apply K-Means clustering
n_clusters = 3  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
aqi_df['cluster'] = kmeans.fit_predict(features_scaled)

# Define features and target variable
X = aqi_df[['pollutant_min', 'pollutant_max', 'pollutant_avg']]
y = aqi_df['cluster']  # Use the cluster labels as the target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost Regressor
XGBC_model = XGBRegressor(n_estimators=100, random_state=42)
XGBC_model.fit(X_train, y_train)

# Make predictions
y_pred = XGBC_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

from sklearn.metrics import mean_squared_error

# Predictions for training data
y_train_pred = XGBC_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Training MSE:", train_mse)

# Predictions for testing data
y_test_pred = XGBC_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Testing MSE:", test_mse)

print(aqi_df)
