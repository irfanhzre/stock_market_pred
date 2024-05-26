import streamlit as st
pip install scikit-learn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


# Step 1: Load your dataset
# Assuming 'data.csv' contains your dataset
df = pd.read_csv('data.csv')

# Assuming X is your feature matrix and y is your target variable
X = df.drop(columns=['target_column_name'])
y = df['target_column_name']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Gradient Boosting regression model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  
gb_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred_gb = gb_model.predict(X_test)

# Step 6: Evaluate the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = mse_gb ** 0.5
r2_gb = r2_score(y_test, y_pred_gb)

# Create a Streamlit UI
st.title('Gradient Boosting Regression Model')
st.write('Mean Squared Error:', mse_gb)
st.write('Root Mean Squared Error:', rmse_gb)
st.write('R-squared:', r2_gb)
