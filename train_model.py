import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
df_train = pd.read_csv("train.csv")

# Prepare features and target
X = df_train[['GrLivArea', 'BedroomAbvGr', 'FullBath']].fillna(0)
y = df_train['SalePrice'].fillna(0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)

print(r2)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(mse)
print(rmse)

#  Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
