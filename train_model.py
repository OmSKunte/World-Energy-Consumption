import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

df = pd.read_csv("/content/Cleaned_Energy_Consumption.csv")

X = df.drop(columns=["EnergyConsumption"])
y = df["EnergyConsumption"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
accuracy_percent = r2 * 100

print(f"Model trained successfully!")
print(f"Accuracy: {accuracy_percent:.2f}%")

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("model.pkl has been saved.")
