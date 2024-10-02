# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
data = pd.read_csv("student-por.csv", sep=';')  # Portuguese student performance data

# Display basic information about the dataset
print(data.head())
print(data.info())

# Data Preprocessing
# Label Encoding for categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Define the target (final grade G3) and features
X = data.drop(['G3'], axis=1)  # Features
y = data['G3']  # Target (final grade)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model (using Mean Squared Error as the metric)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Feature Importance (optional - helps understand which features contributed most)
importances = model.feature_importances_
indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], color='b', align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
