
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load student data
data = pd.read_csv("students.csv")

# Step 2: Select input features and target
X = data[
    ["attendance", "internal_marks",
     "assignment_score", "study_hours"]
]
y = data["performance"]

# Step 3: Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Step 4: Create the Machine Learning model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Test the model
predictions = model.predict(X_test)

print("Model training completed!")
print("Test accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Step 7: Save the trained model
joblib.dump(model, "student_model.pkl")

print("\nModel saved as student_model.pkl")
