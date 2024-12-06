import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # for saving the model

# Load the dataset
data = pd.read_csv("parkinsons_data.csv")

# Split the dataset into features and target
X = data.drop(columns=["status"])  # Drop 'status' column as it is the target
y = data["status"]  # Target variable ('status' column)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train_scaled, y_train)

# Save the model and scaler to disk
joblib.dump(clf, 'parkinsons_model.pkl')
joblib.dump(scaler, 'parkinson_scaler.pkl')

# Predict on the test set and evaluate the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

