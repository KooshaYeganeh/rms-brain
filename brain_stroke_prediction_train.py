import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'healthcare-dataset-stroke-data.csv'  # Path to your CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the data to understand the structure
print(data.head())

# Assuming 'stroke' is the target variable and the rest are features
target_column = 'stroke'

# Separate features (X) and target variable (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# List of numerical columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
# - For numerical columns, impute missing values with the mean
# - For categorical columns, impute missing values with the most frequent value, then apply OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_columns),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Use sparse_output=False
        ]), categorical_columns)
    ])

# Apply preprocessing transformations to the data
# Ensure X is a DataFrame before passing to ColumnTransformer
X_processed = preprocessor.fit_transform(X)

# Convert the transformed X_processed back into a DataFrame for compatibility with later steps
# Get the names of the columns after OneHotEncoder has been applied
# For categorical columns, OneHotEncoder creates multiple columns
cat_columns = list(preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_columns))
columns = numerical_columns.tolist() + cat_columns  # Combine numerical and categorical column names

# Convert to DataFrame (after applying transformations)
X_processed = pd.DataFrame(X_processed, columns=columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize a classifier (RandomForestClassifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline that includes the classifier
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
