#----------------------------------
# AUTHOR | Ryan Woodward
# CLASS  | SWE-452
#----------------------------------

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
iris = load_iris()

# Features (input data)
X = iris.data

# Target (output data)
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Features:\n", feature_names)
print("Target classes:\n", target_names)
print("First 5 samples:\n", X[:5]) 

data = pd.DataFrame(iris.data, columns=feature_names)

# Check for missing values
print("\nMissing values per feature:\n", data.isnull().sum()) # Print sum of each feature, how many values are there

# Check for duplicates
print("Number of duplicate rows: ", data.duplicated().sum())

# Remove duplicates
data.drop_duplicates(inplace=True)

# Check for missing values
print("\nMissing values per feature:\n", data.isnull().sum()) # Print sum of each feature, how many values are there

# Check for duplicates
print("Number of duplicate rows: ", data.duplicated().sum())


#! Data Preprocessing -> Standardize Features
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42) # 20% of data should be used for testing, this format (multiple variables to one call) is called "tuple"

# Initialize standard scalar. 
scaler = StandardScaler()

# Fit the scalar on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# This will be a much smaller range of values
print("\nFirst 5 rows of scaled trained data:\n", X_train_scaled[:5])

#! Step 3: Train and evaluate the logistic regression model
model = LogisticRegression()

# Fit data to model
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Generate a classification report
# ("If you're using a classification model, I want to see this report") - from Prof. Woodward
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))


# confusion matrix for nice visual
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names) # fmt = format of something, annot = annotation
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()