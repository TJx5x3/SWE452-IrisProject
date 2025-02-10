import unittest
from sklearn.datasets import load_iris          
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TestIrisPipeline(unittest.TestCase):
    def setUp(self):
        self.iris = load_iris()
        self.X = self.iris.data # X is uppercase bc X is a matrix, y is a vector
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names

    def test_data_loading(self):
        self.assertEqual(self.X.shape, (150,4)) # 150 samples, 4 features
        self.assertEqual(self.y.shape, (150,))
        self.assertEqual(len(self.feature_names), 4) # 4 feature sepal/petal width & length
        self.assertEqual(len(self.target_names), 3) # target names are the 3 species

    def test_missing_and_duplicates(self): # NOTE: this tests two things, unit tests should only test one thing
        data = pd.DataFrame(self.X, columns=self.feature_names)
        self.assertTrue(data.isnull().sum().sum() == 0) # check for no null answers - sum columns and rows
        
        duplicate_count = data.duplicated().sum()

        if duplicate_count > 0:
            print(f"Number of duplicate rows: {duplicate_count}")
            data.drop_duplicates(inplace=True)

        self.assertEqual(data.duplicated().sum(), 0)

    def test_data_splitting(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) 
        self.assertEqual(len(X_train), 120) # about 80% of data
        self.assertEqual(len(X_test), 30)
    
    def test_scaling(self): # Underscores state function does not need that data
        X_train, X_test, _, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42) 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.assertAlmostEqual(X_train_scaled.mean(), 0, delta=0.1) # This data should be approx 0 
        self.assertAlmostEqual(X_train_scaled.std(), 1, delta=0.1) # Should be approx. 1


    def test_model_training_and_evaluation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build the model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Test the model's accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.9) # 90% or more

        # Test confusion matrix - typically don't do this in unit tests
        cm = confusion_matrix(y_test, y_pred)
        self.assertEqual(cm.shape, (3,3))

if __name__ == '__main__':
    unittest.main()
