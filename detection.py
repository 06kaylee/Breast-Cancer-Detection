import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('breast-cancer-wisconsin.data')
dataset = dataset.replace('?', np.nan)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X)
X = imputer.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
results_arr = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
results_df = pd.DataFrame(data = results_arr, columns = ['Predicted', 'Actual'])
print(results_df.head())

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(f"Accuracy: {(accuracies.mean() * 100):.2f}%")
print(f"Standard Deviation: {(accuracies.std() * 100):.2f}%")