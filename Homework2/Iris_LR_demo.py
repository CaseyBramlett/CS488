# ---------------------------------------------------------
# 1) Import necessary libraries
# ---------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 2) Load the Iris dataset and create DataFrames
# ---------------------------------------------------------
iris = load_iris()

# Create a DataFrame for the features
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create a DataFrame for the target
target_df = pd.DataFrame(data=iris.target, columns=['species'])

# ---------------------------------------------------------
# 3) Generate labels (setosa, versicolor, virginica)
# ---------------------------------------------------------
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'

target_df['species'] = target_df['species'].apply(converter)

# ---------------------------------------------------------
# 4) Concatenate the feature DataFrame and target DataFrame
# ---------------------------------------------------------
iris_df = pd.concat([iris_df, target_df], axis=1)

# ---------------------------------------------------------
# 5) Quick statistics
# ---------------------------------------------------------
print("Iris DataFrame Statistics:")
print(iris_df.describe())
print("\n")

# ---------------------------------------------------------
# 6) Convert 'species' to numerical type by dropping it 
#    (since we want to predict sepal length)
# ---------------------------------------------------------
iris_df.drop(['species'], axis=1, inplace=True)

# ---------------------------------------------------------
# 7) Define X (independent variables) and y (dependent variable)
# ---------------------------------------------------------
X = iris_df.drop('sepal length (cm)', axis=1)
y = iris_df['sepal length (cm)']

# ---------------------------------------------------------
# 8) Split the dataset into training and testing sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# ---------------------------------------------------------
# 9) Create and train the Linear Regression model
# ---------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

# ---------------------------------------------------------
# 10) Make predictions on the test set
# ---------------------------------------------------------
y_pred = lr.predict(X_test)

# ---------------------------------------------------------
# 11) Quantitative analysis to evaluate LR performance
# ---------------------------------------------------------
print("LR beta/slope Coefficients:", lr.coef_)
print("LR intercept Coefficient:", lr.intercept_)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)

print(f"Coefficient of determination (R^2): {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Squared Error (MSE): {mse}")
print("\n")

# ---------------------------------------------------------
# 12) Predict on a new data point (or just show an example 
#     from X_test) and compare to an 'actual' value
# ---------------------------------------------------------
pred = lr.predict(X_test)
print("Predicted Sepal Length (cm):", pred[0])
print("Actual Sepal Length (cm):", 5.4)
