# ---------------------------------------------------------
# 1) Import necessary libraries
# ---------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 2) Load the Iris dataset and create a DataFrame
# ---------------------------------------------------------
iris = load_iris()
# Create DataFrame with numeric features (4 columns)
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# ---------------------------------------------------------
# 3) Prepare Data for Linear Regression Analysis
#    - Our goal: predict petal length.
#    - Drop 'petal length (cm)' from predictors.
# ---------------------------------------------------------
# X: predictors (all features except petal length)
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]
# y: target variable (petal length)
y = iris_df['petal length (cm)']

# =========================================================
# CASE i) Train the model on 20% of the samples
# =========================================================
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(
    X, y, train_size=0.2, random_state=42
)

# Create and train the Linear Regression model for 20% training data
lr_20 = LinearRegression()
lr_20.fit(X_train_20, y_train_20)

# Predict petal length on the test set
y_pred_20 = lr_20.predict(X_test_20)
# Calculate RMSE for 20% training
rmse_20 = np.sqrt(mean_squared_error(y_test_20, y_pred_20))

# Output LR parameters and RMSE for 20% training
print("----- Case i) 20% Training Samples -----")
print("Intercept:", lr_20.intercept_)
print("Coefficients:", lr_20.coef_)
print("RMSE:", rmse_20)

# Choose a sample from the test set (one that is NOT in training)
sample_index_20 = X_test_20.index[0]
# Reshape the sample for prediction
sample_20 = X_test_20.loc[[sample_index_20]]  # Note the double brackets to keep it as a DataFrame
predicted_value_20 = lr_20.predict(sample_20)[0]

# sample_20 = X_test_20.loc[sample_index_20].values.reshape(1, -1)
# predicted_value_20 = lr_20.predict(sample_20)[0]
actual_value_20 = y_test_20.loc[sample_index_20]

print("\nFor sample index {}:".format(sample_index_20))
print("Actual petal length (cm):", actual_value_20)
print("Predicted petal length (cm):", predicted_value_20)
print("\n")

# =========================================================
# CASE ii) Train the model on 80% of the samples
# =========================================================
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# Create and train the Linear Regression model for 80% training data
lr_80 = LinearRegression()
lr_80.fit(X_train_80, y_train_80)

# Predict petal length on the test set
y_pred_80 = lr_80.predict(X_test_80)
# Calculate RMSE for 80% training
rmse_80 = np.sqrt(mean_squared_error(y_test_80, y_pred_80))

# Output LR parameters and RMSE for 80% training
print("----- Case ii) 80% Training Samples -----")
print("Intercept:", lr_80.intercept_)
print("Coefficients:", lr_80.coef_)
print("RMSE:", rmse_80)

# Choose a sample from the test set (one that is NOT in training)
sample_index_80 = X_test_80.index[0]
sample_80 = X_test_80.loc[[sample_index_80]]  # Note the double brackets to keep it as a DataFrame
predicted_value_80 = lr_80.predict(sample_80)[0]

# sample_80 = X_test_80.loc[sample_index_80].values.reshape(1, -1)
# predicted_value_80 = lr_80.predict(sample_80)[0]
actual_value_80 = y_test_80.loc[sample_index_80]

print("\nFor sample index {}:".format(sample_index_80))
print("Actual petal length (cm):", actual_value_80)
print("Predicted petal length (cm):", predicted_value_80)


# What Do These Results Tell Us?
# Similar Relationships:

# Both models have similar intercepts and coefficients. This indicates that the relationship between the predictors (sepal length, sepal width, petal width) and the target (petal length) is consistent regardless of whether you use 20% or 80% of the data for training.
# RMSE Comparison:

# The RMSE values are very close: about 0.33 for 20% training and 0.36 for 80% training. In this particular split, the model trained on 20% of the data produced a slightly lower RMSE, meaning its predictions were marginally closer on average. However, these differences are small and could be due to random variation in the split.
# Prediction on Sample Index 73:

# In both cases, for the chosen sample (index 73), the predicted petal lengths are slightly lower than the actual value of 4.7 cm. The predictions are 4.144 cm (20% case) and 4.128 cm (80% case), which again are very close to each other.
# Simple Summary
# Both models learn almost the same relationship between the features and petal length.
# Slight differences in RMSE (0.33 vs. 0.36) show that, in this case, the model with 20% training data performed a tiny bit better on average—but the difference is so small that both models are quite similar.
# For the specific sample (index 73), both models underestimated the petal length by about 0.55 to 0.57 cm.
# In essence, these results illustrate that even with different training sizes, the learned relationships and prediction errors are very similar. The small differences might come from the specific random splits used for training and testing.