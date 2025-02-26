from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load iris and create dataframe 
iris = load_iris()
# 4 columns
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# prep for linear regression analysis
# X: predictors (all features except petal length)
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]
# y: target variable - petal length 
y = iris_df['petal length (cm)']

# case i) train the model on 20% of the samples
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(
    X, y, train_size=0.2, random_state=42
)

# 20% training data
lr_20 = LinearRegression()
lr_20.fit(X_train_20, y_train_20)

# predict petal length on the test set
y_pred_20 = lr_20.predict(X_test_20)
# calculate RMSE for 20% training
rmse_20 = np.sqrt(mean_squared_error(y_test_20, y_pred_20))

# LR parameters and RMSE for 20% training
print("Case i) 20% Training Samples:")
print("Intercept:", lr_20.intercept_)
print("Coefficients:", lr_20.coef_)
print("RMSE:", rmse_20)

# choose a sample from the test set that was not in training 
sample_index_20 = X_test_20.index[0]
# reshape 
sample_20 = X_test_20.loc[[sample_index_20]] #double blackets to keep as dataframe  
predicted_value_20 = lr_20.predict(sample_20)[0]

# sample_20 = X_test_20.loc[sample_index_20].values.reshape(1, -1)
# predicted_value_20 = lr_20.predict(sample_20)[0]
actual_value_20 = y_test_20.loc[sample_index_20]

print("\nFor sample index {}:".format(sample_index_20))
print("Actual petal length (cm):", actual_value_20)
print("Predicted petal length (cm):", predicted_value_20)
print("\n")

# case ii) train the model on 80% of the samples
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# 80% training data
lr_80 = LinearRegression()
lr_80.fit(X_train_80, y_train_80)

# predict petal length on the test set
y_pred_80 = lr_80.predict(X_test_80)
# calculate RMSE for 80% training
rmse_80 = np.sqrt(mean_squared_error(y_test_80, y_pred_80))

#  LR parameters and RMSE for 80% training
print("Case ii) 80% Training Samples")
print("Intercept:", lr_80.intercept_)
print("Coefficients:", lr_80.coef_)
print("RMSE:", rmse_80)

# choose a sample from the test set that was not in training 
sample_index_80 = X_test_80.index[0]
sample_80 = X_test_80.loc[[sample_index_80]]  #double blackets to keep as dataframe 
predicted_value_80 = lr_80.predict(sample_80)[0]

# sample_80 = X_test_80.loc[sample_index_80].values.reshape(1, -1)
# predicted_value_80 = lr_80.predict(sample_80)[0]
actual_value_80 = y_test_80.loc[sample_index_80]

print("\nFor sample index {}:".format(sample_index_80))
print("Actual petal length (cm):", actual_value_80)
print("Predicted petal length (cm):", predicted_value_80)

