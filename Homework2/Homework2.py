from sklearn.datasets import load_iris
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load iris data
iris = load_iris()

#create pd dataframes
iris_df = pd.DataFrame(data=iris.data, columns = iris.feature_names)
target_df = pd.DataFrame(data= iris.target, columns= ['species'])

# generate labels
def converter(specie): 
    if specie ==0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virgincia'
target_df['species'] = target_df['species'].apply(converter)

#concat the dataframes
df = pd.concat([iris_df, target_df], axis=1)

#display data headers
#df.head

df.info()

#display random data samples
df.sample(10)

#display data columns 
df.columns


#disaply the size of the dataset
df.shape

#output data

print(df)

#display correlation coeff
# df.corr()


#visualize iris deatures as a heatmap
# cor_eff=df.corr()
cor_eff = df.select_dtypes(include=[np.number]).corr()
print(cor_eff)


# visualize the correlation heatmap
plt.figure(figsize=(6,6))
sns.heatmap(cor_eff, linecolor='white', linewidths=1, annot=True)
# plt.show()  # display the heatmap

# plot the lower half of the correlation matrix
fig, ax = plt.subplots(figsize=(6,6))
mask = np.zeros_like(cor_eff)
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(cor_eff, linecolor='white', linewidths=1, mask=mask, ax=ax, annot=True)
# plt.show()  # display the lower half heatmap

# create the pairplot
g = sns.pairplot(df, hue='species')
# plt.show()  # display the pairplot
plt.show() #show all plots at once


### 1B data analysis
# implications of data distribution on data analysis 
# from the histograms and pairplots we can tell that 
# petal length and petal width have very distinict distrivustions for each species. Particularly 'setosa'
# typicall has smaller petals compared to 'versicolor' and virginica
# Sepal width shows more overlap among the 3 species, meaning it could make
#  less discriminitive feature in classification tasks
# All of these distrivutions suggest that setosa is more easily seperable from the other two species based on
# petal distributions, where as the other two have more overlap
#
#In terms of implications:

#Model Selection and Complexity: 
# Highly correlated features (like petal length and petal width) may introduce collinearity issues in linear models. 
# Dimensionality reduction or feature selection might be considered if we need to simplify the model.
#Data Preprocessing:
#  The distinct distributions could allow simpler models to classify setosa well, but distinguishing versicolor 
# from virginica may require more nuanced approaches or more robust feature engineering.
#
#
#(b) Inferences from (i) Correlation Heat Map and (ii) Feature Plots
#Correlation Heat Map

#Petal length and petal width show a very high positive correlation (~0.96). 
# This implies these two features largely move together.

#Sepal length also correlates strongly with both petal length (~0.87) and petal width (~0.82).
#  This suggests a degree of redundancy among these three features if used in certain predictive models.
#Sepal width is negatively correlated with the other features, although the magnitude of correlation is weaker.


#Influence on Analysis:

#Because of the strong correlation among petal dimensions (and with sepal length), 
# you might not need all three features (petal length, petal width, sepal length) if your goal is a simpler model. 
# On the other hand, if interpretability or slight performance gains matter, you could still keep them.


#Feature Plots (Pairplot)

#The pairplot reveals that setosa points are clustered distinctly, especially along petal length and petal width.
#Versicolor and virginica overlap more in sepal-related features but separate more distinctly on 
# the petal-related features.

#The distributions along the diagonal in the pairplot also confirm that petal measurements 
# provide clearer separations than sepal measurements for distinguishing species.


#Influence on Analysis:
#
#Models aiming to separate setosa from the other classes can rely heavily on petal dimensions.
#Distinguishing versicolor and virginica may require more nuanced use of multiple features or 
# potentially non-linear decision boundaries.
#Overall, the data visualizations suggest that petal features are highly informative for species identification, 
# while sepal width is less correlated with the other dimensions and may provide additional,
#  albeit weaker, discriminative power. These insights can guide model design and feature selection
#  in subsequent predictive tasks.


#test