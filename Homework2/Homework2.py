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

#visualize iris features as a heatmap
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
