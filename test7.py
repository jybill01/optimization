from sklearn.datasets import fetch_california_housing
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

california_data = fetch_california_housing()

california = pd.DataFrame(data=california_data.data, columns=california_data.feature_names)
california['target'] = california_data.target

train = california.sample(frac=0.8, random_state=200)
test = california.drop(train.index)

scatter_matrix(california)
plt.show()