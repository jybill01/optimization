from sklearn.datasets import load_boston
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

boston_data = load_boston()

boston = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
boston['target'] = boston_data.target

train = boston.sample(frac=0.8, random_state=200)
test = boston.drop(train.index)

scatter_matrix(boston.drop(columns=['B']))
plt.show()