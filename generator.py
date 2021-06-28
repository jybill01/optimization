import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv("generator.csv")
sns.lmplot('RPM', 'VIBRATION', data=df, hue='STATUS', fit_reg=False)

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train[['RPM', 'VIBRATION']], train['STATUS'])
score = knn.score(test[['RPM', 'VIBRATION']], test['STATUS'])
print(score)

guess = pd.DataFrame(columns=['RPM', 'VIBRATION'])
guess.loc[0] = [800, 200]
print(knn.predict(guess))