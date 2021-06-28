import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#fruit_label,fruit_name,fruit_subtype,mass,width,height,color_score
x = 'width'
y = 'height'

df = pd.read_csv('fruit_data_with_colors.csv')
label_count = len(df['fruit_label'].unique())

sns.lmplot(x, y, data=df, hue='fruit_label', fit_reg=False)
plt.show()

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

knn = KNeighborsClassifier(n_neighbors=label_count)
knn.fit(train[[x, y]], train['fruit_label'])
score = knn.score(test[[x, y]], test['fruit_label'])
print(score)