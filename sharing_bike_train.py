import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


df = pd.read_csv('sharing_bike_train.csv')

df["year"] = pd.to_datetime(df["datetime"]).dt.year
df["month"] = pd.to_datetime(df["datetime"]).dt.month
df["day"] = pd.to_datetime(df["datetime"]).dt.day
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
df["weekday"] = pd.to_datetime(df["datetime"]).dt.weekday

df["year"] = df["year"].astype("category")
df["month"] = df["month"].astype("category")
df["day"] = df["day"].astype("category")
df["hour"] = df["hour"].astype("category")
df["weekday"] = df["weekday"].astype("category")

del df["datetime"]


#season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

"""
sns.pairplot(data=test[['weather','temp', 'atemp', 'humidity', 'windspeed']],hue='weather')
plt.show()
"""

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(train[['temp', 'humidity','atemp','humidity','windspeed']], train['weather'])
score = knn.score(test[['temp', 'humidity','atemp','humidity','windspeed']], test['weather'])
print(score)



