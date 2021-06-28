import pandas as pd
from sklearn.linear_model import LogisticRegression

df1 = pd.read_csv('bank_marketing_full.csv', sep=';')
df2 = pd.read_csv('bank_marketing_full.csv', sep=';')

del df1["duration"]

del df2["duration"]

df1 = pd.get_dummies(df1, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome'])

df2 = pd.get_dummies(df2, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome'])
#print(df.shape)
#print(df.columns.tolist())

#train = df.sample(frac=0.8, random_state=200)
#test = df.drop(train.index)

train = df1
test = df2

train_y = train["y"]
del train["y"]
train_x = train

test_y = test["y"]
del test["y"]
test_x = test

logistic = LogisticRegression(solver='newton-cg')
logistic.fit(train_x, train_y)

score = logistic.score(test_x, test_y)
print(score)