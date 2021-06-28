import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('telecom_churn.csv')
df.dropna(inplace=True)

df2 = pd.read_csv('telecom_churn_test.csv')
df2.dropna(inplace=True)

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['Partner'] = le.fit_transform(df['Partner'])
df['Dependents'] = le.fit_transform(df['Dependents'])
df['PhoneService'] = le.fit_transform(df['PhoneService'])
df['MultipleLines'] = le.fit_transform(df['MultipleLines'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['OnlineSecurity'] = le.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = le.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = le.fit_transform(df['DeviceProtection'])
df['TechSupport'] = le.fit_transform(df['TechSupport'])
df['StreamingTV'] = le.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = le.fit_transform(df['StreamingMovies'])
df['Contract'] = le.fit_transform(df['Contract'])
df['PaperlessBilling'] = le.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])
df['Churn'] = le.fit_transform(df['Churn'])

df2['gender'] = le.fit_transform(df2['gender'])
df2['Partner'] = le.fit_transform(df2['Partner'])
df2['Dependents'] = le.fit_transform(df2['Dependents'])
df2['PhoneService'] = le.fit_transform(df2['PhoneService'])
df2['MultipleLines'] = le.fit_transform(df2['MultipleLines'])
df2['InternetService'] = le.fit_transform(df2['InternetService'])
df2['OnlineSecurity'] = le.fit_transform(df2['OnlineSecurity'])
df2['OnlineBackup'] = le.fit_transform(df2['OnlineBackup'])
df2['DeviceProtection'] = le.fit_transform(df2['DeviceProtection'])
df2['TechSupport'] = le.fit_transform(df2['TechSupport'])
df2['StreamingTV'] = le.fit_transform(df2['StreamingTV'])
df2['StreamingMovies'] = le.fit_transform(df2['StreamingMovies'])
df2['Contract'] = le.fit_transform(df2['Contract'])
df2['PaperlessBilling'] = le.fit_transform(df2['PaperlessBilling'])
df2['PaymentMethod'] = le.fit_transform(df2['PaymentMethod'])
df2['Churn'] = le.fit_transform(df2['Churn'])

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

logistic = LogisticRegression(solver='newton-cg')
logistic.fit(df[["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
                    "TotalCharges"]],
             df['Churn'])

score = logistic.score(df2[["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                             "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                             "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
                             "PaymentMethod", "MonthlyCharges", "TotalCharges"]],
                       df2['Churn'])

print(score)
