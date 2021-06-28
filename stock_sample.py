from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('stock_sample.csv')
del df['Open']
del df['High']
del df['Low']
del df['Volume']
del df['Adj Close']

df = df.rename(columns={
    'Date': 'ds',
    'Close': 'y',
})

m = Prophet(daily_seasonality=True)
forecast = m.fit(df)
future = m.make_future_dataframe(periods=10)

future = m.predict(future)

y_values = []
x_values = []
for i, row in df[190:].iterrows():
    y_values.append(future['ds'][i])
    x_values.append(row['y'])

plt.plot(y_values, x_values)
plt.plot(future['ds'][190:], future['trend'][190:])
plt.plot(future['ds'][190:], future['yhat'][190:])
plt.plot(future['ds'][190:], future['yhat_lower'][190:])
plt.plot(future['ds'][190:], future['yhat_upper'][190:])

plt.show()