from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('example_retail_sales.csv')

m = Prophet()
forcast = m.fit(df)
future = m.make_future_dataframe(periods=200)

future = m.predict(future)

plt.plot(future['ds'][280:], future['trend'][280:])
plt.plot(future['ds'][280:], future['yhat'][280:])
plt.plot(future['ds'][280:], future['yhat_lower'][280:])
plt.plot(future['ds'][280:], future['yhat_upper'][280:])

y_values = []
x_values = []

for i, row in df[280:].iterrows():
    y_values.append(future['ds'][i])
    x_values.append(row['y'])

plt.plot(y_values, x_values)

plt.show()